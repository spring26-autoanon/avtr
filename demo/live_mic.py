#!/usr/bin/env python3
"""
MoshiRAG live mic demo.

Usage:
  uv run demo/live_mic.py --config configs/baseline_with_retrieval.yaml
  uv run demo/live_mic.py --config configs/baseline_with_retrieval.yaml --record

Captures mic audio, detects end-of-utterance via VAD, runs MoshiRAG inference,
and plays back the model's audio response. Input capture and output playback run
on separate sounddevice streams so the model can respond while you're ready to
speak again.
"""

import argparse
import datetime
import io
import queue
import struct
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.retrieval_backend import GeminiAPIBackend, NullBackend

SAMPLE_RATE = 24_000
CHUNK_SAMPLES = 1920          # 80ms at 24kHz — matches moshi's 12.5Hz step rate
SILENCE_RMS_THRESHOLD = 0.01  # float32 amplitude; tune for mic gain
SILENCE_CHUNKS_NEEDED = 10    # 10 × 80ms = 800ms silence → end of utterance
MIN_SPEECH_CHUNKS = 4         # ignore utterances shorter than 320ms


def _rms(chunk) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(chunk ** 2)))


def _f32_to_wav(pcm_f32, sample_rate: int = SAMPLE_RATE) -> bytes:
    import numpy as np
    pcm = (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype(np.int16)
    data = pcm.tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return buf.getvalue()


def _wav_to_f32(wav_bytes: bytes):
    import numpy as np
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def _save_wav(path: Path, wav_bytes: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes)


def _build_model(cfg: dict):
    model_cfg = cfg.get("model", {})
    retrieval_cfg = model_cfg.get("retrieval", {})
    if retrieval_cfg.get("enabled"):
        backend = GeminiAPIBackend(
            model=retrieval_cfg.get("model", "gemini-2.0-flash"),
            latency_gate_ms=retrieval_cfg["latency_gate_ms"],
        )
    else:
        backend = NullBackend()
    from core.checkpoint import resolve_checkpoint
    from core.model_interface import MoshiRAGAdapter
    local_path = resolve_checkpoint(model_cfg.get("checkpoint", "base"))
    return MoshiRAGAdapter(local_path, backend)


def run_demo(config_path: str, record: bool) -> None:
    import numpy as np
    import sounddevice as sd

    cfg = load_config(config_path)
    print("Loading MoshiRAG model — this takes ~30s on first run...")
    model = _build_model(cfg)
    print("Ready. Speak into your microphone. Ctrl-C to stop.\n")

    session_dir: Path | None = None
    if record:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        session_dir = Path(__file__).parent / "sessions" / ts
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"Recording session to {session_dir}/\n")

    capture_q: queue.Queue = queue.Queue()   # raw float32 chunks from mic
    playback_q: queue.Queue = queue.Queue()  # float32 chunks to speaker

    turn_count = [0]
    stop_event = threading.Event()

    # ── sounddevice callbacks (run on audio thread — must be fast) ───────────

    def on_input(indata, frames, _time, _status):
        capture_q.put(indata[:, 0].copy())

    def on_output(outdata, frames, _time, _status):
        try:
            chunk = playback_q.get_nowait()
        except queue.Empty:
            chunk = np.zeros(frames, dtype=np.float32)
        n = min(len(chunk), frames)
        outdata[:n, 0] = chunk[:n]
        if n < frames:
            outdata[n:, 0] = 0.0

    # ── inference worker (runs in background thread) ──────────────────────────

    def inference_worker():
        speech_chunks = []
        silent_chunks = 0
        in_utterance = False

        while not stop_event.is_set():
            try:
                chunk = capture_q.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = _rms(chunk)
            is_speech = rms > SILENCE_RMS_THRESHOLD

            if is_speech:
                speech_chunks.append(chunk)
                silent_chunks = 0
                in_utterance = True
            elif in_utterance:
                speech_chunks.append(chunk)
                silent_chunks += 1

                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    if len(speech_chunks) >= MIN_SPEECH_CHUNKS:
                        _run_turn(speech_chunks)
                    speech_chunks = []
                    silent_chunks = 0
                    in_utterance = False

    def _run_turn(chunks):
        import numpy as np
        turn_count[0] += 1
        n = turn_count[0]

        audio_np = np.concatenate(chunks)
        audio_bytes = _f32_to_wav(audio_np)

        print(f"[turn {n}] inferring...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            audio_out, text_out, meta = model.respond(audio_bytes)
        except Exception as exc:
            print(f"\n[turn {n}] error: {exc}")
            return
        elapsed = time.perf_counter() - t0

        ttfat = meta.get("ttfat_s", 0.0)
        rag = meta.get("rag_triggered", False)
        ret_lat = meta.get("retrieval_latency_s", 0.0)
        flags = f"TTFAT={ttfat:.3f}s"
        if rag:
            flags += f"  RAG={ret_lat*1000:.0f}ms"
        print(f" done ({elapsed:.2f}s total  {flags})")
        if text_out:
            print(f"[turn {n}] model: {text_out[:160]}")

        if record and session_dir:
            _save_wav(session_dir / f"user_{n:04d}.wav", audio_bytes)
            _save_wav(session_dir / f"model_{n:04d}.wav", audio_out)

        # Queue response for playback in CHUNK_SAMPLES-sized pieces
        response_f32 = _wav_to_f32(audio_out)
        for i in range(0, len(response_f32), CHUNK_SAMPLES):
            playback_q.put(response_f32[i:i + CHUNK_SAMPLES])

    worker = threading.Thread(target=inference_worker, daemon=True, name="inference")
    worker.start()

    with (
        sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=on_input,
        ),
        sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=on_output,
        ),
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
            stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="MoshiRAG live mic demo")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save per-turn wav files to demo/sessions/<timestamp>/",
    )
    args = parser.parse_args()
    run_demo(args.config, args.record)


if __name__ == "__main__":
    main()

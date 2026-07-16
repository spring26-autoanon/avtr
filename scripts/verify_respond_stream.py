#!/usr/bin/env python3
"""
Diagnostic: exercise MoshiRAGAdapter.respond_stream() directly against a real
checkpoint on the GPU VM, in isolation from demo/server.py and the browser
client. Run this first when verifying the full-duplex streaming feed loop —
it isolates core/model_interface.py from the WebSocket/FastAPI layer, so a
failure here points straight at respond_stream()/_live_feed_loop rather than
demo plumbing.

Feeds a wav file's audio to respond_stream() in fixed-size chunks, paced to
arrive at roughly real-time speed (like a browser client would), and reports
each output chunk as it's produced — confirming the model is emitting audio
incrementally rather than only at the very end.

Usage (on the VM, inside the moshirag-evals venv):
  uv run --all-extras scripts/verify_respond_stream.py --checkpoint base --wav path/to/sample.wav

--all-extras matters here: torch/transformers live in the gpu extra, and a
plain `uv run` (no --extra flag) won't check or protect their versions — see
CLAUDE.md's "--all-extras on the VM" note.
"""

import argparse
import asyncio
import logging
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.checkpoint import resolve_checkpoint
from core.model_interface import MoshiRAGAdapter
from core.retrieval_backend import NullBackend

CHUNK_SAMPLES = 2048  # matches demo/client/audio.js


def _wav_to_pcm16_bytes(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM wav, got sampwidth={wf.getsampwidth()}")
        if wf.getnchannels() != 1:
            raise ValueError(f"expected mono wav, got nchannels={wf.getnchannels()}")
        return wf.readframes(wf.getnframes())


async def _chunked_realtime(pcm_bytes: bytes, chunk_samples: int, sample_rate: int):
    """Yield fixed-size chunks paced to arrive at roughly real-time speed."""
    chunk_bytes = chunk_samples * 2  # 16-bit -> 2 bytes/sample
    for i in range(0, len(pcm_bytes), chunk_bytes):
        yield pcm_bytes[i : i + chunk_bytes]
        await asyncio.sleep(chunk_samples / sample_rate)


async def _stream(model: MoshiRAGAdapter, wav_path: Path) -> None:
    sample_rate = int(model._state.runner.mimi.sample_rate)
    pcm_bytes = _wav_to_pcm16_bytes(wav_path)
    input_duration_s = len(pcm_bytes) / 2 / sample_rate

    t0 = time.perf_counter()
    n_chunks = 0
    total_out_bytes = 0
    first_out_t = None

    async for audio_out, text_out, _metadata in model.respond_stream(
        _chunked_realtime(pcm_bytes, CHUNK_SAMPLES, sample_rate)
    ):
        n_chunks += 1
        total_out_bytes += len(audio_out)
        if first_out_t is None and len(audio_out) > 0:
            first_out_t = time.perf_counter() - t0
            print(f"  first output chunk at t={first_out_t:.3f}s (input is {input_duration_s:.2f}s long)")
        if text_out:
            print(f"  text: {text_out[:80]!r}")

    elapsed = time.perf_counter() - t0
    print(f"\ndone: {n_chunks} output chunks, {total_out_bytes} bytes, {elapsed:.2f}s total")

    if first_out_t is not None and first_out_t < input_duration_s * 0.9:
        print("  looks streaming: first output arrived before the input finished playing out")
        print("  (i.e. the model responded while still \"listening\" — genuine full-duplex)")
    elif first_out_t is not None:
        print("  ⚠ first output arrived only near/after the end of input")
        print("  — check whether _live_feed_loop is actually feeding incrementally,")
        print("    or whether it's silently falling back to all-silence padding")
    else:
        print("  ⚠ no non-empty output chunks were produced at all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="base", help="Checkpoint alias or path")
    parser.add_argument("--wav", required=True, help="16-bit mono PCM wav file to stream in")
    args = parser.parse_args()

    # Without this, core/model_interface.py's logger.info() calls (checkpoint,
    # ARC-Encoder, warmup progress) are silently discarded — Python's logging
    # module does nothing until configured.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    local_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading MoshiRAGAdapter from {local_path} ...")
    # Constructed synchronously, outside any event loop — moshi's own
    # ServerState.warmup() internally calls asyncio.run() for its warmup
    # coroutine, which fails ("cannot be called from a running event loop")
    # if MoshiRAGAdapter() is constructed from inside one. demo/server.py and
    # evals/runner.py already do this correctly (model built before
    # uvicorn.run()/any asyncio.run() starts); this script must match that.
    model = MoshiRAGAdapter(local_path, NullBackend())
    print("Loaded. Streaming...\n")

    asyncio.run(_stream(model, Path(args.wav)))


if __name__ == "__main__":
    main()

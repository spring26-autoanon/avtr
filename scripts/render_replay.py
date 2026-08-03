#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 5 render: replay dialogues -> time-aligned stereo audio.

Her (assistant) turns  -> ElevenLabs clone (DANIELLE_VOICE_ID), channel 0 (SPEAKER_MAIN).
User (question) turns   -> macOS `say` (free, varied voice), channel 1.
Layout per example: [user question on ch1] [gap] [her answer on ch0], sequential turn-taking
(matches the dialogue-data convention: her=left/ch0, partner=right/ch1).

Writes replay/audio/<id>.wav (24 kHz stereo) + replay/render_manifest.jsonl with, per example,
the reference text, the answer-start time (where ⟨ret⟩/reference apply), and durations — the
inputs the next steps need (precompute reference via :8001, place ⟨ret⟩, align, package).

Runs on the Mac. `source .env` first (needs ELEVENLABS_API_KEY + DANIELLE_VOICE_ID).

    python scripts/render_replay.py --limit 3          # small test batch
    python scripts/render_replay.py --limit 150        # Stage-2b batch (watch char budget)
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import requests
import soundfile as sf

SR = 24000
USER_VOICES = ["Samantha", "Daniel", "Karen"]  # installed macOS `say` voices, rotated for variety
MODEL_ID = "eleven_multilingual_v2"


def tts_her(text: str, key: str, vid: str) -> np.ndarray:
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=pcm_24000",
        headers={"xi-api-key": key, "content-type": "application/json"},
        json={"text": text, "model_id": MODEL_ID}, timeout=120)
    r.raise_for_status()
    return np.frombuffer(r.content, dtype=np.int16).astype(np.float32) / 32768.0


def tts_user(text: str, voice: str) -> np.ndarray:
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        r = subprocess.run(["say", "-v", voice, "-o", path, "--data-format=LEI16@24000", text],
                           capture_output=True)
        if r.returncode != 0:  # voice missing -> fall back to the default voice
            subprocess.run(["say", "-o", path, "--data-format=LEI16@24000", text], check=True,
                           capture_output=True)
        pcm, sr = sf.read(path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        assert sr == SR, f"say produced sr={sr}"
        return pcm
    finally:
        os.unlink(path)


def assemble(user_pcm: np.ndarray, her_pcm: np.ndarray, gap: float = 0.4):
    g = int(gap * SR)
    Tq, Ta = len(user_pcm), len(her_pcm)
    total = Tq + g + Ta
    st = np.zeros((total, 2), dtype=np.float32)
    st[:Tq, 1] = user_pcm                     # user -> ch1 first
    st[Tq + g:Tq + g + Ta, 0] = her_pcm       # her  -> ch0 after gap
    answer_start_sec = (Tq + g) / SR
    return st, answer_start_sec


def ans_of(rec):
    return next(t["text"] for t in rec["turns"] if t["speaker"] == "assistant")


def q_of(rec):
    return next(t["text"] for t in rec["turns"] if t["speaker"] == "user")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/dialogue.filtered.jsonl")
    ap.add_argument("--outdir", default="replay/audio")
    ap.add_argument("--manifest", default="replay/render_manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--gap", type=float, default=0.4)
    args = ap.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    vid = os.environ.get("DANIELLE_VOICE_ID")
    if not key or not vid:
        raise SystemExit("source .env first (need ELEVENLABS_API_KEY + DANIELLE_VOICE_ID)")

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.outdir, exist_ok=True)

    chars = 0
    with open(args.manifest, "w") as mf:
        for i, rec in enumerate(rows):
            ans, q = ans_of(rec), q_of(rec)
            her = tts_her(ans, key, vid)
            usr = tts_user(q, USER_VOICES[i % len(USER_VOICES)])
            st, answer_start = assemble(usr, her, args.gap)
            path = os.path.join(args.outdir, f"{rec['id']}.wav")
            sf.write(path, st, SR)
            chars += len(ans)
            mf.write(json.dumps({
                "id": rec["id"], "kind": rec["kind"], "retrieval": rec["retrieval"],
                "reference": rec["reference"], "path": os.path.abspath(path),
                "answer_start_sec": round(answer_start, 3),
                "duration_sec": round(len(st) / SR, 3),
            }, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(rows)}] {rec['id']}  {len(st)/SR:.1f}s  (her chars so far: {chars})")

    print(f"done: {len(rows)} examples -> {args.outdir}  | ElevenLabs chars used this run: {chars}")


if __name__ == "__main__":
    main()

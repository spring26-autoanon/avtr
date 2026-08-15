#!/usr/bin/env python3
"""Prepare single-speaker mono recordings for moshi-finetune.

The pipeline expects STEREO audio where:
  - left  channel (0) = Moshi's voice  -> we put the target-voice monologue here
  - right channel (1) = the user       -> we leave it silent (Option A, voice-left/silent-right)

This also renames files to snake_case to avoid spaces/camelCase edge cases.
Originals in the source dir are never modified.

Usage:
    python scripts/prepare_stereo.py \
        --src finetune/data/datastereo \
        --dst finetune/data/prepared
"""
import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000  # Mimi's native rate; inputs are already 24 kHz.


def to_snake(stem: str) -> str:
    stem = stem.replace(" ", "_")
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem)  # camelCase -> camel_Case
    stem = re.sub(r"_+", "_", stem)
    return stem.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="finetune/data/datastereo")
    ap.add_argument("--dst", default="finetune/data/prepared")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    wavs = sorted(src.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files found in {src}")

    print(f"{'source':34s} -> {'output':26s} {'dur(s)':>8s}  {'in_ch':>5s}")
    total = 0.0
    for w in wavs:
        data, sr = sf.read(str(w), always_2d=True)  # (frames, channels), float64
        in_ch = data.shape[1]
        # Collapse to a single voice channel (mono expected; average if not).
        voice = data[:, 0] if in_ch == 1 else data.mean(axis=1)
        if sr != TARGET_SR:
            raise SystemExit(
                f"{w.name}: sample rate {sr} != {TARGET_SR}; resample first."
            )
        silence = np.zeros_like(voice)
        stereo = np.column_stack([voice, silence])  # col 0 = left = voice, col 1 = right = silence

        out_name = to_snake(w.stem) + ".wav"
        out_path = dst / out_name
        # Preserve 24-bit depth to match the source PCM.
        sf.write(str(out_path), stereo, TARGET_SR, subtype="PCM_24")

        dur = len(voice) / TARGET_SR
        total += dur
        print(f"{w.name:34s} -> {out_name:26s} {dur:8.2f}  {in_ch:5d}")

    print(f"\nWrote {len(wavs)} stereo files to {dst}  |  total {total:.1f}s = {total/60:.2f} min")


if __name__ == "__main__":
    main()

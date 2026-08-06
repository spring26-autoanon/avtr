#!/usr/bin/env python3
"""Stage 3 — gate the retrieval clips before spending GPU time on training.

Checks, per clip: a non-empty sibling .json exists, the audio is 2ch/24 kHz and not longer
than duration_sec, and the number of <RAG> markers equals the number of reference tensors.
A count mismatch means train.py would silently leave a retrieval turn unconditioned while
still training its answer — which is confabulation taught directly.

    python3 scripts/verify_retrieval_clips.py --clips replay/retrieval
"""
import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune.data.reference_io import load_references  # noqa: E402

MAX_SEC = 100.0
TARGET_SR = 24000


def check_clips(clip_dir):
    clip_dir = Path(clip_dir)
    problems = []
    stats = {"clips": 0, "rag_markers": 0, "reference_tensors": 0, "minutes": 0.0,
             "multi_retrieval_clips": 0}

    for wav in sorted(clip_dir.glob("*.wav")):
        stats["clips"] += 1
        info = sf.info(str(wav))
        stats["minutes"] += info.duration / 60.0
        if info.duration > MAX_SEC + 1e-6:
            problems.append(f"{wav.name}: longer than {MAX_SEC:.0f}s ({info.duration:.1f}s)")
        if info.channels != 2 or info.samplerate != TARGET_SR:
            problems.append(f"{wav.name}: expected 2ch/{TARGET_SR} Hz, got "
                            f"{info.channels}ch/{info.samplerate} Hz")

        transcript = wav.with_suffix(".json")
        if not transcript.exists():
            problems.append(f"{wav.name}: no transcript ({transcript.name})")
            continue
        alignments = json.load(open(transcript)).get("alignments", [])
        if not alignments:
            problems.append(f"{wav.name}: empty alignments")
            continue

        n_markers = sum(1 for w in alignments if w[0] == "<RAG>")
        refs = load_references(str(wav)) or []
        stats["rag_markers"] += n_markers
        stats["reference_tensors"] += len(refs)
        if n_markers > 1:
            stats["multi_retrieval_clips"] += 1
        if n_markers != len(refs):
            problems.append(f"{wav.name}: {n_markers} <RAG> markers but "
                            f"{len(refs)} reference tensors")

    return problems, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="replay/retrieval")
    args = ap.parse_args()

    problems, stats = check_clips(args.clips)
    for p in problems:
        print(f"  FAIL {p}")
    print(f"\n{stats['clips']} clips, {stats['minutes']:.1f} min, "
          f"{stats['rag_markers']} <RAG> markers, "
          f"{stats['reference_tensors']} reference tensors, "
          f"{stats['multi_retrieval_clips']} multi-retrieval clips")
    if problems:
        print(f"{len(problems)} problem(s) — do not train yet.")
        raise SystemExit(1)
    print("gate passed.")


if __name__ == "__main__":
    main()

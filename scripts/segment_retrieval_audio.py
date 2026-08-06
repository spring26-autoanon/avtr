#!/usr/bin/env python3
"""Stage 3 — segment the retrieval recording into per-dialogue units with per-turn labels.

Reads the two Whisper transcripts (channel 0 = Danielle, channel 1 = Joshua), merges them
into a speaker-tagged timestamped transcript, and asks Gemini to mark dialogue boundaries
and label each of HER turns grounded / decline / smalltalk. Grounded and decline turns get
a reference passage — the "retrieved document" the trainer encodes via the :8001 ARC encoder.

Per-turn (not per-segment) labelling is what lets one clip teach the mode switch
chit-chat -> factual -> chit-chat, which is the Stage 3 goal.

LLM timestamps are never trusted: every boundary snaps to a real utterance edge and is then
validated (monotonic, non-overlapping, starts on JOSHUA, ends on DANIELLE, 4-100 s).

    export GEMINI_API_KEY=...
    python3 scripts/segment_retrieval_audio.py \
        --ch0 finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
        --ch1 finetune/data/prepared_retrieval/danielle_joshuarhodes.ch1.json \
        --out replay/retrieval_segments.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_utterances import format_transcript, load_alignments, merge_utterances  # noqa: E402

MAX_SEC = 100.0
MIN_SEC = 4.0


def snap_bounds(start, end, utts):
    """Snap a proposed [start, end] to the nearest real utterance edges."""
    starts = [u["start"] for u in utts]
    ends = [u["end"] for u in utts]
    return (
        min(starts, key=lambda s: abs(s - start)),
        min(ends, key=lambda e: abs(e - end)),
    )


def _turns_in(turns, start, end):
    return [t for t in turns if t["start"] >= start - 1e-6 and t["end"] <= end + 1e-6]


def split_long(segment, utts, max_sec=MAX_SEC):
    """Split an over-long segment at the internal JOSHUA turn nearest its midpoint.

    Each half keeps only the turns inside it, with their own kinds and references.
    Recurses so a very long segment yields as many pieces as needed. A segment with no
    internal JOSHUA turn cannot be split and is returned as-is for the validator to drop.
    """
    start, end = segment["start"], segment["end"]
    if end - start <= max_sec:
        return [segment]

    mid = (start + end) / 2.0
    candidates = [
        t for t in segment["turns"]
        if t["speaker"] == "JOSHUA" and start < t["start"] < end
    ]
    if not candidates:
        return [segment]
    cut = min(candidates, key=lambda t: abs(t["start"] - mid))["start"]
    if cut <= start or cut >= end:
        return [segment]

    left = {**segment, "id": segment["id"] + "a", "start": start, "end": cut,
            "turns": _turns_in(segment["turns"], start, cut)}
    right = {**segment, "id": segment["id"] + "b", "start": cut, "end": end,
             "turns": _turns_in(segment["turns"], cut, end)}
    return split_long(left, utts, max_sec) + split_long(right, utts, max_sec)


def _check_one(seg, min_sec, max_sec):
    """Return a problem string, or None if the segment is usable."""
    dur = seg["end"] - seg["start"]
    if dur < min_sec:
        return f"{seg['id']}: too short ({dur:.1f}s < {min_sec}s)"
    if dur > max_sec:
        return f"{seg['id']}: too long ({dur:.1f}s > {max_sec}s) and unsplittable"
    if not any(t["speaker"] == "DANIELLE" for t in seg["turns"]):
        return f"{seg['id']}: no DANIELLE turn"
    if not seg["turns"] or seg["turns"][0]["speaker"] != "JOSHUA":
        return f"{seg['id']}: does not start on a JOSHUA turn"
    if seg["turns"][-1]["speaker"] != "DANIELLE":
        return f"{seg['id']}: does not end on a DANIELLE turn"
    return None


def validate_segments(segments, utts, max_sec=MAX_SEC, min_sec=MIN_SEC):
    """Snap, split, order and filter. Returns (kept, problems).

    Audio claimed by no segment is fine and is not reported as a problem — retakes and
    false starts are expected. The caller reports the unclaimed total separately.
    """
    problems = []
    snapped = []
    for seg in segments:
        s, e = snap_bounds(seg["start"], seg["end"], utts)
        if e <= s:
            problems.append(f"{seg['id']}: empty after snapping")
            continue
        snapped.append({**seg, "start": s, "end": e,
                        "turns": _turns_in(seg["turns"], s, e)})

    snapped.sort(key=lambda s: s["start"])

    kept = []
    last_end = float("-inf")
    for seg in snapped:
        if seg["start"] < last_end - 1e-6:
            problems.append(f"{seg['id']}: overlaps the previous segment; dropped")
            continue
        for piece in split_long(seg, utts, max_sec):
            problem = _check_one(piece, min_sec, max_sec)
            if problem:
                problems.append(problem)
                continue
            kept.append(piece)
            last_end = piece["end"]
    return kept, problems

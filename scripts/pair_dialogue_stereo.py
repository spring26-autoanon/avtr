#!/usr/bin/env python3
"""Pair two-speaker mono recordings into stereo training WAVs for moshi-finetune.

Each conversation is two isolated mono tracks (one per speaker) that share the
last 10 digits of their filename. We place the MAIN speaker (the cloned voice) on
the LEFT channel (0) and the partner on the RIGHT channel (1), time-aligned, at
24 kHz / PCM_24. Optionally carve an eval slice out of the middle of one
conversation. Originals are never modified.

Usage:
    python scripts/pair_dialogue_stereo.py \
        --src finetune/data/datastereo/clean_moshi_audio_24khz \
        --dst finetune/data/prepared_dialogue \
        --main-name danielle \
        --eval-conv 1411304343 --eval-sec 600 --eval-center-frac 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000


def parse_track_name(name: str) -> tuple[str, str, str]:
    """'audioClayS11556527425_24khz.wav' -> ('clays', '1', '1556527425')."""
    stem = name
    if stem.endswith(".wav"):
        stem = stem[:-4]
    if stem.endswith("_24khz"):
        stem = stem[: -len("_24khz")]
    if not stem.startswith("audio"):
        raise ValueError(f"unexpected name (no 'audio' prefix): {name}")
    stem = stem[len("audio"):]
    if len(stem) < 11 or not stem[-10:].isdigit():
        raise ValueError(f"no 10-digit conversation id in: {name}")
    conv_id = stem[-10:]
    rest = stem[:-10]
    i = len(rest)
    while i > 0 and rest[i - 1].isdigit():
        i -= 1
    idx = rest[i:]
    speaker = rest[:i].lower()
    if not speaker:
        raise ValueError(f"empty speaker name in: {name}")
    return speaker, idx, conv_id

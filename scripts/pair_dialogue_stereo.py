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


def combine_to_stereo(main_path: Path, partner_path: Path) -> tuple[np.ndarray, int, int]:
    """Read two mono tracks, return (stereo[n,2], sr, delta_frames).

    col 0 = left = main (cloned voice); col 1 = right = partner. Truncated to the
    shorter of the two tracks; delta_frames is the length difference removed.
    """
    main, sr_m = sf.read(str(main_path), always_2d=True)
    partner, sr_p = sf.read(str(partner_path), always_2d=True)
    for path, sr in ((main_path, sr_m), (partner_path, sr_p)):
        if sr != TARGET_SR:
            raise SystemExit(f"{path}: sample rate {sr} != {TARGET_SR}; resample first.")
    m = main[:, 0]
    p = partner[:, 0]
    delta = abs(len(m) - len(p))
    n = min(len(m), len(p))
    stereo = np.column_stack([m[:n], p[:n]])
    return stereo, TARGET_SR, delta


def carve_eval(
    stereo: np.ndarray, sr: int, eval_sec: float, center_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut an eval slice from the middle. Returns (before, eval, after)."""
    total = stereo.shape[0]
    eval_frames = int(round(eval_sec * sr))
    if eval_frames >= total:
        raise SystemExit(
            f"eval slice ({eval_sec}s) >= conversation length ({total / sr:.1f}s)."
        )
    center = int(round(total * center_frac))
    start = center - eval_frames // 2
    start = max(0, min(start, total - eval_frames))
    end = start + eval_frames
    return stereo[:start], stereo[start:end], stereo[end:]


def discover_conversations(
    src: Path, main_name: str
) -> dict[str, tuple[Path, str, Path, str]]:
    """Group *.wav by conv_id and split each into (main, partner) by identity."""
    groups: dict[str, list[tuple[Path, str]]] = {}
    for wav in sorted(src.glob("*.wav")):
        speaker, _idx, conv_id = parse_track_name(wav.name)
        groups.setdefault(conv_id, []).append((wav, speaker))

    convs: dict[str, tuple[Path, str, Path, str]] = {}
    key = main_name.lower()
    for conv_id, tracks in groups.items():
        if len(tracks) != 2:
            raise SystemExit(
                f"conversation {conv_id} has {len(tracks)} tracks, expected 2: "
                + ", ".join(p.name for p, _ in tracks)
            )
        mains = [(p, s) for p, s in tracks if key in s]
        partners = [(p, s) for p, s in tracks if key not in s]
        if len(mains) != 1 or len(partners) != 1:
            raise SystemExit(
                f"conversation {conv_id}: need exactly one '{main_name}' track and "
                f"one partner; got mains={[s for _, s in mains]}, "
                f"partners={[s for _, s in partners]}"
            )
        (main_path, main_spk), (partner_path, partner_spk) = mains[0], partners[0]
        convs[conv_id] = (main_path, main_spk, partner_path, partner_spk)
    return convs


def _write(dst: Path, name: str, stereo: np.ndarray) -> None:
    sf.write(str(dst / name), stereo, TARGET_SR, subtype="PCM_24")
    print(f"  wrote {name:42s} {stereo.shape[0] / TARGET_SR:8.1f}s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="finetune/data/datastereo/clean_moshi_audio_24khz")
    ap.add_argument("--dst", default="finetune/data/prepared_dialogue")
    ap.add_argument("--main-name", default="danielle",
                    help="case-insensitive substring identifying the cloned voice")
    ap.add_argument("--eval-conv", default="1411304343",
                    help="conv_id to carve an eval slice from")
    ap.add_argument("--eval-sec", type=float, default=600.0)
    ap.add_argument("--eval-center-frac", type=float, default=0.5)
    args = ap.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    convs = discover_conversations(src, args.main_name)
    if args.eval_conv not in convs:
        raise SystemExit(
            f"--eval-conv {args.eval_conv} not found among: {sorted(convs)}"
        )

    prefix = args.main_name.lower()
    for conv_id, (main_path, _ms, partner_path, partner_spk) in sorted(convs.items()):
        stereo, _sr, delta = combine_to_stereo(main_path, partner_path)
        print(f"conv {conv_id}  partner={partner_spk}  "
              f"len={stereo.shape[0] / TARGET_SR:.1f}s  align_delta={delta} frames")
        if conv_id == args.eval_conv:
            before, ev, after = carve_eval(
                stereo, TARGET_SR, args.eval_sec, args.eval_center_frac
            )
            if before.shape[0]:
                _write(dst, f"{prefix}_{partner_spk}_train_a.wav", before)
            _write(dst, f"{prefix}_{partner_spk}_eval.wav", ev)
            if after.shape[0]:
                _write(dst, f"{prefix}_{partner_spk}_train_b.wav", after)
        else:
            _write(dst, f"{prefix}_{partner_spk}.wav", stereo)

    print(f"\nDone -> {dst}")


if __name__ == "__main__":
    main()

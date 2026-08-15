#!/usr/bin/env python3
"""Build .jsonl manifests for moshi-finetune from a folder of prepared wavs.

Each manifest line: {"path": "<absolute wav path>", "duration": <seconds>}
Writes three files so you can pick a training strategy:
  - all.jsonl    : every file (train on everything, no eval)
  - train.jsonl  : every file EXCEPT the held-out eval file
  - eval.jsonl   : the single held-out file (for do_eval)

Paths are written ABSOLUTE. The trainer loads audio via sphn.dataset_jsonl
(dataset.py), which resolves each manifest `path` RELATIVE TO THE JSONL FILE's
own directory. Since the jsonl sits alongside the wavs, a repo-root-relative
path like "finetune/data/prepared/x.wav" would be doubled into
".../prepared/finetune/data/prepared/x.wav". Absolute paths sidestep that and
also let the interleaver find each sibling ".json" transcript. Run this on the
box where training happens (paths are machine-specific). Use --relative-to only
if you know your loader resolves against a different base.

Usage:
    python scripts/build_manifest.py \
        --wav-dir finetune/data/prepared \
        --out-dir finetune/data/prepared \
        --eval-file movies.wav
"""
import argparse
import json
from pathlib import Path

import soundfile as sf


def duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def write_jsonl(path: Path, entries: list[tuple[str, float]]) -> None:
    with open(path, "w") as f:
        for rel, dur in entries:
            f.write(json.dumps({"path": rel, "duration": dur}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", default="finetune/data/prepared")
    ap.add_argument("--out-dir", default="finetune/data/prepared")
    ap.add_argument("--eval-file", default="movies.wav",
                    help="basename of the file to hold out for eval")
    ap.add_argument("--relative-to", default=None,
                    help="if set, write paths relative to this dir instead of "
                         "absolute (default: absolute, which is what the sphn "
                         "dataloader needs)")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_root = Path(args.relative_to).resolve() if args.relative_to else None

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files in {wav_dir}")

    entries = []
    for w in wavs:
        abspath = w.resolve()
        p = str(abspath.relative_to(rel_root)) if rel_root else str(abspath)
        entries.append((p, duration_sec(w)))

    all_e = entries
    eval_e = [(r, d) for r, d in entries if Path(r).name == args.eval_file]
    train_e = [(r, d) for r, d in entries if Path(r).name != args.eval_file]

    if not eval_e:
        raise SystemExit(
            f"--eval-file {args.eval_file} not found among: "
            + ", ".join(Path(r).name for r, _ in entries)
        )

    write_jsonl(out_dir / "all.jsonl", all_e)
    write_jsonl(out_dir / "train.jsonl", train_e)
    write_jsonl(out_dir / "eval.jsonl", eval_e)

    tot = sum(d for _, d in all_e)
    print(f"all.jsonl   : {len(all_e)} files, {tot/60:.2f} min")
    print(f"train.jsonl : {len(train_e)} files, {sum(d for _,d in train_e)/60:.2f} min")
    print(f"eval.jsonl  : {len(eval_e)} file  ({args.eval_file})")


if __name__ == "__main__":
    main()

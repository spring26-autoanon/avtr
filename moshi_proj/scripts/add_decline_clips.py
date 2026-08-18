#!/usr/bin/env python3
"""Stage 3 — top up the retrieval clips with synthetic `decline` examples from Stage 2b.

Danielle's recording contains ZERO decline turns: she never says "I don't know that one".
Decline examples are what taught trial 4's checkpoint_000400 to decline gracefully instead
of confabulating (see docs/moshi-rag-experiments.md), so training with none risks teaching
that every question has an answer worth producing.

INTERIM MEASURE. These 16 clips are in the ElevenLabs clone voice, not hers, and are the
short terminal Q->A shape that Stage 2b blamed for empty-buffer stalls. At ~3.2 min against
~40 min of real retrieval audio they are too small a fraction to move voice or stalls much.
Replace them with real recorded declines when a session is possible.

Their .json (with <RAG> already inserted) and .ref.safetensors (legacy single-key format,
which reference_io.load_references still reads) were produced during Stage 2b, so nothing is
re-encoded here — this only copies. Run it AFTER precompute_references.py and BEFORE
build_manifest.py, so the clips land in the manifest and the gate checks them.

    ~/fork-venv/bin/python scripts/add_decline_clips.py \
        --manifest replay/render_manifest.jsonl \
        --srcdir replay/audio --dstdir replay/retrieval
"""
import argparse
import json
import shutil
from pathlib import Path

SIBLINGS = (".wav", ".json", ".ref.safetensors")


def selectable(rows, kind):
    return [r for r in rows if r.get("kind") == kind]


def copy_clips(rows, srcdir, dstdir):
    """Copy each clip's three sibling files. Returns (copied_ids, skip_messages).

    A clip missing any sibling is skipped whole — copying a .wav without its
    .ref.safetensors would put a <RAG> marker in the data with nothing to condition it,
    which is exactly the mismatch the gate exists to catch.
    """
    srcdir, dstdir = Path(srcdir), Path(dstdir)
    dstdir.mkdir(parents=True, exist_ok=True)
    copied, skipped = [], []
    for r in rows:
        cid = r["id"]
        missing = [ext for ext in SIBLINGS if not (srcdir / f"{cid}{ext}").exists()]
        if missing:
            skipped.append(f"{cid}: missing {', '.join(missing)}")
            continue
        for ext in SIBLINGS:
            shutil.copy2(srcdir / f"{cid}{ext}", dstdir / f"{cid}{ext}")
        copied.append(cid)
    return copied, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="replay/render_manifest.jsonl")
    ap.add_argument("--srcdir", default="replay/audio")
    ap.add_argument("--dstdir", default="replay/retrieval")
    ap.add_argument("--kind", default="decline")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    targets = selectable(rows, args.kind)
    copied, skipped = copy_clips(targets, args.srcdir, args.dstdir)

    for s in skipped:
        print(f"  SKIP {s}")
    total = sum(r.get("duration_sec", 0) for r in targets if r["id"] in copied)
    print(f"copied {len(copied)}/{len(targets)} '{args.kind}' clips "
          f"({total / 60:.1f} min) -> {args.dstdir}")
    if skipped:
        print(f"{len(skipped)} skipped — their .json/.ref.safetensors were built on the box "
              f"during Stage 2b; run this there, not on the Mac.")


if __name__ == "__main__":
    main()

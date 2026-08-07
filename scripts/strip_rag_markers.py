#!/usr/bin/env python3
"""Produce a marker-free copy of the retrieval clips, for training on PLAIN moshika.

`interleaver._tokenize` maps the word "<RAG>" to raw token id 4. On moshika-rag that is the
rag token; on plain moshika it is an ordinary vocabulary token. Training the voice track
(`example/moshika_voice_max.yaml`) on the marked clips would therefore inject a meaningless
token into the text stream at every retrieval turn.

The audio is the same 72 minutes of her real voice either way, so the clips are worth using
for voice cloning — they just need the markers removed. Reference tensors are NOT copied:
plain moshika has no `reference_with_time` conditioner, and a stray `.ref.safetensors` would
have `interleaver` load a tensor the model has nowhere to put.

    python3 scripts/strip_rag_markers.py \
        --src replay/retrieval --dst replay/retrieval_nomarkers
    # then build a manifest over the copy, on the box (absolute paths):
    #   python scripts/build_manifest.py --wav-dir replay/retrieval_nomarkers \
    #       --out-dir replay/retrieval_nomarkers --no-eval
"""
import argparse
import json
import shutil
from pathlib import Path

MARKER = "<RAG>"


def strip_markers(alignments):
    """Drop every <RAG> alignment, leaving the spoken words untouched."""
    return [w for w in alignments if w[0] != MARKER]


def strip_tree(src, dst):
    """Copy each clip's wav and a marker-free json. Returns the number of clips written.

    Reference tensors are deliberately left behind. A clip whose alignments are entirely
    markers has nothing to train on and is skipped.
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for wav in sorted(src.glob("*.wav")):
        transcript = wav.with_suffix(".json")
        if not transcript.exists():
            print(f"  SKIP {wav.name}: no transcript")
            continue
        alignments = strip_markers(json.load(open(transcript)).get("alignments", []))
        if not alignments:
            print(f"  SKIP {wav.name}: nothing left after stripping markers")
            continue
        shutil.copy2(wav, dst / wav.name)
        with open(dst / transcript.name, "w") as f:
            json.dump({"alignments": alignments}, f, ensure_ascii=False)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="replay/retrieval")
    ap.add_argument("--dst", default="replay/retrieval_nomarkers")
    args = ap.parse_args()

    n = strip_tree(args.src, args.dst)
    print(f"wrote {n} marker-free clips -> {args.dst}")
    print("reference tensors intentionally not copied (plain moshika has no conditioner)")


if __name__ == "__main__":
    main()

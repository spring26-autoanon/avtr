#!/usr/bin/env python3
"""Write a synthetic <wav-stem>.ref.safetensors to smoke-test the replay data plumbing.

Produces a random `reference_with_time` tensor [T_ref, D] under key "reference" — the same
shape/role a real Gemini->ARC(:8001) reference will have. Lets us run the full trainer data
path (interleaver -> collate -> train.py injection -> forward/backward) end-to-end before any
real replay data exists. D defaults to 4096 (moshika model dim; matches the verified spike).

    python scripts/make_synthetic_ref.py finetune/data/prepared/<some>.wav
"""
import argparse
import os

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="path to the wav; sibling <stem>.ref.safetensors is written")
    ap.add_argument("--t-ref", type=int, default=8, help="reference length in frames")
    ap.add_argument("--dim", type=int, default=4096, help="model dim D (moshika=4096)")
    a = ap.parse_args()
    out = os.path.splitext(a.wav)[0] + ".ref.safetensors"
    save_file({"reference": torch.randn(a.t_ref, a.dim, dtype=torch.float32)}, out)
    print(f"wrote {out}  shape=({a.t_ref}, {a.dim})")


if __name__ == "__main__":
    main()

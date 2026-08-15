#!/usr/bin/env python3
"""Produce a RAG-conditioner-free copy of moshika-rag's config.json.

WHY THIS EXISTS
---------------
`train.py:136-143` feeds the loaded config.json straight into model
construction. moshika-rag's config carries RAG-only fields (`conditioners`,
`fuser`, `rag_token_id`) that the moshi-finetune trainer may try to build /
expect conditioning tensors for. The Phase 2 smoke test decides whether that's
a problem:

  - smoke test loads + steps cleanly  -> use the ORIGINAL config, ignore this.
  - smoke test fails on conditioners  -> run this script and point
                                         moshi_paths.config_path at the output.

The stripped config keeps every transformer/depformer dimension identical, so
the LoRA still targets the same attention/MLP/depformer linears. The resulting
lora.safetensors stays valid to apply over the FULL moshika-rag weights at
serve time (Phase 4) — we only drop conditioner *config*, not any shared layer.

USAGE (on the A100 box, after Phase 0 downloads the real config)
----------------------------------------------------------------
    python scripts/strip_config.py \
        --in  checkpoints/moshika-rag/config.json \
        --out checkpoints/moshika-rag/config.stripped.json

Then set in example/moshika_rag_voice.yaml:
    moshi_paths:
      config_path: "checkpoints/moshika-rag/config.stripped.json"
"""
import argparse
import json
from pathlib import Path

# Top-level keys that carry RAG conditioning and must go for a plain-LM load.
# (Removing config only — the actual conditioner *weights* in model.safetensors
# are untouched and remain available when moshi-rag serves with its own config.)
STRIP_KEYS = ["conditioners", "fuser", "rag_token_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="checkpoints/moshika-rag/config.json")
    ap.add_argument("--out", dest="out", default="checkpoints/moshika-rag/config.stripped.json")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(
            f"{inp} not found. Download it first (Phase 0):\n"
            "  python -c \"from huggingface_hub import hf_hub_download as d; "
            "import os,shutil; os.makedirs('checkpoints/moshika-rag',exist_ok=True); "
            "shutil.copy(d('kyutai/moshika-rag-pytorch-bf16','config.json'),"
            "'checkpoints/moshika-rag/config.json')\""
        )

    cfg = json.loads(inp.read_text())

    removed = [k for k in STRIP_KEYS if k in cfg]
    for k in removed:
        cfg.pop(k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"read   : {inp}")
    print(f"removed: {removed or '(none — config had no RAG conditioner keys)'}")
    print(f"kept   : {len(cfg)} top-level keys")
    print(f"wrote  : {out}")
    if not removed:
        print(
            "\nNOTE: none of the expected RAG keys were present. Inspect the config "
            "manually — the field names may differ in your checkout."
        )


if __name__ == "__main__":
    main()

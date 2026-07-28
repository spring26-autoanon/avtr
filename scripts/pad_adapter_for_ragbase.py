#!/usr/bin/env python3
"""Pad a plain-moshika LoRA adapter so it loads onto moshika-rag under the fork.

The moshi-rag fork's `get_lora_moshi` wraps EVERY linear with LoRA — including the
RAG conditioners (`condition_provider.conditioners.*`). A voice adapter trained on
plain moshika only fills the transformer+depformer slots, leaving the conditioner
slots on the meta device -> `Cannot copy out of meta tensor` at load (loaders.py
~L560). This fills those missing conditioner slots with ZERO tensors (a no-op LoRA:
`W + scaling * (0 @ 0) = W`), so the adapter loads cleanly and the conditioners are
preserved bit-for-bit.

Verified 2026-07-28: checkpoint_000700 has 674 keys, all matching moshika-rag's
transformer/depformer slots exactly (0 extra); the 372 missing slots are all
conditioners -> padded to 1046 tensors, loads + fuses on CUDA (~10.7B params).

Run under the FORK env (moshi 0.2.13). Needs HF access to the base repo and its
gated ARC-encoder dep (`meta-llama/Llama-3.2-3B-Instruct`).

Usage (on the A100, fork venv):
    ~/fork-venv/bin/python scripts/pad_adapter_for_ragbase.py \
        --lora runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora.safetensors \
        --out  runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora_padded.safetensors

Then serve with the padded file + these overrides:
    get_moshi(fuse_lora=True, lm_kwargs_overrides={"lora":True,"lora_rank":64,"lora_scaling":2.0})
"""
import argparse

import torch
from safetensors.torch import load_file, save_file

from moshi.models.loaders import CheckpointInfo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="plain-moshika LoRA safetensors")
    ap.add_argument("--base", default="kyutai/moshika-rag-pytorch-bf16")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="output padded safetensors")
    args = ap.parse_args()

    ov = {"lora": True, "lora_rank": args.rank, "lora_scaling": args.scaling}
    # Build the architecture skeleton ONLY (load_weight=False -> no 16 GB base load,
    # no meta crash) purely to enumerate every LoRA slot name + shape.
    skel = CheckpointInfo.from_hf_repo(args.base).get_moshi(
        device="cpu", dtype=torch.bfloat16, load_weight=False, lm_kwargs_overrides=ov
    )
    slots = {n: p for n, p in skel.named_parameters() if "lora" in n.lower()}

    state = load_file(args.lora)
    have = set(state)
    added = 0
    for name, param in slots.items():
        if name not in have:
            state[name] = torch.zeros(tuple(param.shape), dtype=torch.bfloat16)
            added += 1
    save_file(state, args.out)
    print(f"padded {added} conditioner slots -> {len(state)} tensors; wrote {args.out}")


if __name__ == "__main__":
    main()

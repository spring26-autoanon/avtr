#!/usr/bin/env python3
"""Measure how far moshika-rag's weights drifted from plain moshika.

WHY THIS EXISTS
---------------
Path B trains a LoRA on PLAIN moshika (`kyutai/moshika-pytorch-bf16`) but serves
it over moshika-RAG (`kyutai/moshika-rag-pytorch-bf16`). A LoRA is a low-rank
delta `ΔW` calibrated against the base it was trained on. Applying it to a
DIFFERENT base `W'` only reproduces the same behavior if `W' ≈ W` in the layers
`ΔW` touches — and the voice lives in the ATTENTION layers (proven by the
freeze-attention experiment, NEXT_STEPS.md Part 2b).

moshika-rag is (almost certainly) moshika + a RAG fine-tuning delta, not an
independent model. This script QUANTIFIES that delta so you can decide whether
"confirm by ear" is a formality or a real risk *before* spending A100 serve time:

  - small attention drift (a few %) -> ΔW should transfer cleanly; voice preserved.
  - large attention drift            -> expect a weaker/smeared timbre; the LoRA is
                                        operating on a shifted feature space.

The metric per tensor is the relative Frobenius distance:

    ‖W_rag - W_base‖_F / ‖W_base‖_F

CPU-only (no CUDA needed) but it reads two large checkpoints (~16 GB each). It
loads tensors LAZILY via safetensors `safe_open`, so peak memory is one tensor
at a time, not the whole model. Prefer running it on the A100 box where the HF
weights are already cached.

USAGE
-----
    python scripts/compare_attention.py            # defaults: moshika vs moshika-rag
    python scripts/compare_attention.py --top 40   # show more per-layer rows
    python scripts/compare_attention.py \
        --base-repo kyutai/moshika-pytorch-bf16 \
        --rag-repo  kyutai/moshika-rag-pytorch-bf16
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# A weight name is "attention" if it carries one of these. Fused QKV is stored as
# `...self_attn.in_proj_weight`; the output projection is `...self_attn.out_proj.weight`.
# This matches both the main transformer and the depformer attention blocks.
ATTN_MARKERS = ("in_proj_weight", "in_proj.weight", "out_proj.weight")


def _resolve_files(repo: str) -> dict[str, str]:
    """Return {tensor_key: local_safetensors_path}, handling single-file OR sharded repos."""
    # Sharded checkpoints ship an index that maps every key to its shard.
    try:
        index_path = hf_hub_download(repo, "model.safetensors.index.json")
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        local = {s: hf_hub_download(repo, s) for s in shards}
        return {key: local[shard] for key, shard in weight_map.items()}
    except Exception:
        pass
    # Single-file checkpoint (the kyutai moshika repos today).
    path = hf_hub_download(repo, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        return {key: path for key in f.keys()}


def _is_attn(key: str) -> bool:
    return "attn" in key and any(m in key for m in ATTN_MARKERS)


def _category(key: str) -> str:
    if _is_attn(key):
        return "attention"
    if any(m in key for m in ("gating", "mlp", "linear1", "linear2")):
        return "mlp/gating"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-repo", default="kyutai/moshika-pytorch-bf16")
    ap.add_argument("--rag-repo", default="kyutai/moshika-rag-pytorch-bf16")
    ap.add_argument("--top", type=int, default=20, help="how many most-changed attention layers to list")
    args = ap.parse_args()

    print(f"base : {args.base_repo}")
    print(f"rag  : {args.rag_repo}")
    print("resolving checkpoint files (may download ~16 GB each on first run)...\n")

    base_files = _resolve_files(args.base_repo)
    rag_files = _resolve_files(args.rag_repo)

    base_keys = set(base_files)
    rag_keys = set(rag_files)
    common = sorted(base_keys & rag_keys)
    only_base = sorted(base_keys - rag_keys)
    only_rag = sorted(rag_keys - base_keys)

    # Cache open handles per file so we don't reopen for every tensor.
    handles: dict[str, object] = {}

    def get(files: dict[str, str], key: str) -> torch.Tensor:
        path = files[key]
        if path not in handles:
            handles[path] = safe_open(path, framework="pt").__enter__()
        return handles[path].get_tensor(key).float()

    rows: list[tuple[str, str, float]] = []  # (category, key, rel_diff)
    shape_mismatch: list[str] = []
    for key in common:
        wb = get(base_files, key)
        wr = get(rag_files, key)
        if wb.shape != wr.shape:
            shape_mismatch.append(key)
            continue
        denom = wb.norm().item()
        if denom == 0.0:
            continue
        rel = (wr - wb).norm().item() / denom
        rows.append((_category(key), key, rel))

    # ---- per-category aggregates -------------------------------------------------
    def stats(vals: list[float]) -> str:
        if not vals:
            return "  (none)"
        t = torch.tensor(vals)
        return (
            f"  n={len(vals):4d}  "
            f"median={t.median():.4f}  mean={t.mean():.4f}  "
            f"min={t.min():.4f}  max={t.max():.4f}"
        )

    print("=" * 78)
    print("RELATIVE WEIGHT DRIFT  ‖W_rag - W_base‖ / ‖W_base‖  (0 = identical)")
    print("=" * 78)
    for cat in ("attention", "mlp/gating", "other"):
        vals = [r for c, _, r in rows if c == cat]
        print(f"\n[{cat}]")
        print(stats(vals))

    # ---- the layers that matter most: attention, most-changed first --------------
    attn_rows = sorted((r for r in rows if r[0] == "attention"), key=lambda x: -x[2])
    print("\n" + "=" * 78)
    print(f"TOP {min(args.top, len(attn_rows))} MOST-CHANGED ATTENTION TENSORS (voice lives here)")
    print("=" * 78)
    for _, key, rel in attn_rows[: args.top]:
        print(f"  {rel:7.4f}   {key}")

    # ---- structural differences (the RAG conditioners) ---------------------------
    print("\n" + "=" * 78)
    print("STRUCTURE")
    print("=" * 78)
    print(f"  shared tensors        : {len(common)}")
    print(f"  only in base (moshika): {len(only_base)}")
    print(f"  only in rag  (RAG add): {len(only_rag)}   <- conditioners / fusion / retrieval")
    if shape_mismatch:
        print(f"  shape mismatches      : {len(shape_mismatch)} (skipped): {shape_mismatch[:5]}")

    # ---- verdict -----------------------------------------------------------------
    attn_vals = [r for _, _, r in attn_rows]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not attn_vals:
        print("  No attention tensors matched — check ATTN_MARKERS against these repos' key names.")
    else:
        med = torch.tensor(attn_vals).median().item()
        mx = max(attn_vals)
        print(f"  attention median drift = {med:.4f} | max = {mx:.4f}")
        if med < 0.05:
            print("  → SMALL. Strong reason to expect the voice LoRA transfers cleanly to moshika-rag.")
        elif med < 0.15:
            print("  → MODERATE. Voice should mostly transfer; audition carefully for timbre loss.")
        else:
            print("  → LARGE. moshika-rag's attention has drifted a lot; expect degraded voice.")
            print("    Consider that Path B's overlay assumption is weak here.")
    print("\n  Reminder: this bounds the RISK; the final gate is still confirm-by-ear.")


if __name__ == "__main__":
    main()

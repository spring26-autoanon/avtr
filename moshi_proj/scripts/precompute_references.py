#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 5: precompute reference_with_time tensors via the :8001 ARC encoder.

For each retrieval example in render_manifest.jsonl, POST its reference text to the running
reference-encoder service (:8001 /embed) and save the returned tensor as a sibling
<audiodir>/<id>.ref.safetensors under key "reference" — exactly the format the trainer's data
plumbing loads (interleaver._load_reference_tensor). smalltalk / no-retrieval examples get no
file (-> Sample.reference_tensor is None -> no conditioning injected, as intended).

Runs ON THE BOX (needs the :8001 service up; see Task 3). Stdlib POST + safetensors.

    ~/fork-venv/bin/python scripts/precompute_references.py \
        --manifest replay/render_manifest.jsonl --audiodir replay/audio
"""
import argparse
import json
import os
import sys
import urllib.request

from safetensors.torch import load as st_load

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune.data.reference_io import save_references  # noqa: E402


def embed(url: str, text: str):
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/embed", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content = r.read()
    tensors = st_load(content)                     # :8001 returns safetensors, key "tensor"
    key = "tensor" if "tensor" in tensors else next(iter(tensors))
    t = tensors[key]
    if t.dim() == 3 and t.shape[0] == 1:           # drop batch dim -> [T, D] (what the trainer wants)
        t = t[0]
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="replay/render_manifest.jsonl")
    ap.add_argument("--audiodir", default="replay/audio")
    ap.add_argument("--url", default=os.environ.get("REFERENCE_ENCODER_URL", "http://localhost:8001"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]

    # Preflight: one probe against the encoder before iterating, so a service that is not
    # running fails in one clear line rather than a urllib traceback on the first clip.
    try:
        probe = embed(args.url, "preflight check")
    except Exception as e:
        raise SystemExit(
            f"reference encoder unreachable at {args.url} ({type(e).__name__}: {e}).\n"
            f"Start it on the second GPU before running this, e.g.:\n"
            f"  CUDA_VISIBLE_DEVICES=1 ~/fork-venv/bin/python -m moshi.server_conditioner \\\n"
            f"    --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \\\n"
            f"    --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \\\n"
            f"    --conditioner reference_with_time --cuda-device 0 --port 8001\n"
            f"(--cuda-device 0 is required: it makes attn_bias and the model share a device.)"
        ) from e
    print(f"encoder OK at {args.url}: probe tensor {tuple(probe.shape)} {probe.dtype}")

    n_ref = n_skip = 0
    for i, rec in enumerate(rows):
        # Stage 3 manifests carry an ordered `references` list (one per <RAG> marker);
        # the Stage 2b synthetic manifests carried a single `reference` + `retrieval` flag.
        refs = rec.get("references")
        if refs is None:
            refs = [rec["reference"]] if rec.get("retrieval") and rec.get("reference") else []
        if not refs:
            n_skip += 1
            continue
        tensors = [embed(args.url, text) for text in refs]
        out = os.path.join(args.audiodir, f"{rec['id']}.ref.safetensors")
        save_references(out, tensors)
        n_ref += len(tensors)
        if i < 3 or i % 25 == 0:
            shapes = ", ".join(str(tuple(t.shape)) for t in tensors)
            print(f"  [{i+1}/{len(rows)}] {rec['id']}  {len(tensors)} ref(s) {shapes} -> {out}")

    print(f"done: {n_ref} reference tensors written, {n_skip} skipped (no-retrieval) -> {args.audiodir}")


if __name__ == "__main__":
    main()

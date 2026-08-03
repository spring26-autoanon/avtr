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
import urllib.request

from safetensors.torch import load as st_load
from safetensors.torch import save_file


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
    n_ref = n_skip = 0
    for i, rec in enumerate(rows):
        if not rec.get("retrieval") or not rec.get("reference"):
            n_skip += 1
            continue
        tensor = embed(args.url, rec["reference"])
        out = os.path.join(args.audiodir, f"{rec['id']}.ref.safetensors")
        save_file({"reference": tensor.contiguous()}, out)
        n_ref += 1
        if i < 3 or i % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] {rec['id']}  ref {tuple(tensor.shape)} {tensor.dtype} -> {out}")

    print(f"done: {n_ref} reference tensors written, {n_skip} skipped (no-retrieval) -> {args.audiodir}")


if __name__ == "__main__":
    main()

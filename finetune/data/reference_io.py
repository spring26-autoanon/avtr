"""Read/write the per-clip precomputed reference_with_time tensors.

One clip can contain several retrieval turns, so a clip's .ref.safetensors holds N tensors
under keys reference_0 .. reference_{N-1}, in the same order as the clip's <RAG> markers.
train.py pairs the i-th marker with the i-th tensor.

Deliberately free of any `moshi` import so it can be unit-tested off the A100.
"""
import os

from safetensors.torch import load_file, save_file

KEY = "reference"


def ref_path_for(wav_path) -> str:
    return os.path.splitext(str(wav_path))[0] + ".ref.safetensors"


def save_references(path: str, tensors: list) -> None:
    """Write N tensors as reference_0 .. reference_{N-1}. An empty list writes no file."""
    if not tensors:
        return
    save_file({f"{KEY}_{i}": t.contiguous() for i, t in enumerate(tensors)}, path)


def load_references(wav_path):
    """Return the clip's reference tensors in marker order, or None if it has none.

    Accepts the legacy single-`reference`-key files written for the Stage 2b synthetic set.
    """
    path = ref_path_for(wav_path)
    if not os.path.exists(path):
        return None
    tensors = load_file(path)
    if KEY in tensors:                      # legacy single-reference format
        return [tensors[KEY]]
    indexed = []
    for key, value in tensors.items():
        if key.startswith(KEY + "_"):
            indexed.append((int(key[len(KEY) + 1:]), value))
    if not indexed:
        return None
    indexed.sort(key=lambda kv: kv[0])      # numeric, not lexicographic
    return [value for _i, value in indexed]

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from add_decline_clips import copy_clips, selectable
from finetune.data.reference_io import save_references


def _synthetic(d, cid, kind, with_json=True, with_ref=True):
    sf.write(str(d / f"{cid}.wav"), np.zeros((24000, 2), dtype="float32"), 24000,
             subtype="PCM_24")
    if with_json:
        al = [["<RAG>", [0.0, 0.1], "SPEAKER_MAIN"], ["answer", [0.1, 0.6], "SPEAKER_MAIN"]]
        json.dump({"alignments": al}, open(d / f"{cid}.json", "w"))
    if with_ref:
        save_references(str(d / f"{cid}.ref.safetensors"), [torch.ones(2, 3)])
    return {"id": cid, "kind": kind, "retrieval": kind != "smalltalk",
            "reference": "p", "duration_sec": 1.0}


def test_selects_only_the_requested_kind(tmp_path):
    rows = [_synthetic(tmp_path, "decline-1", "decline"),
            _synthetic(tmp_path, "grounded-1", "grounded"),
            _synthetic(tmp_path, "small-1", "smalltalk")]
    assert [r["id"] for r in selectable(rows, "decline")] == ["decline-1"]


def test_copies_all_three_sibling_files(tmp_path):
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    rows = [_synthetic(src, "decline-1", "decline")]
    copied, skipped = copy_clips(rows, src, dst)
    assert copied == ["decline-1"] and skipped == []
    for ext in (".wav", ".json", ".ref.safetensors"):
        assert (dst / f"decline-1{ext}").exists(), ext


def test_skips_a_clip_missing_its_alignments(tmp_path):
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    rows = [_synthetic(src, "decline-1", "decline", with_json=False)]
    copied, skipped = copy_clips(rows, src, dst)
    assert copied == []
    assert skipped and "decline-1" in skipped[0] and ".json" in skipped[0]
    assert not (dst / "decline-1.wav").exists()   # nothing partially copied


def test_skips_a_clip_missing_its_reference_tensor(tmp_path):
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    rows = [_synthetic(src, "decline-1", "decline", with_ref=False)]
    copied, skipped = copy_clips(rows, src, dst)
    assert copied == []
    assert skipped and "ref.safetensors" in skipped[0]


def test_copied_clip_passes_the_gate(tmp_path):
    """The whole point: a topped-up clip must satisfy marker-count == tensor-count."""
    from verify_retrieval_clips import check_clips
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    copy_clips([_synthetic(src, "decline-1", "decline")], src, dst)
    problems, stats = check_clips(dst)
    assert problems == []
    assert stats["rag_markers"] == stats["reference_tensors"] == 1

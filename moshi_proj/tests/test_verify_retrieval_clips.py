import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data.reference_io import save_references
from verify_retrieval_clips import check_clips


def _clip(d, name, words, n_refs, seconds=10.0, sr=24000, channels=2):
    n = int(seconds * sr)
    sf.write(str(d / f"{name}.wav"), np.zeros((n, channels), dtype="float32"), sr,
             subtype="PCM_24")
    alignments = [[w, [float(i), float(i) + 0.5], "SPEAKER_MAIN"] for i, w in enumerate(words)]
    json.dump({"alignments": alignments}, open(d / f"{name}.json", "w"))
    if n_refs:
        save_references(str(d / f"{name}.ref.safetensors"),
                        [torch.ones(2, 3) for _ in range(n_refs)])


def test_clean_directory_reports_no_problems(tmp_path):
    _clip(tmp_path, "a", ["hi", "<RAG>", "fact"], 1)
    _clip(tmp_path, "b", ["just", "chatting"], 0)
    problems, stats = check_clips(tmp_path)
    assert problems == []
    assert stats["clips"] == 2
    assert stats["rag_markers"] == 1
    assert stats["reference_tensors"] == 1


def test_marker_without_a_reference_tensor_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi", "<RAG>", "fact"], 0)
    problems, _ = check_clips(tmp_path)
    assert any("1 <RAG>" in p and "0 reference" in p for p in problems)


def test_reference_tensor_without_a_marker_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi", "fact"], 1)
    problems, _ = check_clips(tmp_path)
    assert any("0 <RAG>" in p and "1 reference" in p for p in problems)


def test_missing_transcript_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi"], 0)
    (tmp_path / "a.json").unlink()
    problems, _ = check_clips(tmp_path)
    assert any("no transcript" in p for p in problems)


def test_empty_transcript_is_a_problem(tmp_path):
    _clip(tmp_path, "a", [], 0)
    problems, _ = check_clips(tmp_path)
    assert any("empty alignments" in p for p in problems)


def test_clip_over_duration_sec_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi"], 0, seconds=120.0)
    problems, _ = check_clips(tmp_path)
    assert any("longer than 100" in p for p in problems)


def test_mono_clip_is_a_problem(tmp_path):
    """Channel 1 carries Joshua; a mono clip would train against a silent user."""
    _clip(tmp_path, "a", ["hi"], 0, channels=1)
    problems, _ = check_clips(tmp_path)
    assert any("2ch" in p for p in problems)


def test_multi_retrieval_clips_are_counted(tmp_path):
    _clip(tmp_path, "a", ["hi", "<RAG>", "one", "<RAG>", "two"], 2)
    problems, stats = check_clips(tmp_path)
    assert problems == []
    assert stats["multi_retrieval_clips"] == 1
    assert stats["rag_markers"] == 2

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from strip_rag_markers import strip_markers, strip_tree


def _al(w, s, e):
    return [w, [s, e], "SPEAKER_MAIN"]


def test_removes_rag_markers_and_keeps_words():
    al = [_al("hi", 0.0, 0.5), _al("<RAG>", 0.9, 1.0), _al("fact", 1.0, 1.5)]
    assert [w[0] for w in strip_markers(al)] == ["hi", "fact"]


def test_leaves_marker_free_alignments_untouched():
    al = [_al("just", 0.0, 0.5), _al("chatting", 0.6, 1.0)]
    assert strip_markers(al) == al


def test_removes_several_markers():
    al = [_al("<RAG>", 0.0, 0.1), _al("a", 0.1, 0.5),
          _al("<RAG>", 2.0, 2.1), _al("b", 2.1, 2.5)]
    assert [w[0] for w in strip_markers(al)] == ["a", "b"]


def _clip(d, name, words, with_ref=True):
    sf.write(str(d / f"{name}.wav"), np.zeros((24000, 2), dtype="float32"), 24000,
             subtype="PCM_24")
    json.dump({"alignments": [_al(w, float(i), i + 0.5) for i, w in enumerate(words)]},
              open(d / f"{name}.json", "w"))
    if with_ref:
        (d / f"{name}.ref.safetensors").write_bytes(b"not-a-real-tensor")


def test_tree_copies_audio_and_strips_json(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _clip(src, "ret-0", ["hi", "<RAG>", "fact"])
    n = strip_tree(src, dst)
    assert n == 1
    assert (dst / "ret-0.wav").exists()
    data = json.load(open(dst / "ret-0.json"))
    assert [w[0] for w in data["alignments"]] == ["hi", "fact"]


def test_tree_does_not_copy_reference_tensors(tmp_path):
    """Plain moshika has no reference conditioner; a .ref.safetensors would make
    interleaver load a tensor the model has nowhere to put."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _clip(src, "ret-0", ["hi", "<RAG>", "fact"])
    strip_tree(src, dst)
    assert not (dst / "ret-0.ref.safetensors").exists()


def test_tree_skips_a_clip_with_no_words_left(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _clip(src, "ret-0", ["<RAG>"])
    assert strip_tree(src, dst) == 0
    assert not (dst / "ret-0.wav").exists()

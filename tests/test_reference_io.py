import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data.reference_io import load_references, save_references


def test_roundtrip_preserves_order(tmp_path):
    ref_path = tmp_path / "clip.ref.safetensors"
    a = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    b = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    save_references(str(ref_path), [a, b])
    out = load_references(str(tmp_path / "clip.wav"))
    assert len(out) == 2
    assert torch.equal(out[0], a)
    assert torch.equal(out[1], b)


def test_order_holds_past_ten_references(tmp_path):
    """Guards against lexicographic key sorting putting reference_10 before reference_2."""
    ref_path = tmp_path / "clip.ref.safetensors"
    tensors = [torch.full((1, 2), float(i)) for i in range(12)]
    save_references(str(ref_path), tensors)
    out = load_references(str(tmp_path / "clip.wav"))
    assert [float(t[0, 0]) for t in out] == [float(i) for i in range(12)]


def test_missing_file_returns_none(tmp_path):
    assert load_references(str(tmp_path / "absent.wav")) is None


def test_legacy_single_reference_key_still_loads(tmp_path):
    """Synthetic replay clips from Stage 2b use the old single 'reference' key."""
    from safetensors.torch import save_file
    t = torch.ones(2, 3)
    save_file({"reference": t.contiguous()}, str(tmp_path / "clip.ref.safetensors"))
    out = load_references(str(tmp_path / "clip.wav"))
    assert len(out) == 1
    assert torch.equal(out[0], t)


def test_empty_list_writes_nothing(tmp_path):
    save_references(str(tmp_path / "clip.ref.safetensors"), [])
    assert not (tmp_path / "clip.ref.safetensors").exists()
    assert load_references(str(tmp_path / "clip.wav")) is None


def test_ref_path_is_derived_from_the_wav_stem(tmp_path):
    from finetune.data.reference_io import ref_path_for
    assert ref_path_for("/clips/ret-0.wav").endswith("/clips/ret-0.ref.safetensors")

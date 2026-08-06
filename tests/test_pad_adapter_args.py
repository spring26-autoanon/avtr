from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(fn_name):
    """pad_adapter_for_ragbase imports moshi at module scope (A100-only), so exec just
    the helper's source in a bare namespace."""
    src = (REPO / "scripts/pad_adapter_for_ragbase.py").read_text()
    start = src.index(f"def {fn_name}(")
    end = src.index("\ndef ", start + 1)
    ns: dict = {"Path": Path}
    exec(src[start:end], ns)
    return ns[fn_name]


def test_default_out_is_a_sibling_of_the_adapter():
    out_path_for = _load("out_path_for")
    got = out_path_for("/r/checkpoint_000400/consolidated/lora.safetensors", None)
    assert got == "/r/checkpoint_000400/consolidated/lora_padded.safetensors"


def test_explicit_out_wins():
    out_path_for = _load("out_path_for")
    assert out_path_for("/r/lora.safetensors", "/tmp/x.safetensors") == "/tmp/x.safetensors"


def test_default_out_handles_an_already_padded_name():
    out_path_for = _load("out_path_for")
    got = out_path_for("/r/lora_padded.safetensors", None)
    assert got.endswith("lora_padded_padded.safetensors")

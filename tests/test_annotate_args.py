from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_out_path_for():
    """annotate.py imports torch/whisper at module scope (A100-only), so exec just
    the helper's source in a bare namespace."""
    src = (REPO / "annotate.py").read_text()
    marker = "def out_path_for("
    start = src.index(marker)
    end = src.index("\ndef ", start + 1)
    ns: dict = {"Path": Path}
    exec(src[start:end], ns)
    return ns["out_path_for"]


def test_default_suffix_is_plain_json():
    out_path_for = _load_out_path_for()
    assert out_path_for(Path("/d/a.wav"), ".json") == Path("/d/a.json")


def test_channel_suffix_does_not_clobber_channel_zero():
    out_path_for = _load_out_path_for()
    assert out_path_for(Path("/d/a.wav"), ".ch1.json") == Path("/d/a.ch1.json")


def test_suffix_replaces_only_the_final_extension():
    out_path_for = _load_out_path_for()
    assert out_path_for(Path("/d/a.b.wav"), ".ch1.json") == Path("/d/a.b.ch1.json")

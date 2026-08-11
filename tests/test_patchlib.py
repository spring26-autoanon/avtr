import pytest
from serving_exp.patches.patchlib import apply_patch, revert_patch, tag_marker, PatchError


def _write(tmp_path, text):
    p = tmp_path / "target.py"
    p.write_text(text)
    return str(p)


def test_apply_replaces_anchor_and_writes_backup(tmp_path):
    p = _write(tmp_path, "a = 1\nANCHOR\nb = 2\n")
    backup = apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    text = open(p).read()
    assert "PATCHED" in text and "ANCHOR" not in text
    assert tag_marker("e1") in text
    assert open(backup).read() == "a = 1\nANCHOR\nb = 2\n"


def test_apply_missing_anchor_raises_and_leaves_file(tmp_path):
    p = _write(tmp_path, "no anchor here\n")
    with pytest.raises(PatchError):
        apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    assert open(p).read() == "no anchor here\n"


def test_apply_twice_raises(tmp_path):
    p = _write(tmp_path, "ANCHOR\n")
    apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    with pytest.raises(PatchError):
        apply_patch(p, "PATCHED", "AGAIN", tag="e1")


def test_revert_restores_original(tmp_path):
    p = _write(tmp_path, "ANCHOR\n")
    apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    revert_patch(p, tag="e1")
    assert open(p).read() == "ANCHOR\n"


def test_revert_without_backup_raises(tmp_path):
    p = _write(tmp_path, "x\n")
    with pytest.raises(PatchError):
        revert_patch(p, tag="nope")

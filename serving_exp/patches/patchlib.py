"""Anchored, revertible surgery for box-side files (stdlib only — runs on the GPU box)."""
import shutil


class PatchError(RuntimeError):
    pass


def tag_marker(tag: str) -> str:
    return f"# PATCH:{tag}"


def apply_patch(path: str, anchor: str, replacement: str, tag: str) -> str:
    try:
        src = open(path).read()
    except OSError as e:
        raise PatchError(f"cannot read {path}: {e}") from e
    if tag_marker(tag) in src:
        raise PatchError(f"{path} already carries patch {tag!r}")
    if anchor not in src:
        raise PatchError(f"anchor not found in {path} — file drifted from expected")
    backup = f"{path}.bak_{tag}"
    shutil.copyfile(path, backup)
    first, sep, rest = replacement.partition("\n")
    marked = first + "  " + tag_marker(tag) + sep + rest
    open(path, "w").write(src.replace(anchor, marked, 1))
    return backup


def revert_patch(path: str, tag: str) -> None:
    backup = f"{path}.bak_{tag}"
    try:
        shutil.copyfile(backup, path)
    except OSError as e:
        raise PatchError(f"no backup for tag {tag!r} at {backup}: {e}") from e

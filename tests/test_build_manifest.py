import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_manifest import main


def _wav(path, seconds, sr=24000):
    sf.write(str(path), np.zeros((int(seconds * sr), 2), dtype="float32"), sr, subtype="PCM_24")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_no_eval_puts_every_file_in_train(tmp_path, monkeypatch):
    _wav(tmp_path / "a.wav", 1.0)
    _wav(tmp_path / "b.wav", 2.0)
    monkeypatch.setattr(sys, "argv", [
        "build_manifest.py", "--wav-dir", str(tmp_path), "--out-dir", str(tmp_path), "--no-eval",
    ])
    main()
    assert len(_rows(tmp_path / "train.jsonl")) == 2
    assert len(_rows(tmp_path / "all.jsonl")) == 2
    assert not (tmp_path / "eval.jsonl").exists()


def test_no_eval_does_not_require_an_eval_file_to_exist(tmp_path, monkeypatch):
    _wav(tmp_path / "a.wav", 1.0)
    monkeypatch.setattr(sys, "argv", [
        "build_manifest.py", "--wav-dir", str(tmp_path), "--out-dir", str(tmp_path),
        "--no-eval", "--eval-file", "absent.wav",
    ])
    main()   # must not raise
    assert len(_rows(tmp_path / "train.jsonl")) == 1


def test_missing_eval_file_still_fails_without_the_flag(tmp_path, monkeypatch):
    _wav(tmp_path / "a.wav", 1.0)
    monkeypatch.setattr(sys, "argv", [
        "build_manifest.py", "--wav-dir", str(tmp_path), "--out-dir", str(tmp_path),
        "--eval-file", "absent.wav",
    ])
    with pytest.raises(SystemExit):
        main()


def test_paths_are_absolute(tmp_path, monkeypatch):
    _wav(tmp_path / "a.wav", 1.0)
    monkeypatch.setattr(sys, "argv", [
        "build_manifest.py", "--wav-dir", str(tmp_path), "--out-dir", str(tmp_path), "--no-eval",
    ])
    main()
    assert Path(_rows(tmp_path / "train.jsonl")[0]["path"]).is_absolute()

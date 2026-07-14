import pytest
from core.config import _interpolate, load_config


def test_interpolate_string(monkeypatch):
    monkeypatch.setenv("MY_VAR", "hello")
    assert _interpolate("${MY_VAR} world") == "hello world"


def test_interpolate_nested(monkeypatch):
    monkeypatch.setenv("BUCKET", "my-bucket")
    data = {"key": "gs://${BUCKET}/path", "nested": {"x": "${BUCKET}"}}
    assert _interpolate(data) == {"key": "gs://my-bucket/path", "nested": {"x": "my-bucket"}}


def test_interpolate_list(monkeypatch):
    monkeypatch.setenv("VAL", "x")
    assert _interpolate(["${VAL}", "literal"]) == ["x", "literal"]


def test_interpolate_passthrough():
    assert _interpolate(42) == 42
    assert _interpolate(None) is None


def test_interpolate_missing_var(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(ValueError, match="MISSING_VAR"):
        _interpolate("${MISSING_VAR}")


def test_load_config_no_retrieval(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    cfg = load_config(str(f))
    assert cfg["model"]["checkpoint"] == "local/path"
    assert cfg["model"]["retrieval"]["enabled"] is False


def test_load_config_retrieval_requires_gate(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: true\n"
        "    backend: gemini_api\nevals: []\noutput_dir: ./out/\n"
    )
    with pytest.raises(ValueError, match="latency_gate_ms"):
        load_config(str(f))


def test_load_config_alias_resolution(tmp_path, monkeypatch):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: base\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    cfg = load_config(str(f))
    assert cfg["model"]["checkpoint"] == "gs://my-bucket/checkpoints/base/moshirag-base-bf16"


def test_load_config_env_var_in_values(tmp_path, monkeypatch):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: false\n"
        "output_dir: ${HOME}/results/\nevals: []\n"
    )
    monkeypatch.setenv("HOME", "/home/user")
    cfg = load_config(str(f))
    assert cfg["output_dir"] == "/home/user/results/"

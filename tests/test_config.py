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


# ── model.retrieval.backend resolution against configs/retrieval_backends.yaml ──


def test_load_config_resolves_known_backend(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: true\n"
        "    backend: gemini_api\n    latency_gate_ms: 5000\nevals: []\noutput_dir: ./out/\n"
    )
    cfg = load_config(str(f))
    resolved = cfg["model"]["retrieval"]["_resolved_backend"]
    assert resolved["type"] == "gemini_api"
    assert resolved["model"] == "gemini-3.5-flash"


def test_load_config_resolves_null_backend(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: false\n"
        "    backend: null\nevals: []\noutput_dir: ./out/\n"
    )
    cfg = load_config(str(f))
    # backend: null (YAML) parses as Python None, which is falsy — the
    # literal string "null" (the key in retrieval_backends.yaml) is a
    # different thing entirely and only resolves if actually written that way.
    assert "_resolved_backend" not in cfg["model"]["retrieval"]


def test_load_config_unknown_backend_raises(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: true\n"
        "    backend: totally_not_a_real_backend\n    latency_gate_ms: 5000\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    with pytest.raises(ValueError, match="totally_not_a_real_backend"):
        load_config(str(f))


def test_load_config_no_backend_field_skips_resolution(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text(
        "model:\n  checkpoint: local/path\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    cfg = load_config(str(f))
    assert "_resolved_backend" not in cfg["model"]["retrieval"]

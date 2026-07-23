"""Tests for scripts/print_demo_env.py — no moshi/torch dependency."""

import sys

import pytest

from scripts.print_demo_env import _generation_flags, _retrieval_env, main
from core.model_interface import _DEFAULT_LLM_BASE_URL, _DEFAULT_LLM_MODEL_NAME


# ── _retrieval_env ───────────────────────────────────────────────────────────


def test_retrieval_env_uses_resolved_backend(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "real-key")
    config = {
        "model": {
            "retrieval": {
                "backend": "gemini_api",
                "_resolved_backend": {
                    "type": "gemini_api",
                    "model": "gemini-3.5-flash",
                    "api_key_env": "GEMINI_API_KEY",
                },
            }
        }
    }
    env = _retrieval_env(config)
    assert env["LLM_MODEL_NAME"] == "gemini-3.5-flash"
    assert env["LLM_BASE_URL"] == _DEFAULT_LLM_BASE_URL  # gemini_api has no base_url field
    assert env["LLM_API_KEY"] == "real-key"


def test_retrieval_env_falls_back_to_defaults_when_retrieval_disabled():
    config = {"model": {"retrieval": {"enabled": False}}}
    env = _retrieval_env(config)
    assert env["LLM_MODEL_NAME"] == _DEFAULT_LLM_MODEL_NAME
    assert env["LLM_BASE_URL"] == _DEFAULT_LLM_BASE_URL


def test_retrieval_env_missing_api_key_env_var_is_empty_string():
    config = {
        "model": {
            "retrieval": {
                "backend": "gemini_api",
                "_resolved_backend": {"type": "gemini_api", "model": "gemini-3.5-flash", "api_key_env": "NOT_SET_VAR"},
            }
        }
    }
    env = _retrieval_env(config)
    assert env["LLM_API_KEY"] == ""


# ── _generation_flags ────────────────────────────────────────────────────────


def test_generation_flags_uses_defaults_when_no_config_block():
    flags = _generation_flags({"model": {}})
    assert "--cfg-coef 1.0" in flags
    assert "--rag-timeout 8.0" in flags
    assert "--vad-threshold 0.5" in flags
    assert "--power-threshold -65" in flags


def test_generation_flags_excludes_tail_silence_steps():
    """No demo (moshi.server) CLI equivalent — respond()-only."""
    flags = _generation_flags({"model": {}})
    assert "tail-silence-steps" not in flags
    assert "tail_silence_steps" not in flags


def test_generation_flags_overrides_merge_over_defaults():
    flags = _generation_flags({"model": {"generation": {"rag_timeout": 12.0}}})
    assert "--rag-timeout 12.0" in flags
    assert "--cfg-coef 1.0" in flags  # untouched default


def test_generation_flags_is_a_pure_name_transform():
    """Field names -> --flag-names via underscore/hyphen swap only, no
    per-field mapping table — confirmed against moshi.server's real argparse
    flag names (see core/model_interface.py's Generation parameters table)."""
    flags = _generation_flags({"model": {}})
    assert "--stt-wait-time 0.5" in flags
    assert "--max-reference-tokens 64" in flags
    assert "--vad-window-size 4" in flags


# ── main() end-to-end ────────────────────────────────────────────────────────


def test_main_prints_shell_sourceable_exports(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  checkpoint: local/whatever\n  retrieval:\n"
        "    enabled: true\n    backend: gemini_api\n    latency_gate_ms: 5000\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    monkeypatch.setattr(sys, "argv", ["print_demo_env.py", "--config", str(cfg)])

    main()

    out = capsys.readouterr().out
    assert "export LLM_MODEL_NAME=gemini-3.5-flash" in out
    assert "export LLM_API_KEY=test-key" in out
    assert "export DEMO_GENERATION_FLAGS=" in out
    assert "--rag-timeout 8.0" in out


def test_main_requires_config_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["print_demo_env.py"])
    with pytest.raises(SystemExit):
        main()

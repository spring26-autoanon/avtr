"""
Tests for evals/runner.py — runner framework, JSON I/O, comparison CLI,
resumability, and delta indicators. Uses a fixture eval class; does not
require any real eval registry modules.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.runner import (
    BaseEval,
    EvalResult,
    _indicator,
    build_model,
    compare_runs,
    find_partial_result,
    load_eval_class,
    run_evals,
    write_results,
)
from core.model_interface import ModelInterface, StubModelAdapter
from core.retrieval_backend import NullBackend


# ── Fixtures ──────────────────────────────────────────────────────────────────


class ConstantEval(BaseEval):
    """Fixture eval that always returns the same canned result."""

    METRIC_DIRECTIONS = {"score": "higher", "latency": "lower"}

    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult:
        n = {"smoke": 5, "sample": 10, "full": 100}.get(mode, 10)
        return EvalResult(
            eval_name="fixture.constant",
            scores={"score": 0.75, "latency": 0.12},
            metadata={"n": n},
            errors=[],
            completed=True,
            last_completed_index=n,
        )


def _minimal_config(tmp_path: Path, evals: list[str] | None = None) -> str:
    cfg = tmp_path / "test.yaml"
    eval_list = evals or []
    cfg.write_text(
        f"model:\n  checkpoint: stub\n  adapter: stub\n  retrieval:\n    enabled: false\n"
        f"evals:\n" + "".join(f"  - {e}\n" for e in eval_list)
        + "output_dir: ./evals/results/\n"
    )
    return str(cfg)


# ── EvalResult ────────────────────────────────────────────────────────────────


def test_eval_result_defaults():
    r = EvalResult(eval_name="test")
    assert r.scores == {}
    assert r.errors == []
    assert r.completed is False
    assert r.last_completed_index == 0


def test_eval_result_fields():
    r = EvalResult(
        eval_name="foo",
        scores={"acc": 0.9},
        completed=True,
        last_completed_index=100,
    )
    assert r.scores["acc"] == 0.9
    assert r.completed is True


# ── BaseEval / format_results ─────────────────────────────────────────────────


def test_base_eval_format_results():
    instance = ConstantEval()
    result = EvalResult(
        eval_name="fixture.constant",
        scores={"score": 0.75, "latency": 0.12},
    )
    lines = instance.format_results(result)
    assert any("0.75" in l for l in lines)
    assert any("higher" in l for l in lines)
    assert any("lower" in l for l in lines)


# ── _indicator (comparison CLI) ───────────────────────────────────────────────


def test_indicator_higher_positive():
    assert _indicator("triviaqa_acc", +0.02) == "✓"


def test_indicator_higher_negative():
    assert _indicator("triviaqa_acc", -0.02) == "↓"


def test_indicator_lower_positive():
    assert _indicator("mean_s", +0.01) == "↓"


def test_indicator_lower_negative():
    assert _indicator("mean_s", -0.01) == "✓"


def test_indicator_unknown_key():
    assert _indicator("unknown_metric", 0.5) == ""


# ── build_model ───────────────────────────────────────────────────────────────


def test_build_model_stub(tmp_path):
    from core.config import load_config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "model:\n  checkpoint: stub\n  adapter: stub\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    cfg = load_config(str(cfg_file))
    model = build_model(cfg)
    assert isinstance(model, StubModelAdapter)


def test_build_model_stub_with_null_backend(tmp_path):
    from core.config import load_config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "model:\n  checkpoint: stub\n  adapter: stub\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    cfg = load_config(str(cfg_file))
    model = build_model(cfg)
    audio, text, meta = model.respond(b"")
    assert isinstance(text, str)


# ── write_results / find_partial_result ───────────────────────────────────────


def test_write_and_read_results(tmp_path):
    from evals import runner as runner_mod
    orig = runner_mod.RESULTS_DIR
    runner_mod.RESULTS_DIR = tmp_path / "results"

    try:
        run_doc = {
            "run_id": "2025-01-01T00-00-00Z",
            "checkpoint": "base",
            "checkpoint_uri": "local/base",
            "config": "configs/test.yaml",
            "mode": "smoke",
            "git_hash": "abc1234",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        results = {
            "fixture.constant": EvalResult(
                eval_name="fixture.constant",
                scores={"score": 0.75},
                completed=True,
                last_completed_index=5,
            )
        }
        out = runner_mod.RESULTS_DIR / "run-test.json"
        write_results(run_doc, results, completed=True, output_path=out)

        with open(out) as f:
            data = json.load(f)

        assert data["completed"] is True
        assert data["evals"]["fixture.constant"]["scores"]["score"] == 0.75
        assert data["evals"]["fixture.constant"]["last_completed_index"] == 5
    finally:
        runner_mod.RESULTS_DIR = orig


def test_find_partial_result(tmp_path):
    from evals import runner as runner_mod
    orig = runner_mod.RESULTS_DIR
    runner_mod.RESULTS_DIR = tmp_path / "results"
    runner_mod.RESULTS_DIR.mkdir()

    try:
        run_doc = {
            "run_id": "2025-01-01T00-00-00Z",
            "checkpoint": "base",
            "checkpoint_uri": "local/base",
            "config": "configs/test.yaml",
            "mode": "smoke",
            "git_hash": "abc1234",
            "timestamp": "2025-01-01T00:00:00Z",
            "completed": False,
            "evals": {
                "fixture.constant": {
                    "scores": {"score": 0.5},
                    "metadata": {},
                    "errors": [],
                    "completed": False,
                    "last_completed_index": 3,
                }
            },
        }
        out = runner_mod.RESULTS_DIR / "run-test.json"
        with open(out, "w") as f:
            json.dump(run_doc, f)

        partial = find_partial_result("base", "configs/test.yaml", "smoke")
        assert partial is not None
        assert partial["evals"]["fixture.constant"]["last_completed_index"] == 3
    finally:
        runner_mod.RESULTS_DIR = orig


def test_find_partial_result_none_when_completed(tmp_path):
    from evals import runner as runner_mod
    orig = runner_mod.RESULTS_DIR
    runner_mod.RESULTS_DIR = tmp_path / "results"
    runner_mod.RESULTS_DIR.mkdir()

    try:
        completed_doc = {
            "checkpoint": "base",
            "config": "configs/test.yaml",
            "mode": "smoke",
            "completed": True,
            "evals": {},
        }
        out = runner_mod.RESULTS_DIR / "run-done.json"
        with open(out, "w") as f:
            json.dump(completed_doc, f)

        assert find_partial_result("base", "configs/test.yaml", "smoke") is None
    finally:
        runner_mod.RESULTS_DIR = orig


# ── Full run_evals pipeline ───────────────────────────────────────────────────


def test_run_evals_end_to_end(tmp_path, capsys):
    from evals import runner as runner_mod
    orig_results = runner_mod.RESULTS_DIR
    runner_mod.RESULTS_DIR = tmp_path / "results"

    cfg_path = _minimal_config(tmp_path, evals=["fixture.constant"])

    try:
        with patch.object(runner_mod, "load_eval_class", return_value=ConstantEval):
            run_evals(cfg_path, mode="smoke", spot_check=False)

        out_files = list(runner_mod.RESULTS_DIR.glob("run-*.json"))
        assert len(out_files) == 1

        with open(out_files[0]) as f:
            data = json.load(f)

        assert data["completed"] is True
        assert data["mode"] == "smoke"
        assert "fixture.constant" in data["evals"]
        assert data["evals"]["fixture.constant"]["scores"]["score"] == 0.75
        assert data["evals"]["fixture.constant"]["completed"] is True
    finally:
        runner_mod.RESULTS_DIR = orig_results


def test_run_evals_skips_missing_eval(tmp_path, capsys):
    from evals import runner as runner_mod
    orig_results = runner_mod.RESULTS_DIR
    runner_mod.RESULTS_DIR = tmp_path / "results"

    cfg_path = _minimal_config(tmp_path, evals=["does.not.exist"])

    try:
        run_evals(cfg_path, mode="smoke", spot_check=False)
        captured = capsys.readouterr()
        assert "skipped" in captured.out
    finally:
        runner_mod.RESULTS_DIR = orig_results


def test_run_evals_resumes_completed_evals(tmp_path, capsys):
    from evals import runner as runner_mod
    orig_results = runner_mod.RESULTS_DIR
    runner_mod.RESULTS_DIR = tmp_path / "results"
    runner_mod.RESULTS_DIR.mkdir()

    checkpoint_name = "stub"
    cfg_path = _minimal_config(tmp_path, evals=["fixture.constant"])

    # Write a fake partial result showing fixture.constant already done
    partial = {
        "checkpoint": checkpoint_name,
        "config": cfg_path,
        "mode": "smoke",
        "run_id": "old",
        "checkpoint_uri": "stub",
        "git_hash": "abc",
        "timestamp": "2025-01-01T00:00:00Z",
        "completed": False,
        "evals": {
            "fixture.constant": {
                "scores": {"score": 0.9},
                "metadata": {},
                "errors": [],
                "completed": True,
                "last_completed_index": 5,
            }
        },
    }
    with open(runner_mod.RESULTS_DIR / "run-old.json", "w") as f:
        json.dump(partial, f)

    call_count = {"n": 0}

    class CountingEval(ConstantEval):
        def run(self, model, config, mode):
            call_count["n"] += 1
            return super().run(model, config, mode)

    try:
        with patch.object(runner_mod, "load_eval_class", return_value=CountingEval):
            run_evals(cfg_path, mode="smoke", spot_check=False)

        # The eval was already completed in the partial — should not run again
        assert call_count["n"] == 0
        captured = capsys.readouterr()
        assert "resumed" in captured.out
    finally:
        runner_mod.RESULTS_DIR = orig_results


# ── compare_runs ──────────────────────────────────────────────────────────────


def _write_run(path: Path, checkpoint: str, resp_acc: float, total_p95: float, gate_breached: bool = False) -> None:
    data = {
        "run_id": "test",
        "checkpoint": checkpoint,
        "config": "configs/baseline_with_retrieval.yaml",
        "mode": "sample",
        "git_hash": "abc",
        "timestamp": "2025-01-01T00:00:00Z",
        "completed": True,
        "evals": {
            "knowledge.halu_eval_audio": {
                "scores": {"ref_acc": 0.42, "resp_acc": resp_acc},
                "metadata": {"n": 100, "judge_model": "gemini-2.0-flash"},
                "errors": [],
                "completed": True,
                "last_completed_index": 100,
            },
            "latency.retrieval_breakdown": {
                "scores": {"total_p95_s": total_p95},
                "metadata": {"latency_gate_ms": 1500, "gate_breached_p95": gate_breached},
                "errors": [],
                "completed": True,
                "last_completed_index": 200,
            },
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)


def test_compare_runs_output(tmp_path, capsys):
    a = tmp_path / "run-a.json"
    b = tmp_path / "run-b.json"
    _write_run(a, "base", resp_acc=0.363, total_p95=1.82)
    _write_run(b, "lora-v1", resp_acc=0.379, total_p95=1.77)

    compare_runs(str(a), str(b))
    out = capsys.readouterr().out

    assert "base" in out
    assert "lora-v1" in out
    assert "rag_lift_pp" in out
    assert "+1.6pp" in out
    assert "knowledge.halu_eval_audio" in out


def test_compare_runs_gate_warning(tmp_path, capsys):
    a = tmp_path / "run-a.json"
    b = tmp_path / "run-b.json"
    _write_run(a, "base", resp_acc=0.36, total_p95=1.82, gate_breached=True)
    _write_run(b, "lora-v1", resp_acc=0.37, total_p95=1.77, gate_breached=True)

    compare_runs(str(a), str(b))
    out = capsys.readouterr().out
    assert "⚠" in out
    assert "exceeds" in out
    assert "infrastructure" in out

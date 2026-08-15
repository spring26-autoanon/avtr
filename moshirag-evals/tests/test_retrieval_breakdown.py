"""
Tests for evals/registry/latency/retrieval_breakdown.py — per-stage
aggregation, rag_triggered filtering, latency-gate detection, resumability,
and format_results. Mocks _load_rows/_load_audio_bytes and the model — no
network, no GPU. context_injection_s is read directly from
resp_metadata["conditioning_latency_s"] (the real ARC-encoder /embed round
trip timed in core/model_interface.py), so no clock patching is needed here.
"""

from unittest.mock import patch

import pytest

from evals.registry.latency import retrieval_breakdown as rb
from evals.registry.latency import ttfat
from evals.runner import EvalResult


def _fake_load_rows(subset_cfg, limit, _n=3):
    key = subset_cfg["question_col"]
    rows = [{key: f"{subset_cfg['csv']}::q{i}"} for i in range(_n)]
    return rows if limit is None else rows[:limit]


class _FakeModel:
    def __init__(
        self,
        rag_triggered_seq=None,
        asr_wait_s=0.10,
        retrieval_latency_s=0.20,
        conditioning_latency_s=0.05,
        retrieval_context="user: some question",
    ):
        self.rag_triggered_seq = rag_triggered_seq
        self.asr_wait_s = asr_wait_s
        self.retrieval_latency_s = retrieval_latency_s
        self.conditioning_latency_s = conditioning_latency_s
        self.retrieval_context = retrieval_context
        self.calls = 0

    def respond(self, audio_in: bytes):
        i = self.calls
        self.calls += 1
        triggered = self.rag_triggered_seq[i] if self.rag_triggered_seq is not None else True
        metadata = {
            "rag_triggered": triggered,
            "asr_wait_s": self.asr_wait_s,
            "retrieval_latency_s": self.retrieval_latency_s,
            "conditioning_latency_s": self.conditioning_latency_s,
            "retrieval_context": self.retrieval_context,
        }
        return b"", "some answer", metadata


def _patched_run(model, config, mode, n_per_subset=3):
    # _load_rows lives in ttfat's namespace (used internally by the
    # _load_question_pool retrieval_breakdown imports from there) — patch
    # it there, not on retrieval_breakdown itself, which never imports it.
    with patch.object(ttfat, "_load_rows", side_effect=lambda cfg, limit: _fake_load_rows(cfg, limit, _n=n_per_subset)), \
         patch.object(rb, "_load_audio_bytes", return_value=b"wav-bytes"):
        return rb.RetrievalBreakdownEval().run(model, config, mode)


# ── RetrievalBreakdownEval.run() ─────────────────────────────────────────────


def test_run_computes_all_four_stages():
    model = _FakeModel(asr_wait_s=0.10, retrieval_latency_s=0.20, conditioning_latency_s=0.05)
    result = _patched_run(model, {}, "tiny", n_per_subset=1)  # 1 question total

    assert result.metadata["n"] == 1
    assert result.scores["asr_wait_mean_s"] == pytest.approx(0.10)
    assert result.scores["api_call_mean_s"] == pytest.approx(0.20)
    assert result.scores["context_injection_mean_s"] == pytest.approx(0.05)
    assert result.scores["total_mean_s"] == pytest.approx(0.35)


def test_run_p95_matches_mean_for_uniform_values():
    model = _FakeModel(asr_wait_s=0.10, retrieval_latency_s=0.20, conditioning_latency_s=0.05)
    result = _patched_run(model, {}, "smoke", n_per_subset=3)  # 5 identical questions

    assert result.metadata["n"] == 5
    assert result.scores["asr_wait_p95_s"] == pytest.approx(0.10)
    assert result.scores["total_p95_s"] == pytest.approx(0.35)


def test_run_excludes_non_triggered_questions_from_stage_stats():
    # 5 questions (smoke), only 3 trigger retrieval
    model = _FakeModel(rag_triggered_seq=[True, False, True, False, True])
    result = _patched_run(model, {}, "smoke", n_per_subset=3)

    assert result.metadata["n"] == 3  # only the triggered ones count
    assert result.last_completed_index == 5  # but all 5 were attempted
    non_triggered = [e for e in result.transcript if not e["rag_triggered"]]
    assert len(non_triggered) == 2
    assert "asr_wait_s" not in non_triggered[0]


def test_run_no_triggers_at_all_yields_empty_scores():
    model = _FakeModel(rag_triggered_seq=[False] * 5)
    result = _patched_run(model, {}, "smoke", n_per_subset=3)

    assert result.scores == {}
    assert result.metadata["n"] == 0
    assert result.last_completed_index == 5


def test_run_records_per_question_errors_without_aborting():
    model = _FakeModel()

    def _fake_respond(audio_in):
        raise RuntimeError("respond failed")

    model.respond = _fake_respond
    result = _patched_run(model, {}, "smoke", n_per_subset=3)

    assert result.scores == {}
    assert len(result.errors) == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["error"] == "respond failed"
    assert "rag_triggered" not in result.transcript[0]


# ── latency_gate_ms / gate_breached_p95 ──────────────────────────────────────


def test_run_flags_gate_breach():
    model = _FakeModel(asr_wait_s=1.0, retrieval_latency_s=1.0)  # total ~2.05s
    config = {"model": {"retrieval": {"latency_gate_ms": 1500}}}
    result = _patched_run(model, config, "tiny", n_per_subset=1)

    assert result.metadata["latency_gate_ms"] == 1500
    assert result.metadata["gate_breached_p95"] is True
    assert any("exceeds" in e for e in result.errors)


def test_run_no_gate_breach_when_under_gate():
    model = _FakeModel(asr_wait_s=0.05, retrieval_latency_s=0.05)
    config = {"model": {"retrieval": {"latency_gate_ms": 1500}}}
    result = _patched_run(model, config, "tiny", n_per_subset=1)

    assert result.metadata["latency_gate_ms"] == 1500
    assert "gate_breached_p95" not in result.metadata
    assert result.errors == []


def test_run_no_gate_configured_skips_gate_check_entirely():
    model = _FakeModel(asr_wait_s=10.0, retrieval_latency_s=10.0)  # would breach any real gate
    result = _patched_run(model, {}, "tiny", n_per_subset=1)

    assert "latency_gate_ms" not in result.metadata
    assert "gate_breached_p95" not in result.metadata
    assert result.errors == []


# ── resumability ──────────────────────────────────────────────────────────────


def test_run_resumes_without_recalling_model_for_done_items():
    model = _FakeModel(asr_wait_s=0.10, retrieval_latency_s=0.20)
    prior = EvalResult(
        eval_name="latency.retrieval_breakdown",
        scores={"asr_wait_mean_s": 0.10, "asr_wait_p95_s": 0.10},
        metadata={
            "n": 3,
            "_progress": {
                "total": 3,
                "asr_wait_values": [0.10, 0.10, 0.10],
                "api_call_values": [0.20, 0.20, 0.20],
                "context_injection_values": [0.05, 0.05, 0.05],
                "total_values": [0.35, 0.35, 0.35],
            },
        },
        completed=False,
        last_completed_index=3,
        transcript=[{"index": i, "rag_triggered": True} for i in range(3)],
    )

    result = _patched_run(model, {"_prior_result": prior}, "smoke", n_per_subset=3)  # 5 total for smoke

    assert model.calls == 2  # 3 already done -> only 2 new model calls
    assert result.metadata["n"] == 5
    assert result.last_completed_index == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["rag_triggered"] is True  # untouched prior entry


# ── format_results ────────────────────────────────────────────────────────────


def test_format_results_layout():
    result = EvalResult(
        eval_name="latency.retrieval_breakdown",
        scores={
            "asr_wait_mean_s": 0.51, "asr_wait_p95_s": 0.63,
            "api_call_mean_s": 0.87, "api_call_p95_s": 1.24,
            "context_injection_mean_s": 0.03, "context_injection_p95_s": 0.05,
            "total_mean_s": 1.41, "total_p95_s": 1.82,
        },
        metadata={"n": 200},
    )
    lines = rb.RetrievalBreakdownEval().format_results(result)
    joined = "\n".join(lines)
    assert "asr transcription wait" in joined
    assert "gemini api call" in joined
    assert "context injection" in joined
    assert "total retrieval delay" in joined
    assert "0.51s" in joined and "0.63s" in joined
    assert "1.41s" in joined and "1.82s" in joined


def test_format_results_empty_when_no_scores():
    result = EvalResult(eval_name="latency.retrieval_breakdown", scores={}, metadata={"n": 0})
    assert rb.RetrievalBreakdownEval().format_results(result) == []

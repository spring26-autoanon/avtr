"""
Tests for evals/registry/latency/ttfat.py — question pool interleaving,
percentile math, TTFAT aggregation, resumability, and format_results. Mocks
_load_rows/_load_audio_bytes (imported from knowledge.open_audio_bench) and
the model — no network, no GPU.
"""

from unittest.mock import patch

import pytest

from evals.registry.latency import ttfat
from evals.runner import EvalResult


# ── _percentile ───────────────────────────────────────────────────────────────


def test_percentile_empty_returns_zero():
    assert ttfat._percentile([], 95) == 0.0


def test_percentile_single_value():
    assert ttfat._percentile([0.5], 50) == 0.5
    assert ttfat._percentile([0.5], 95) == 0.5


def test_percentile_p50_of_sorted_list():
    assert ttfat._percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_p95_interpolates():
    values = list(range(1, 101))  # 1..100
    # k = 99 * 0.95 = 94.05 -> interpolate between index 94 (value 95) and 95 (value 96)
    assert ttfat._percentile([float(v) for v in values], 95) == pytest.approx(95.05)


# ── _load_question_pool ──────────────────────────────────────────────────────


def _fake_load_rows(subset_cfg, limit, _n=3):
    key = subset_cfg["question_col"]
    rows = [{key: f"{subset_cfg['csv']}::q{i}"} for i in range(_n)]
    return rows if limit is None else rows[:limit]


def test_load_question_pool_round_robins_across_subsets():
    with patch.object(ttfat, "_load_rows", side_effect=_fake_load_rows):
        pool = ttfat._load_question_pool(None)

    subset_order = [entry[0] for entry in pool[:3]]
    assert subset_order == ["triviaqa", "webq", "llamaq"]
    assert len(pool) == 9  # 3 subsets x 3 rows each


def test_load_question_pool_truncates_to_limit():
    with patch.object(ttfat, "_load_rows", side_effect=_fake_load_rows):
        pool = ttfat._load_question_pool(2)

    assert len(pool) == 2
    assert [entry[0] for entry in pool] == ["triviaqa", "webq"]


def test_load_question_pool_handles_uneven_subset_sizes():
    def _uneven(subset_cfg, limit):
        n = {"triviaqa": 1, "webq": 2, "llamaq": 3}[
            "triviaqa" if "trivia" in subset_cfg["csv"] else "webq" if "web" in subset_cfg["csv"] else "llamaq"
        ]
        return _fake_load_rows(subset_cfg, limit, _n=n)

    with patch.object(ttfat, "_load_rows", side_effect=_uneven):
        pool = ttfat._load_question_pool(None)

    assert len(pool) == 6  # 1 + 2 + 3


# ── TtfatEval.run() ───────────────────────────────────────────────────────────


class _FakeModel:
    def __init__(self, ttfat_values=None, degenerate_indices=frozenset()):
        self.ttfat_values = ttfat_values
        self.degenerate_indices = degenerate_indices
        self.calls = 0

    def respond(self, audio_in: bytes):
        i = self.calls
        self.calls += 1
        ttfat_s = self.ttfat_values[i] if self.ttfat_values else 0.05
        metadata = {"ttfat_s": ttfat_s, "degenerate_silence": i in self.degenerate_indices}
        return b"", "some answer", metadata


def _patched_run(model, config, mode, n_per_subset=3):
    with patch.object(ttfat, "_load_rows", side_effect=lambda cfg, limit: _fake_load_rows(cfg, limit, _n=n_per_subset)), \
         patch.object(ttfat, "_load_audio_bytes", return_value=b"wav-bytes"):
        return ttfat.TtfatEval().run(model, config, mode)


def test_run_computes_mean_p50_p95():
    model = _FakeModel(ttfat_values=[0.02, 0.04, 0.06])
    result = _patched_run(model, {}, "tiny", n_per_subset=1)  # 3 total (1 per subset), tiny limits to 1

    assert result.metadata["n"] == 1
    assert result.scores["mean_s"] == pytest.approx(0.02)
    assert model.calls == 1


def test_run_smoke_mode_aggregates_across_pool():
    model = _FakeModel(ttfat_values=[0.01, 0.02, 0.03, 0.04, 0.05])
    result = _patched_run(model, {}, "smoke", n_per_subset=3)  # smoke -> limit 5

    assert result.metadata["n"] == 5
    assert result.scores["mean_s"] == pytest.approx(0.03)
    assert result.scores["p50_s"] == pytest.approx(0.03)
    assert result.completed is True
    assert result.errors == []


def test_run_records_errors_without_aborting_and_excludes_from_n():
    model = _FakeModel()

    def _fake_respond(audio_in):
        raise RuntimeError("respond failed")

    model.respond = _fake_respond

    result = _patched_run(model, {}, "smoke", n_per_subset=3)

    assert result.scores == {}  # no valid measurements at all
    assert result.metadata["n"] == 0
    assert len(result.errors) == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["error"] == "respond failed"


def test_run_partial_errors_only_count_successful_measurements_in_n():
    model = _FakeModel(ttfat_values=[0.01, 0.02, 0.03, 0.04, 0.05])
    call_count = {"n": 0}
    real_respond = model.respond

    def _sometimes_fails(audio_in):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("transient failure")
        return real_respond(audio_in)

    model.respond = _sometimes_fails
    result = _patched_run(model, {}, "smoke", n_per_subset=3)

    assert result.metadata["n"] == 4  # 5 attempted, 1 failed
    assert len(result.errors) == 1


def test_run_flags_degenerate_silence():
    model = _FakeModel(ttfat_values=[0.01] * 5, degenerate_indices={0, 2})
    result = _patched_run(model, {}, "smoke", n_per_subset=3)

    assert result.metadata["degenerate_silence_count"] == 2
    flagged = [e for e in result.transcript if e["degenerate_silence"]]
    assert len(flagged) == 2


def test_run_no_degenerate_silence_key_when_none_occur():
    model = _FakeModel(ttfat_values=[0.01, 0.02, 0.03])
    result = _patched_run(model, {}, "tiny", n_per_subset=1)

    assert "degenerate_silence_count" not in result.metadata


def test_run_resumes_without_recalling_model_for_done_items():
    model = _FakeModel(ttfat_values=[0.04, 0.05])
    prior = EvalResult(
        eval_name="latency.ttfat",
        scores={"mean_s": 0.03, "p50_s": 0.03, "p95_s": 0.03},
        metadata={"n": 3, "_progress": {"total": 3, "ttfat_values": [0.02, 0.03, 0.04], "degenerate_silence": 0}},
        completed=False,
        last_completed_index=3,
        transcript=[{"index": i, "ttfat_s": 0.03} for i in range(3)],
    )

    result = _patched_run(model, {"_prior_result": prior}, "smoke", n_per_subset=3)  # 5 total for smoke

    # 3 already done -> only 2 new model calls
    assert model.calls == 2
    assert result.metadata["n"] == 5
    assert result.last_completed_index == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["ttfat_s"] == 0.03  # untouched prior entry


# ── format_results ────────────────────────────────────────────────────────────


def test_format_results_layout():
    result = EvalResult(
        eval_name="latency.ttfat",
        scores={"mean_s": 0.04, "p50_s": 0.03, "p95_s": 0.07},
        metadata={"n": 200},
    )
    lines = ttfat.TtfatEval().format_results(result)
    joined = "\n".join(lines)
    assert "0.04s" in joined
    assert "0.03s" in joined
    assert "0.07s" in joined
    assert "(n=200)" in joined


def test_format_results_empty_when_no_scores():
    result = EvalResult(eval_name="latency.ttfat", scores={}, metadata={"n": 0})
    assert ttfat.TtfatEval().format_results(result) == []

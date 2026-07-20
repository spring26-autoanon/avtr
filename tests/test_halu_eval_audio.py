"""
Tests for evals/registry/knowledge/halu_eval_audio.py — judge-verdict
parsing, ref/resp scoring, resumability, and format_results. Mocks
_load_rows and the Gemini judge call — no network, no GPU.
"""

from unittest.mock import patch

import pytest

from evals.registry.knowledge import halu_eval_audio as hea
from evals.runner import EvalResult


# ── _judge ────────────────────────────────────────────────────────────────────


def test_judge_parses_correct_verdict():
    with patch.object(hea, "call_gemini", return_value="reasoning... the score is [Correct]"):
        assert hea._judge("q", "paris", "Paris") is True


def test_judge_parses_incorrect_verdict():
    with patch.object(hea, "call_gemini", return_value="reasoning... the score is [Incorrect]"):
        assert hea._judge("q", "paris", "London") is False


def test_judge_raises_on_unparseable_response():
    with patch.object(hea, "call_gemini", return_value="no verdict here"):
        with pytest.raises(ValueError, match="no parseable verdict"):
            hea._judge("q", "paris", "London")


# ── HaluEvalAudioEval.run() ───────────────────────────────────────────────────


class _FakeModel:
    def __init__(self, text_out="some answer", retrieval_text="some reference"):
        self.text_out = text_out
        self.retrieval_text = retrieval_text
        self.calls = 0

    def respond(self, audio_in: bytes):
        self.calls += 1
        metadata = {"retrieval_context": "ctx", "retrieval_text": self.retrieval_text}
        return b"", self.text_out, metadata


def _fake_rows(limit):
    n = limit if limit is not None else 3
    return [
        {
            "text": f"question {i}",
            "answer": f"answer {i}",
            "knowledge": f"knowledge {i}",
            "audio": {"bytes": b"wav-bytes"},
        }
        for i in range(n)
    ]


def test_run_smoke_mode_scores_ref_and_resp():
    model = _FakeModel()
    with patch.object(hea, "_load_rows", side_effect=_fake_rows), \
         patch.object(hea, "call_gemini", return_value="the score is [Correct]"):
        result = hea.HaluEvalAudioEval().run(model, {}, "smoke")

    assert result.scores["resp_acc"] == 1.0
    assert result.scores["ref_acc"] == 1.0
    assert result.metadata["n"] == 5
    assert result.metadata["judge_model"] == hea._JUDGE_MODEL
    assert result.last_completed_index == 5
    assert result.completed is True
    assert result.errors == []
    assert model.calls == 5


def test_run_tiny_mode_limits_to_one():
    model = _FakeModel()
    with patch.object(hea, "_load_rows", side_effect=_fake_rows), \
         patch.object(hea, "call_gemini", return_value="the score is [Correct]"):
        result = hea.HaluEvalAudioEval().run(model, {}, "tiny")

    assert result.metadata["n"] == 1
    assert result.last_completed_index == 1
    assert model.calls == 1


def test_run_empty_retrieval_text_skips_judge_call_and_scores_incorrect():
    model = _FakeModel(retrieval_text="")
    judge_calls = {"n": 0}

    def _fake_call_gemini(model_name, prompt):
        judge_calls["n"] += 1
        return "the score is [Correct]"

    with patch.object(hea, "_load_rows", side_effect=_fake_rows), \
         patch.object(hea, "call_gemini", side_effect=_fake_call_gemini):
        result = hea.HaluEvalAudioEval().run(model, {}, "smoke")

    assert result.scores["ref_acc"] == 0.0
    assert result.scores["resp_acc"] == 1.0
    # Only the resp_acc judge call happens per question (5), not 10 —
    # ref judging is skipped entirely for empty retrieval text.
    assert judge_calls["n"] == 5


def test_run_records_errors_without_aborting():
    model = _FakeModel()

    def _boom(limit):
        raise RuntimeError("download failed")

    with patch.object(hea, "_load_rows", side_effect=_boom):
        with pytest.raises(RuntimeError):
            hea.HaluEvalAudioEval().run(model, {}, "smoke")


def test_run_records_per_question_errors_without_aborting():
    model = _FakeModel()

    def _fake_respond(audio_in):
        raise RuntimeError("respond failed")

    model.respond = _fake_respond

    with patch.object(hea, "_load_rows", side_effect=_fake_rows):
        result = hea.HaluEvalAudioEval().run(model, {}, "smoke")

    assert result.scores["resp_acc"] == 0.0
    assert result.scores["ref_acc"] == 0.0
    assert len(result.errors) == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["error"] == "respond failed"
    assert "resp_verdict" not in result.transcript[0]


def test_run_resumes_without_recalling_model_for_done_items():
    model = _FakeModel()
    prior = EvalResult(
        eval_name="knowledge.halu_eval_audio",
        scores={"ref_acc": 1.0, "resp_acc": 1.0},
        metadata={
            "n": 3,
            "judge_model": hea._JUDGE_MODEL,
            "_progress": {"total": 3, "correct_ref": 3, "correct_resp": 3},
        },
        completed=False,
        last_completed_index=3,
        transcript=[{"index": i, "question": f"prior q{i}"} for i in range(3)],
    )

    with patch.object(hea, "_load_rows", side_effect=_fake_rows), \
         patch.object(hea, "call_gemini", return_value="the score is [Correct]"):
        result = hea.HaluEvalAudioEval().run(model, {"_prior_result": prior}, "smoke")

    # 3 already done -> only 2 new model calls (5 total smoke rows)
    assert model.calls == 2
    assert result.metadata["n"] == 5
    assert result.scores["resp_acc"] == 1.0
    assert result.scores["ref_acc"] == 1.0
    assert result.last_completed_index == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["question"] == "prior q0"  # untouched prior entry


# ── format_results ────────────────────────────────────────────────────────────


def test_format_results_layout():
    result = EvalResult(
        eval_name="knowledge.halu_eval_audio",
        scores={"ref_acc": 0.420, "resp_acc": 0.363},
        metadata={"n": 100, "judge_model": "gemini-3.5-flash"},
    )
    lines = hea.HaluEvalAudioEval().format_results(result)
    joined = "\n".join(lines)
    assert "42.0%" in joined
    assert "36.3%" in joined
    assert "(n=100)" in joined
    assert "gemini-3.5-flash" in joined

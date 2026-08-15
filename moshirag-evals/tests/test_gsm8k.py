"""
Tests for evals/registry/knowledge/gsm8k.py — ground-truth parsing, answer
extraction, exact-match scoring, resumability, and format_results. Mocks
_load_rows, call_gemini, and the TTS backend — no network, no GPU.
"""

from unittest.mock import patch

import pytest

from evals.registry.knowledge import gsm8k
from evals.runner import DEFAULT_JUDGE_MODEL, EvalResult

_SAMPLE_ANSWER = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72"


@pytest.fixture(autouse=True)
def _mock_tts_backend():
    """
    Gsm8kEval.run() builds a real TTSBackend (a real genai.Client) once per
    run via build_tts_backend() — none of these tests need that (they mock
    synthesize_cached separately, which is what actually matters for
    scoring behavior), so avoid it uniformly rather than per-test.
    """
    with patch.object(gsm8k, "build_tts_backend", return_value=object()):
        yield


# ── _parse_ground_truth ──────────────────────────────────────────────────────


def test_parse_ground_truth_extracts_final_number():
    assert gsm8k._parse_ground_truth(_SAMPLE_ANSWER) == 72.0


def test_parse_ground_truth_strips_commas():
    assert gsm8k._parse_ground_truth("some reasoning\n#### 1,234") == 1234.0


def test_parse_ground_truth_handles_negative_and_decimal():
    assert gsm8k._parse_ground_truth("#### -3.5") == -3.5


def test_parse_ground_truth_raises_on_missing_marker():
    with pytest.raises(ValueError, match="no parseable"):
        gsm8k._parse_ground_truth("no marker here at all")


# ── _extract_model_answer ─────────────────────────────────────────────────────


def test_extract_model_answer_parses_plain_number():
    with patch.object(gsm8k, "call_gemini", return_value="72"):
        assert gsm8k._extract_model_answer("q", "the answer is 72", DEFAULT_JUDGE_MODEL) == 72.0


def test_extract_model_answer_strips_currency_and_commas():
    with patch.object(gsm8k, "call_gemini", return_value="$1,234"):
        assert gsm8k._extract_model_answer("q", "resp", DEFAULT_JUDGE_MODEL) == 1234.0


def test_extract_model_answer_none_when_extractor_says_none():
    with patch.object(gsm8k, "call_gemini", return_value="NONE"):
        assert gsm8k._extract_model_answer("q", "resp", DEFAULT_JUDGE_MODEL) is None


def test_extract_model_answer_none_when_unparseable():
    with patch.object(gsm8k, "call_gemini", return_value="no number in here"):
        assert gsm8k._extract_model_answer("q", "resp", DEFAULT_JUDGE_MODEL) is None


# ── Gsm8kEval.run() ───────────────────────────────────────────────────────────


class _FakeModel:
    def __init__(self, text_out="the answer is 72"):
        self.text_out = text_out
        self.calls = 0

    def respond(self, audio_in: bytes):
        self.calls += 1
        metadata = {"retrieval_context": "", "retrieval_text": ""}
        return b"", self.text_out, metadata


def _fake_rows(limit):
    n = limit if limit is not None else 3
    return [{"question": f"question {i}", "answer": f"reasoning {i}\n#### 72"} for i in range(n)]


_TTS_CFG = {"tts": {"backend": "gemini_tts", "_resolved_backend": {"type": "gemini_tts", "model": "m", "voice": "Kore"}}}


def _run_config(**overrides):
    cfg = {**_TTS_CFG, **overrides}
    return cfg


def test_run_requires_resolved_tts_backend():
    model = _FakeModel()
    with pytest.raises(ValueError, match="requires a resolved tts.backend"):
        gsm8k.Gsm8kEval().run(model, {}, "smoke")


def test_run_smoke_mode_scores_correct():
    model = _FakeModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "smoke")

    assert result.scores["gsm8k_acc"] == 1.0
    assert result.metadata["n"] == 5
    assert result.metadata["judge_model"] == DEFAULT_JUDGE_MODEL
    assert result.last_completed_index == 5
    assert result.completed is True
    assert result.errors == []
    assert model.calls == 5


def test_run_tiny_mode_limits_to_one():
    model = _FakeModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    assert result.metadata["n"] == 1
    assert result.last_completed_index == 1
    assert model.calls == 1


def test_run_scores_incorrect_on_wrong_answer():
    model = _FakeModel(text_out="the answer is 99")
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="99"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    assert result.scores["gsm8k_acc"] == 0.0
    assert result.transcript[0]["verdict"] == "incorrect"


def test_run_scores_incorrect_when_extraction_returns_none():
    model = _FakeModel(text_out="I don't know")
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="NONE"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    assert result.scores["gsm8k_acc"] == 0.0
    assert result.transcript[0]["extracted_answer"] is None
    assert "error" not in result.transcript[0]  # a legitimate scoring outcome, not an error


def test_run_uses_judge_model_from_config():
    model = _FakeModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72") as mock_call, \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(judge={"model": "custom-judge"}), "tiny")

    assert result.metadata["judge_model"] == "custom-judge"
    assert mock_call.call_args[0][0] == "custom-judge"


def test_run_passes_resolved_backend_to_synthesize_cached():
    model = _FakeModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes") as mock_synth:
        gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    args = mock_synth.call_args[0]
    assert args[0] == "question 0"
    assert args[2] == "gemini_tts"
    assert args[3] == {"type": "gemini_tts", "model": "m", "voice": "Kore"}


def test_run_reports_tts_backend_metadata():
    model = _FakeModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    assert result.metadata["tts_backend"] == {"name": "gemini_tts", "type": "gemini_tts", "model": "m", "voice": "Kore"}


def test_run_records_per_question_errors_without_aborting():
    model = _FakeModel()

    def _fake_respond(audio_in):
        raise RuntimeError("respond failed")

    model.respond = _fake_respond

    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "smoke")

    assert result.scores["gsm8k_acc"] == 0.0
    assert len(result.errors) == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["error"] == "respond failed"
    assert "verdict" not in result.transcript[0]


def test_run_ground_truth_parse_failure_recorded_as_error_not_abort():
    model = _FakeModel()

    def _bad_rows(limit):
        return [{"question": "q0", "answer": "no marker here"}]

    with patch.object(gsm8k, "_load_rows", side_effect=_bad_rows), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    assert result.scores["gsm8k_acc"] == 0.0
    assert len(result.errors) == 1
    assert "no parseable" in result.errors[0]
    assert model.calls == 0  # failed before ever calling respond()


def test_run_resumes_without_recalling_model_for_done_items():
    model = _FakeModel()
    prior = EvalResult(
        eval_name="knowledge.gsm8k",
        scores={"gsm8k_acc": 1.0},
        metadata={
            "n": 3,
            "judge_model": DEFAULT_JUDGE_MODEL,
            "_progress": {"total": 3, "correct": 3, "degenerate_silence": 0},
        },
        completed=False,
        last_completed_index=3,
        transcript=[{"index": i, "question": f"prior q{i}"} for i in range(3)],
    )

    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(_prior_result=prior), "smoke")

    # 3 already done -> only 2 new model calls (5 total smoke rows)
    assert model.calls == 2
    assert result.metadata["n"] == 5
    assert result.scores["gsm8k_acc"] == 1.0
    assert result.last_completed_index == 5
    assert len(result.transcript) == 5
    assert result.transcript[0]["question"] == "prior q0"  # untouched prior entry


# ── degenerate_silence flagging ──────────────────────────────────────────────


class _SometimesSilentModel:
    def __init__(self):
        self.calls = 0

    def respond(self, audio_in: bytes):
        i = self.calls
        self.calls += 1
        degenerate = i % 2 == 0
        text_out = "" if degenerate else "the answer is 72"
        return b"", text_out, {"retrieval_context": "", "retrieval_text": "", "degenerate_silence": degenerate}


def test_run_flags_degenerate_silence_in_transcript_and_metadata():
    model = _SometimesSilentModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="NONE"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "smoke")

    flagged = [e for e in result.transcript if e["degenerate_silence"]]
    assert len(flagged) == 3  # indices 0, 2, 4 of 5
    assert result.metadata["degenerate_silence_count"] == 3


def test_run_no_degenerate_silence_key_when_none_occur():
    model = _FakeModel()
    with patch.object(gsm8k, "_load_rows", side_effect=_fake_rows), \
         patch.object(gsm8k, "call_gemini", return_value="72"), \
         patch.object(gsm8k, "synthesize_cached", return_value=b"wav-bytes"):
        result = gsm8k.Gsm8kEval().run(model, _run_config(), "tiny")

    assert "degenerate_silence_count" not in result.metadata
    assert result.transcript[0]["degenerate_silence"] is False


# ── format_results ────────────────────────────────────────────────────────────


def test_format_results_layout():
    result = EvalResult(
        eval_name="knowledge.gsm8k",
        scores={"gsm8k_acc": 0.614},
        metadata={
            "n": 100,
            "judge_model": "gemini-3.5-flash",
            "tts_backend": {"name": "gemini_tts", "type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore"},
        },
    )
    lines = gsm8k.Gsm8kEval().format_results(result)
    joined = "\n".join(lines)
    assert "61.4%" in joined
    assert "(n=100)" in joined
    assert "gemini-3.5-flash" in joined
    assert "gemini_tts" in joined
    assert "voice: Kore" in joined

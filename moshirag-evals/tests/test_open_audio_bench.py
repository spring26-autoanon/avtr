"""
Tests for evals/registry/knowledge/open_audio_bench.py — dataset row parsing,
judge-verdict parsing, and the eval's run()/resumability/format_results
behavior. Mocks HF downloads and the Gemini judge call — no network, no GPU.
"""

from unittest.mock import patch

import pytest

from evals.registry.knowledge import open_audio_bench as oab
from evals.runner import DEFAULT_JUDGE_MODEL, EvalResult


# ── _valid_answers ───────────────────────────────────────────────────────────


def test_valid_answers_triviaqa_includes_aliases():
    row = {
        "question": "q",
        "answer_normalized_value": "paris",
        "answer_normalized_aliases": "city of paris, paris france",
    }
    answers = oab._valid_answers(oab._SUBSETS["triviaqa"], row)
    assert answers == ["paris", "city of paris", "paris france"]


def test_valid_answers_triviaqa_empty_aliases():
    row = {"question": "q", "answer_normalized_value": "paris", "answer_normalized_aliases": ""}
    assert oab._valid_answers(oab._SUBSETS["triviaqa"], row) == ["paris"]


def test_valid_answers_webq_splits_on_semicolon():
    row = {"question": "q", "answers": "George Seifert;Dom Capers;John Fox"}
    answers = oab._valid_answers(oab._SUBSETS["webq"], row)
    assert answers == ["George Seifert", "Dom Capers", "John Fox"]


def test_valid_answers_webq_single_answer():
    row = {"question": "q", "answers": "Paris"}
    assert oab._valid_answers(oab._SUBSETS["webq"], row) == ["Paris"]


def test_valid_answers_llamaq_single_only():
    row = {"Questions": "q", "Answer": "Denali"}
    assert oab._valid_answers(oab._SUBSETS["llamaq"], row) == ["Denali"]


# ── _judge ────────────────────────────────────────────────────────────────────


def test_judge_parses_correct_verdict():
    with patch.object(oab, "call_gemini", return_value="reasoning... the score is [Correct]"):
        assert oab._judge("q", ["paris"], "Paris", DEFAULT_JUDGE_MODEL) is True


def test_judge_parses_incorrect_verdict():
    with patch.object(oab, "call_gemini", return_value="reasoning... the score is [Incorrect]"):
        assert oab._judge("q", ["paris"], "London", DEFAULT_JUDGE_MODEL) is False


def test_judge_raises_on_unparseable_response():
    with patch.object(oab, "call_gemini", return_value="no verdict here"):
        with pytest.raises(ValueError, match="no parseable verdict"):
            oab._judge("q", ["paris"], "London", DEFAULT_JUDGE_MODEL)


def test_judge_passes_judge_model_through_to_call_gemini():
    with patch.object(oab, "call_gemini", return_value="the score is [Correct]") as mock_call:
        oab._judge("q", ["paris"], "Paris", "custom-judge-model")
    assert mock_call.call_args[0][0] == "custom-judge-model"


# ── OpenAudioBenchEval.run() ──────────────────────────────────────────────────


class _FakeModel:
    def __init__(self, text_out="some answer"):
        self.text_out = text_out
        self.calls = 0

    def respond(self, audio_in: bytes):
        self.calls += 1
        return b"", self.text_out, {"retrieval_context": "ctx", "retrieval_text": "ref"}


def _fake_rows(subset_cfg, limit):
    n = limit if limit is not None else 3
    if subset_cfg is oab._SUBSETS["triviaqa"]:
        return [
            {"question": f"q{i}", "answer_normalized_value": f"a{i}", "answer_normalized_aliases": "", "audio_filename": f"t{i}.wav"}
            for i in range(n)
        ]
    if subset_cfg is oab._SUBSETS["webq"]:
        return [{"question": f"q{i}", "answers": f"a{i}", "audio_filename": f"w{i}.wav"} for i in range(n)]
    return [{"Questions": f"q{i}", "Answer": f"a{i}", "audio_filename": f"l{i}.wav"} for i in range(n)]


def test_run_smoke_mode_scores_all_subsets():
    model = _FakeModel()
    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", return_value=b"wav-bytes"), \
         patch.object(oab, "call_gemini", return_value="the score is [Correct]"):
        result = oab.OpenAudioBenchEval().run(model, {}, "smoke")

    assert result.scores["triviaqa_acc"] == 1.0
    assert result.scores["webq_acc"] == 1.0
    assert result.scores["llamaq_acc"] == 1.0
    assert result.metadata["n_triviaqa"] == 5
    assert result.metadata["judge_model"] == DEFAULT_JUDGE_MODEL
    assert result.last_completed_index == 15  # 5 per subset x 3 subsets
    assert result.completed is True
    assert result.errors == []
    assert model.calls == 15

    assert len(result.transcript) == 15
    entry = result.transcript[0]
    assert entry["subset"] == "triviaqa"
    assert entry["index"] == 0
    assert entry["question"] == "q0"
    assert entry["model_response"] == "some answer"
    assert entry["retrieval_context"] == "ctx"
    assert entry["retrieval_text"] == "ref"
    assert entry["judge_verdict"] == "correct"


def test_run_uses_judge_model_from_config():
    model = _FakeModel()
    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", return_value=b"wav-bytes"), \
         patch.object(oab, "call_gemini", return_value="the score is [Correct]") as mock_call:
        result = oab.OpenAudioBenchEval().run(model, {"judge": {"model": "custom-judge"}}, "tiny")

    assert result.metadata["judge_model"] == "custom-judge"
    assert mock_call.call_args[0][0] == "custom-judge"


def test_run_tiny_mode_limits_to_one_per_subset():
    model = _FakeModel()
    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", return_value=b"wav-bytes"), \
         patch.object(oab, "call_gemini", return_value="the score is [Correct]"):
        result = oab.OpenAudioBenchEval().run(model, {}, "tiny")

    assert result.metadata["n_triviaqa"] == 1
    assert result.metadata["n_webq"] == 1
    assert result.metadata["n_llamaq"] == 1
    assert result.last_completed_index == 3  # 1 per subset x 3 subsets
    assert model.calls == 3


def test_run_records_errors_without_aborting():
    model = _FakeModel()

    def _boom(subset_cfg, row):
        raise RuntimeError("download failed")

    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", side_effect=_boom):
        result = oab.OpenAudioBenchEval().run(model, {}, "smoke")

    assert result.scores["triviaqa_acc"] == 0.0
    assert len(result.errors) == 15
    assert len(result.transcript) == 15
    assert result.transcript[0]["error"] == "download failed"
    assert "judge_verdict" not in result.transcript[0]
    assert result.completed is True  # per-question failures don't abort the eval


def test_run_resumes_without_recalling_model_for_done_items():
    model = _FakeModel()
    prior = EvalResult(
        eval_name="knowledge.open_audio_bench",
        scores={"triviaqa_acc": 1.0, "webq_acc": 1.0, "llamaq_acc": 1.0},
        metadata={
            "judge_model": DEFAULT_JUDGE_MODEL,
            "n_triviaqa": 5, "n_webq": 5, "n_llamaq": 5,
            "_progress": {
                "triviaqa": {"correct": 5, "total": 5},
                "webq": {"correct": 3, "total": 3},  # partial: only 3/5 scored
                "llamaq": {"correct": 0, "total": 0},
            },
        },
        completed=False,
        last_completed_index=13,
        transcript=[{"subset": "triviaqa", "index": i, "question": f"prior q{i}"} for i in range(5)]
        + [{"subset": "webq", "index": i, "question": f"prior q{i}"} for i in range(3)],
    )

    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", return_value=b"wav-bytes"), \
         patch.object(oab, "call_gemini", return_value="the score is [Correct]"):
        result = oab.OpenAudioBenchEval().run(model, {"_prior_result": prior}, "smoke")

    # triviaqa fully done already (5/5) -> 0 model calls
    # webq had 3/5 done -> 2 more calls
    # llamaq had 0/5 done -> 5 calls
    assert model.calls == 7
    assert result.metadata["n_webq"] == 5
    assert result.scores["webq_acc"] == 1.0  # 3 prior correct + 2 new correct, all correct
    assert result.last_completed_index == 15

    # 8 prior transcript entries preserved + 7 newly-scored ones appended
    assert len(result.transcript) == 15
    assert result.transcript[0]["question"] == "prior q0"  # untouched prior entry
    new_webq_entries = [e for e in result.transcript if e.get("subset") == "webq" and "model_response" in e]
    assert len(new_webq_entries) == 2


# ── degenerate_silence flagging ──────────────────────────────────────────────
# See MoshiRAGAdapter.respond()'s docstring — an empty response
# with no <ret> is a distinct failure mode the eval must surface, not fold
# silently into judge_verdict=incorrect.


class _SometimesSilentModel:
    """Flags every 3rd call (0-indexed) as degenerate_silence, others normal."""

    def __init__(self):
        self.calls = 0

    def respond(self, audio_in: bytes):
        i = self.calls
        self.calls += 1
        degenerate = i % 3 == 0
        text_out = "" if degenerate else "some answer"
        return b"", text_out, {
            "retrieval_context": "ctx", "retrieval_text": "ref",
            "degenerate_silence": degenerate,
        }


def test_run_flags_degenerate_silence_in_transcript_and_metadata():
    model = _SometimesSilentModel()
    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", return_value=b"wav-bytes"), \
         patch.object(oab, "call_gemini", return_value="the score is [Correct]"):
        result = oab.OpenAudioBenchEval().run(model, {}, "tiny")

    # tiny mode: 1 call per subset (indices 0, 1, 2) -> only index 0 is flagged
    flagged = [e for e in result.transcript if e["degenerate_silence"]]
    assert len(flagged) == 1
    assert flagged[0]["subset"] == "triviaqa"
    assert result.metadata["degenerate_silence_count"] == 1


def test_run_no_degenerate_silence_key_when_none_occur():
    model = _FakeModel()
    with patch.object(oab, "_load_rows", side_effect=_fake_rows), \
         patch.object(oab, "_load_audio_bytes", return_value=b"wav-bytes"), \
         patch.object(oab, "call_gemini", return_value="the score is [Correct]"):
        result = oab.OpenAudioBenchEval().run(model, {}, "tiny")

    assert "degenerate_silence_count" not in result.metadata
    assert all(e["degenerate_silence"] is False for e in result.transcript)


# ── format_results ────────────────────────────────────────────────────────────


def test_format_results_layout():
    result = EvalResult(
        eval_name="knowledge.open_audio_bench",
        scores={"triviaqa_acc": 0.732, "webq_acc": 0.747, "llamaq_acc": 0.803},
        metadata={"n_triviaqa": 100, "n_webq": 100, "n_llamaq": 100, "judge_model": "gemini-3.5-flash"},
    )
    lines = oab.OpenAudioBenchEval().format_results(result)
    joined = "\n".join(lines)
    assert "TriviaQA" in joined and "73.2%" in joined
    assert "WebQ" in joined and "74.7%" in joined
    assert "LlamaQ" in joined and "80.3%" in joined
    assert "gemini-3.5-flash" in joined

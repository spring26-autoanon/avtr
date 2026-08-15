"""Tests for scripts/summarize_demo_session.py."""

import json
import sys

import pytest

from scripts.summarize_demo_session import (
    _fmt_ret_flag,
    _fmt_ret_total,
    _mean_p95,
    compute_retrievals_from_raw_events,
    load_raw_events,
    load_records,
    main,
    merge_e2ekd,
    print_console_report,
    print_json_report,
    print_raw_stream_report,
)

SESSION_START = {
    "type": "session_start",
    "session_id": "2026-01-01T00-00-00Z",
    "checkpoint": "base",
    "git_hash": "abc1234",
    "timestamp": "2026-01-01T00:00:00Z",
    "retrieval_backend": {"name": "gemini_api", "type": "gemini_api", "model": "gemini-3.5-flash"},
    "rag_timeout_s": 8,
    "stt_mode": "local",
}

# Raw `turn` records — turns.jsonl carries no retrieval fields at all;
# those are always computed separately from raw_events.jsonl by
# compute_retrievals_from_raw_events().
TURN_1 = {
    "type": "turn",
    "turn_index": 1,
    "timestamp": "2026-01-01T00:00:45Z",
    "user_question_text": "What's the capital of France?",
    "model_response_text": "It's Paris.",
    "ttfat_s": 0.05,
    "e2ekd_s": None,
    "keyword_delay_s": None,
}

TURN_2 = {
    "type": "turn",
    "turn_index": 2,
    "timestamp": "2026-01-01T00:01:20Z",
    "user_question_text": "What did the Q3 report say about growth?",
    "model_response_text": "Revenue grew twelve percent year over year, driven mainly by the new EMEA rollout.",
    "ttfat_s": 1.42,
    "e2ekd_s": None,
    "keyword_delay_s": None,
}

# Turn fixtures already carrying a computed `retrievals` list — the shape
# print_console_report/print_json_report actually consume (i.e. as if
# compute_retrievals_from_raw_events already ran). Used by tests of the
# print/render layer, which is deliberately decoupled from raw-event
# derivation.
TURN_NO_RET = {**TURN_1, "retrievals": [], "rag_triggered": False, "rag_trigger_count": 0}
_BREAKDOWN = {"asr_wait_s": 0.12, "api_call_s": 0.71, "context_injection_s": 0.02, "total_s": 0.85}
TURN_WITH_RET = {
    **TURN_2,
    "retrievals": [{"context": "ctx", "reference_text": "ref", "breakdown": _BREAKDOWN}],
    "rag_triggered": True,
    "rag_trigger_count": 1,
}


def _write_jsonl(path, records):
    with open(path / "turns.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_raw_jsonl(path, events):
    with open(path / "raw_events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _ret_triggered(t_rel_s, turn_index):
    return {"type": "raw_event", "event": "ret_triggered", "t_rel_s": t_rel_s, "turn_index": turn_index}


def _retrieval_complete(t_rel_s, turn_index, *, context="ctx", reference_text="ref", total_s=1.0):
    return {
        "type": "raw_event",
        "event": "retrieval_complete",
        "t_rel_s": t_rel_s,
        "turn_index": turn_index,
        "retrieval_context": context,
        "retrieved_reference_text": reference_text,
        "retrieval_breakdown_s": {"asr_wait_s": 0.1, "api_call_s": total_s - 0.1, "context_injection_s": 0.0, "total_s": total_s},
    }


def _utterance_end(t_rel_s, turn_index):
    return {"type": "raw_event", "event": "utterance_end", "t_rel_s": t_rel_s, "turn_index": turn_index}


# ── load_records ────────────────────────────────────────────────────────────


def test_load_records_parses_all_record_types(tmp_path):
    e2ekd_update = {"type": "turn_e2ekd_update", "turn_index": 2, "timestamp": "x", "keyword": "growth", "keyword_delay_s": 1.6, "e2ekd_s": 3.0}
    _write_jsonl(tmp_path, [SESSION_START, TURN_1, TURN_2, e2ekd_update])

    session_start, turns, e2ekd_updates = load_records(tmp_path)

    assert session_start["session_id"] == "2026-01-01T00-00-00Z"
    assert [t["turn_index"] for t in turns] == [1, 2]
    assert e2ekd_updates == {2: e2ekd_update}


def test_load_records_ignores_blank_lines(tmp_path):
    with open(tmp_path / "turns.jsonl", "w") as f:
        f.write(json.dumps(SESSION_START) + "\n\n\n")

    session_start, turns, e2ekd_updates = load_records(tmp_path)
    assert session_start is not None
    assert turns == []
    assert e2ekd_updates == {}


def test_load_records_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_records(tmp_path)


def test_load_records_missing_session_start_exits(tmp_path):
    _write_jsonl(tmp_path, [TURN_1])
    with pytest.raises(SystemExit):
        load_records(tmp_path)


# ── load_raw_events ──────────────────────────────────────────────────────────


def test_load_raw_events_missing_file_returns_empty_list(tmp_path):
    assert load_raw_events(tmp_path) == []


def test_load_raw_events_parses_jsonl(tmp_path):
    events = [
        {"type": "raw_event", "event": "user_text", "t_rel_s": 0.1, "turn_index": 1, "text": "hi"},
        {"type": "raw_event", "event": "model_text", "t_rel_s": 0.2, "turn_index": 1, "text": "hello"},
    ]
    with open(tmp_path / "raw_events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    loaded = load_raw_events(tmp_path)
    assert loaded == events


# ── merge_e2ekd ──────────────────────────────────────────────────────────────


def test_merge_e2ekd_joins_matching_turn_index():
    update = {"turn_index": 2, "e2ekd_s": 3.0, "keyword_delay_s": 1.6}
    merged = merge_e2ekd([TURN_1, TURN_2], {2: update})

    assert merged[0]["e2ekd_s"] is None  # turn 1 had no update
    assert merged[1]["e2ekd_s"] == 3.0
    assert merged[1]["keyword_delay_s"] == 1.6


def test_merge_e2ekd_leaves_originals_untouched_without_updates():
    merged = merge_e2ekd([TURN_1], {})
    assert merged[0]["e2ekd_s"] is None
    assert merged[0] is not TURN_1  # merge_e2ekd copies, doesn't mutate in place


# ── compute_retrievals_from_raw_events ──────────────────────────────────────


def test_compute_retrievals_no_raw_events_yields_empty_retrievals():
    computed = compute_retrievals_from_raw_events([TURN_1, TURN_2], [])

    assert computed[0]["retrievals"] == []
    assert computed[0]["rag_triggered"] is False
    assert computed[0]["rag_trigger_count"] == 0
    assert computed[1]["retrievals"] == []


def test_compute_retrievals_stays_put_when_it_is_the_last_turn_of_the_session():
    """No utterance_end for turn 2 anywhere in the stream — the channel
    closed before another user utterance, so there's nothing to reassign
    this retrieval forward into; it stays tagged where it fired."""
    raw_events = [_ret_triggered(10.0, 1), _retrieval_complete(12.0, 1, total_s=2.0)]

    computed = compute_retrievals_from_raw_events([TURN_1], raw_events)

    assert len(computed[0]["retrievals"]) == 1
    assert computed[0]["retrievals"][0]["breakdown"]["total_s"] == 2.0
    assert computed[0]["rag_triggered"] is True


def test_compute_retrievals_reassigns_to_the_turn_whose_boundary_follows():
    """The core reattribution rule: a <ret> trigger tagged turn 1, whose
    retrieval completes after turn 2's boundary already happened, actually
    belongs to turn 2 — the model reacted to turn 2's question before our
    own VAD-based boundary caught up (see instrumented_server.py's module
    docstring)."""
    raw_events = [
        _ret_triggered(10.0, 1),
        _utterance_end(11.0, 2),  # turn 2 starts before the retrieval below completes
        _retrieval_complete(12.0, 1, total_s=2.0),  # still tagged 1 (live, best-effort) at write time
    ]

    computed = compute_retrievals_from_raw_events([TURN_1, TURN_2], raw_events)

    assert computed[0]["retrievals"] == []  # turn 1 keeps none
    assert len(computed[1]["retrievals"]) == 1  # turn 2 gets it
    assert computed[1]["retrievals"][0]["breakdown"]["total_s"] == 2.0


def test_compute_retrievals_same_turn_double_trigger_only_reassigns_the_last_one():
    """Regression test for a real VM session: a turn triggered <ret> twice
    — once as a genuine same-turn re-trigger (the model correcting itself
    mid-monologue, no new user input) and once again right before the next
    turn's boundary. Only the second is eligible to move forward; the
    first must stay exactly where it's tagged, or it gets misattributed to
    a turn it has nothing to do with."""
    raw_events = [
        _ret_triggered(5.0, 1),
        _retrieval_complete(6.0, 1, total_s=1.0, context="same-turn correction"),
        _ret_triggered(50.0, 1),  # much later, still turn 1 — no boundary in between
        _utterance_end(52.0, 2),
        _retrieval_complete(53.0, 1, total_s=2.0, context="grounds turn 2"),
    ]

    computed = compute_retrievals_from_raw_events([TURN_1, TURN_2], raw_events)

    assert len(computed[0]["retrievals"]) == 1
    assert computed[0]["retrievals"][0]["context"] == "same-turn correction"
    assert len(computed[1]["retrievals"]) == 1
    assert computed[1]["retrievals"][0]["context"] == "grounds turn 2"


def test_compute_retrievals_three_turn_chain_matches_real_session_shape():
    """End-to-end shape check mirroring the real VM session that motivated
    this design: three consecutive turns, each triggering right before the
    next turn's boundary — every one should land on the turn it actually
    grounds, not the turn it happened to be live-tagged with."""
    raw_events = [
        _ret_triggered(1.0, 1),
        _utterance_end(2.0, 2),
        _retrieval_complete(3.0, 1, total_s=1.1, context="grounds turn 2"),
        _ret_triggered(4.0, 2),
        _utterance_end(5.0, 3),
        _retrieval_complete(6.0, 2, total_s=1.2, context="grounds turn 3"),
        _ret_triggered(7.0, 3),
        _utterance_end(8.0, 4),
        _retrieval_complete(9.0, 3, total_s=1.3, context="grounds turn 4"),
    ]
    turn3 = {**TURN_1, "turn_index": 3}
    turn4 = {**TURN_1, "turn_index": 4}
    turns = [TURN_1, TURN_2, turn3, turn4]

    computed = compute_retrievals_from_raw_events(turns, raw_events)

    by_index = {t["turn_index"]: t for t in computed}
    assert by_index[1]["retrievals"] == []
    assert by_index[2]["retrievals"][0]["context"] == "grounds turn 2"
    assert by_index[3]["retrievals"][0]["context"] == "grounds turn 3"
    assert by_index[4]["retrievals"][0]["context"] == "grounds turn 4"


def test_compute_retrievals_more_triggers_than_completions_pairs_up_to_the_shorter_list():
    """Documented simplification: if a trigger were ever cancelled before
    completing (RAGManager.trigger() cancels any prior in-flight retrieval
    — no observed case of this in any real session so far), it has no
    completion to pair with and is dropped rather than causing a crash or
    a misaligned pairing."""
    raw_events = [_ret_triggered(1.0, 1), _ret_triggered(2.0, 1), _retrieval_complete(3.0, 1, total_s=1.0)]

    computed = compute_retrievals_from_raw_events([TURN_1], raw_events)

    assert len(computed[0]["retrievals"]) == 1


def test_compute_retrievals_does_not_mutate_input_turns():
    computed = compute_retrievals_from_raw_events([TURN_1], [])
    assert computed[0] is not TURN_1
    assert "retrievals" not in TURN_1


# ── _fmt_ret_flag / _fmt_ret_total ──────────────────────────────────────────


def test_fmt_ret_flag_variants():
    assert _fmt_ret_flag([]) == "no"
    assert _fmt_ret_flag([{}]) == "yes"
    assert _fmt_ret_flag([{}, {}]) == "yes ×2"


def test_fmt_ret_total_variants():
    assert _fmt_ret_total([]) == "—"
    assert _fmt_ret_total([{"breakdown": {"total_s": 2.5}}]) == "2.50s"
    # Multiple retrievals list every value, comma-separated — not summed or
    # averaged, so the behavior stays visible rather than blended away.
    assert _fmt_ret_total([{"breakdown": {"total_s": 2.5}}, {"breakdown": {"total_s": 1.1}}]) == "2.50s, 1.10s"


# ── _mean_p95 ────────────────────────────────────────────────────────────────


def test_mean_p95_empty_returns_none():
    assert _mean_p95([]) is None


def test_mean_p95_single_value():
    mean, p95 = _mean_p95([2.0])
    assert mean == 2.0
    assert p95 == 2.0


def test_mean_p95_computes_mean_correctly():
    mean, _ = _mean_p95([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)


# ── console / json report ───────────────────────────────────────────────────


def test_print_console_report_smoke(capsys):
    print_console_report(SESSION_START, [TURN_NO_RET, TURN_WITH_RET])
    out = capsys.readouterr().out

    assert "MoshiRAG Demo Session" in out
    assert "session id : 2026-01-01T00-00-00Z" in out
    assert "gemini-3.5-flash" in out
    assert "What's the capital of France?" in out
    assert "<ret> trigger rate : 50.0%   (1/2)" in out
    assert "e2ekd not yet implemented" in out


def test_print_console_report_ret_column_shows_multiplicity(capsys):
    breakdown1 = {"asr_wait_s": 0.1, "api_call_s": 1.3, "context_injection_s": 0.0, "total_s": 1.5}
    breakdown2 = {"asr_wait_s": 0.1, "api_call_s": 2.3, "context_injection_s": 0.0, "total_s": 2.5}
    two_retrievals = {
        **TURN_WITH_RET,
        "retrievals": [
            {"context": "c1", "reference_text": "r1", "breakdown": breakdown1},
            {"context": "c2", "reference_text": "r2", "breakdown": breakdown2},
        ],
        "rag_trigger_count": 2,
    }
    print_console_report(SESSION_START, [two_retrievals])
    out = capsys.readouterr().out

    assert "yes ×2" in out
    assert "1.50s, 2.50s" in out


def test_print_console_report_includes_full_untruncated_transcript(capsys):
    print_console_report(SESSION_START, [TURN_NO_RET, TURN_WITH_RET])
    out = capsys.readouterr().out

    assert "[full transcript]" in out
    assert "#2 user : What did the Q3 report say about growth?" in out
    assert "#2 moshi: Revenue grew twelve percent year over year, driven mainly by the new EMEA rollout." in out
    assert "#1 user : What's the capital of France?" in out
    assert "#1 moshi: It's Paris." in out


def test_print_console_report_transcript_includes_retrieval_context_and_reference(capsys):
    print_console_report(SESSION_START, [TURN_WITH_RET])
    out = capsys.readouterr().out

    assert "retrieval (what the model asked with):" in out
    assert "retrieval (what came back):" in out
    assert "ctx" in out
    assert "ref" in out


def test_print_console_report_transcript_numbers_multiple_retrievals(capsys):
    breakdown1 = {"asr_wait_s": 0.1, "api_call_s": 1.3, "context_injection_s": 0.0, "total_s": 1.5}
    breakdown2 = {"asr_wait_s": 0.1, "api_call_s": 2.3, "context_injection_s": 0.0, "total_s": 2.5}
    two_retrievals = {
        **TURN_WITH_RET,
        "retrievals": [
            {"context": "first context", "reference_text": "first ref", "breakdown": breakdown1},
            {"context": "second context", "reference_text": "second ref", "breakdown": breakdown2},
        ],
        "rag_trigger_count": 2,
    }
    print_console_report(SESSION_START, [two_retrievals])
    out = capsys.readouterr().out

    assert "retrieval 1/2 (what the model asked with):" in out
    assert "retrieval 2/2 (what the model asked with):" in out
    assert "first context" in out
    assert "second context" in out


def test_print_console_report_transcript_omits_retrieval_block_when_no_retrieval(capsys):
    print_console_report(SESSION_START, [TURN_NO_RET])
    out = capsys.readouterr().out

    assert "retrieval (what" not in out


def test_print_console_report_retrieval_detail_indented_deeper_than_dialogue_lines(capsys):
    """Regression test for the user's explicit ask: retrieval detail must be
    visually subordinate to the #N user/moshi lines, not compete with them,
    even after a long line wraps (hanging-indent, not terminal wrapping)."""
    long_context = "word " * 40  # long enough to force textwrap to wrap
    turn = {**TURN_WITH_RET, "retrievals": [{"context": long_context, "reference_text": None, "breakdown": None}]}
    print_console_report(SESSION_START, [turn])
    out = capsys.readouterr().out

    lines = out.splitlines()
    dialogue_line = next(line for line in lines if line.startswith("  #2 user"))
    context_lines = [line for line in lines if "word" in line]
    dialogue_indent = len(dialogue_line) - len(dialogue_line.lstrip(" "))
    for line in context_lines:
        indent = len(line) - len(line.lstrip(" "))
        assert indent > dialogue_indent
    # And the wrapped continuation must indent deeper than the first line —
    # this is the actual bug being regression-tested (terminal wrapping has
    # no concept of "continuation", so it used to land flush-left).
    assert len(context_lines) > 1
    first_indent = len(context_lines[0]) - len(context_lines[0].lstrip(" "))
    cont_indent = len(context_lines[1]) - len(context_lines[1].lstrip(" "))
    assert cont_indent > first_indent


def test_print_console_report_handles_zero_turns(capsys):
    print_console_report(SESSION_START, [])
    out = capsys.readouterr().out

    assert "turns              : 0" in out
    assert "<ret> trigger rate : 0.0%   (0/0)" in out
    assert "n/a" in out  # no retrieval breakdown / no ttfat data


def test_print_console_report_aligns_ttfat_mean_with_retrieval_breakdown_mean(capsys):
    """Regression test: ttfat's "mean"/"p95" columns must line up with the
    retrieval breakdown rows directly above — they were previously 11
    columns off."""
    print_console_report(SESSION_START, [TURN_WITH_RET])
    out = capsys.readouterr().out

    breakdown_line = next(line for line in out.splitlines() if "asr_wait" in line)
    ttfat_line = next(line for line in out.splitlines() if line.lstrip().startswith("ttfat"))
    assert breakdown_line.index("mean") == ttfat_line.index("mean")


def test_print_console_report_aggregate_flattens_multiple_retrievals_per_turn(capsys):
    """n in the aggregate retrieval breakdown must count individual
    retrievals, not turns — a turn with 2 retrievals contributes 2, not 1
    (and not 0, which is what the pre-raw-stream-derived design could
    silently do when a turn's second retrieval had nowhere to land)."""
    two_retrievals = {
        **TURN_WITH_RET,
        "retrievals": [
            {"context": "c1", "reference_text": "r1", "breakdown": {"asr_wait_s": 0.1, "api_call_s": 0.9, "context_injection_s": 0.0, "total_s": 1.0}},
            {"context": "c2", "reference_text": "r2", "breakdown": {"asr_wait_s": 0.1, "api_call_s": 1.9, "context_injection_s": 0.0, "total_s": 2.0}},
        ],
    }
    print_console_report(SESSION_START, [two_retrievals])
    out = capsys.readouterr().out

    assert "(n=2)" in out


def test_print_console_report_includes_raw_stream_section(capsys):
    print_console_report(SESSION_START, [TURN_NO_RET], raw_events=None)
    out = capsys.readouterr().out

    assert "[raw stream]" in out
    assert "n/a" in out  # no raw_events.jsonl passed


def test_print_console_report_renders_passed_raw_events(capsys):
    raw_events = [{"event": "user_text", "t_rel_s": 1.234, "turn_index": 1, "text": "hi there"}]
    print_console_report(SESSION_START, [TURN_NO_RET], raw_events=raw_events)
    out = capsys.readouterr().out

    assert "USER" in out
    assert "hi there" in out


# ── print_raw_stream_report ─────────────────────────────────────────────────


def test_print_raw_stream_report_empty_shows_na(capsys):
    print_raw_stream_report([])
    out = capsys.readouterr().out

    assert "[raw stream]" in out
    assert "n/a" in out


def test_print_raw_stream_report_formats_each_event_kind(capsys):
    raw_events = [
        {"event": "user_text", "t_rel_s": 0.1, "turn_index": 1, "text": "hi"},
        {"event": "utterance_end", "t_rel_s": 0.2, "turn_index": 1},
        {"event": "model_text", "t_rel_s": 0.3, "turn_index": 1, "text": "hello"},
        {"event": "ret_triggered", "t_rel_s": 0.4, "turn_index": 1},
        {
            "event": "retrieval_complete",
            "t_rel_s": 3.5,
            "turn_index": 1,
            "retrieval_context": "ctx",
            "retrieved_reference_text": "ref",
            "retrieval_breakdown_s": {"total_s": 3.1},
        },
        {"event": "first_audio", "t_rel_s": 3.6, "turn_index": 1, "ttfat_s": 0.05},
    ]
    print_raw_stream_report(raw_events)
    out = capsys.readouterr().out

    assert "USER" in out and "hi" in out
    assert "turn boundary" in out
    assert "MOSHI" in out and "hello" in out
    assert "<ret> triggered" in out
    assert "retrieval complete (3.10s)" in out
    assert "first audio (ttfat 0.05s)" in out


def test_print_raw_stream_report_preserves_chronological_order(capsys):
    raw_events = [
        {"event": "user_text", "t_rel_s": 0.1, "turn_index": 1, "text": "first"},
        {"event": "model_text", "t_rel_s": 0.2, "turn_index": 1, "text": "second"},
    ]
    print_raw_stream_report(raw_events)
    out = capsys.readouterr().out

    assert out.index("first") < out.index("second")


def test_print_json_report_emits_valid_pretty_json(tmp_path, capsys):
    print_json_report(SESSION_START, [TURN_NO_RET, TURN_WITH_RET], tmp_path)
    out = capsys.readouterr().out

    doc = json.loads(out)
    assert doc["session_id"] == "2026-01-01T00-00-00Z"
    assert len(doc["turns"]) == 2
    assert "\n  " in out  # indent=2 pretty-printed, not compact


# ── main() end-to-end ────────────────────────────────────────────────────────


def test_main_console_end_to_end(tmp_path, capsys, monkeypatch):
    _write_jsonl(tmp_path, [SESSION_START, TURN_1, TURN_2])
    monkeypatch.setattr(sys, "argv", ["summarize_demo_session.py", str(tmp_path)])

    main()

    out = capsys.readouterr().out
    assert "MoshiRAG Demo Session" in out
    assert f"raw logs: {tmp_path}/{{server,conditioner}}.log" in out


def test_main_json_end_to_end(tmp_path, capsys, monkeypatch):
    _write_jsonl(tmp_path, [SESSION_START, TURN_1])
    monkeypatch.setattr(sys, "argv", ["summarize_demo_session.py", str(tmp_path), "--json"])

    main()

    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["checkpoint"] == "base"
    assert "raw logs" in captured.err


def test_main_console_end_to_end_reattributes_a_raced_retrieval(tmp_path, capsys, monkeypatch):
    """Full pipeline, matching the real VM session that motivated this
    design: turn 1's <ret> trigger fires while still tagged turn 1, but
    the retrieval only completes after turn 2's boundary already started
    — main() must still show it under turn 2 in the rendered report, not
    blank and not misattributed to turn 1."""
    _write_jsonl(tmp_path, [SESSION_START, TURN_1, TURN_2])
    _write_raw_jsonl(
        tmp_path,
        [
            _ret_triggered(10.0, 1),
            _utterance_end(11.0, 2),
            _retrieval_complete(12.0, 1, total_s=3.70, context="late-arriving context", reference_text="late-arriving reference"),
        ],
    )
    monkeypatch.setattr(sys, "argv", ["summarize_demo_session.py", str(tmp_path)])

    main()

    out = capsys.readouterr().out
    turns_section = out.split("[turns]")[1].split("[aggregate]")[0]
    lines = [line for line in turns_section.splitlines() if line.strip()]
    turn1_line = next(line for line in lines if line.strip().startswith("1 "))
    turn2_line = next(line for line in lines if line.strip().startswith("2 "))
    assert "3.70s" not in turn1_line  # stays where tagged only if there's no next turn — here there is
    assert "yes" in turn2_line
    assert "3.70s" in turn2_line
    assert "late-arriving context" in out
    assert "late-arriving reference" in out


# ── pad_stall_steps column (docs/demo-turn-onset-regression.md) ──────────────


def test_console_report_shows_pad_stall_steps(capsys):
    turn = {**TURN_1, "ttfat_s": 4.24, "pad_stall_steps": 53}

    print_console_report(SESSION_START, [turn], [])

    out = capsys.readouterr().out
    assert "pad" in out
    assert "53" in out
    assert "pad stall (steps)" in out
    # x80ms conversion is what makes the step count readable as latency.
    assert "4.2s mean" in out


def test_console_report_marks_sessions_predating_pad_stall_steps(capsys):
    """Pre-2026-07-30 sessions have no pad_stall_steps, which is also the
    signal that their ttfat_s used the old turn-switch anchor and is not
    comparable with later sessions."""
    print_console_report(SESSION_START, [{**TURN_1, "ttfat_s": 0.16}], [])

    out = capsys.readouterr().out
    assert "predates pad_stall_steps" in out

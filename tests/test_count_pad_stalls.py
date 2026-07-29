"""Tests for scripts/count_pad_stalls.py — no moshi/torch dependency."""

from scripts.count_pad_stalls import analyze_runs, count_stalls_per_switch, resolve_log_path


# ── count_stalls_per_switch ──────────────────────────────────────────────────


def test_counts_stalls_between_switch_events():
    lines = [
        "some unrelated line\n",
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "[VAD] switching to model\n",
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "[VAD] switching to model\n",
    ]
    assert count_stalls_per_switch(lines) == [2, 1]


def test_first_switch_with_zero_prior_stalls():
    # The controlled experiment this script exists for: a fresh session's
    # very first switch-to-model event, no prior stall lines at all.
    lines = ["[VAD] switching to model\n"]
    assert count_stalls_per_switch(lines) == [0]


def test_no_switch_events_returns_empty_list():
    lines = ["[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n"] * 5
    assert count_stalls_per_switch(lines) == []


def test_interleaved_unrelated_lines_do_not_break_a_run():
    lines = [
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "WARNING batched step (1/1 active) took 95.3ms\n",
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "[Reference] ARC encoding received in 0.454s\n",
        "[VAD] switching to model\n",
    ]
    assert count_stalls_per_switch(lines) == [2]


def test_switch_match_is_case_insensitive():
    lines = ["[VAD] Switching To Model\n"]
    assert count_stalls_per_switch(lines) == [0]


def test_running_count_resets_after_each_switch():
    lines = [
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "[VAD] switching to model\n",
        "[VAD] switching to model\n",
    ]
    assert count_stalls_per_switch(lines) == [1, 0]


# ── analyze_runs ──────────────────────────────────────────────────────────────


def test_analyze_runs_flags_a_step_warning_inside_a_run():
    lines = [
        "18:10:57.26 INFO    [VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "18:10:57.34 WARNING batched step (1/16 active) took 210.5ms\n",
        "18:10:57.42 INFO    [VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "18:10:57.50 INFO    [VAD] Waiting ended, switching to model\n",
    ]
    runs = analyze_runs(lines)
    assert len(runs) == 1
    assert runs[0]["count"] == 2
    assert runs[0]["start_ts"] == "18:10:57.26"
    assert runs[0]["switch_ts"] == "18:10:57.50"
    assert runs[0]["step_warnings"] == ["18:10:57.34 WARNING batched step (1/16 active) took 210.5ms"]


def test_analyze_runs_reports_no_warnings_when_cadence_is_clean():
    lines = [
        "18:10:57.26 INFO    [VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "18:10:57.50 INFO    [VAD] Waiting ended, switching to model\n",
    ]
    runs = analyze_runs(lines)
    assert runs[0]["step_warnings"] == []


def test_analyze_runs_resets_step_warnings_between_runs():
    lines = [
        "18:10:57.26 INFO    [VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "18:10:57.34 WARNING batched step (1/16 active) took 210.5ms\n",
        "18:10:57.50 INFO    [VAD] Waiting ended, switching to model\n",
        "18:10:58.00 INFO    [VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "18:10:58.10 INFO    [VAD] Waiting ended, switching to model\n",
    ]
    runs = analyze_runs(lines)
    assert len(runs) == 2
    assert runs[0]["step_warnings"] != []
    assert runs[1]["step_warnings"] == []


def test_analyze_runs_ignores_a_step_warning_before_the_runs_first_stall_line():
    # A warning logged during the *previous* turn's own speech (before this
    # run's first stall line even happened) must not be credited to this
    # run — real bug found on a live session where a warning from well
    # before turn 1's stall window got misattributed to it.
    lines = [
        "18:38:17.70 WARNING batched step (1/16 active) took 82.2ms\n",
        "18:38:27.01 INFO    [VAD] User stopped speaking but LM buffer empty, remaining in user turn\n",
        "18:38:33.14 INFO    [VAD] Waiting ended, switching to model\n",
    ]
    runs = analyze_runs(lines)
    assert runs[0]["step_warnings"] == []


def test_analyze_runs_start_ts_none_when_run_has_no_stalls():
    lines = ["18:10:57.50 INFO    [VAD] Waiting ended, switching to model\n"]
    runs = analyze_runs(lines)
    assert runs[0]["count"] == 0
    assert runs[0]["start_ts"] is None


# ── resolve_log_path ─────────────────────────────────────────────────────────


def test_resolve_log_path_appends_server_log_for_a_directory(tmp_path):
    assert resolve_log_path(tmp_path) == tmp_path / "server.log"


def test_resolve_log_path_leaves_a_file_path_unchanged(tmp_path):
    log_file = tmp_path / "server.log"
    log_file.write_text("")
    assert resolve_log_path(log_file) == log_file


# ── main() end-to-end via a real fake log file ───────────────────────────────


def test_main_reports_zero_events_for_a_log_with_no_switches(tmp_path, capsys):
    from scripts.count_pad_stalls import main
    import sys

    import pytest

    session_dir = tmp_path / "2026-07-29T07-19-59Z"
    session_dir.mkdir()
    (session_dir / "server.log").write_text("nothing relevant here\n")

    old_argv = sys.argv
    sys.argv = ["count_pad_stalls.py", str(session_dir)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "No 'switching to model' events matched" in out


def test_main_reports_first_turn_count(tmp_path, capsys):
    from scripts.count_pad_stalls import main
    import sys

    session_dir = tmp_path / "2026-07-29T07-19-59Z"
    session_dir.mkdir()
    (session_dir / "server.log").write_text(
        "[VAD] User stopped speaking but LM buffer empty, remaining in user turn\n" * 17
        + "[VAD] switching to model\n"
        + "[VAD] switching to model\n"
    )

    old_argv = sys.argv
    sys.argv = ["count_pad_stalls.py", str(session_dir)]
    try:
        main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "17, 0" in out
    assert "First turn (no prior conversation depth): 17" in out

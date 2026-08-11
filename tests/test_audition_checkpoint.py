"""Scorecard math and log parsing for the audition harness.

The verdict weighting encodes the decision of record: stalls + voice/persona first,
trigger rate second. known_good.md is the Sunday demo script, so the ≥2-of-3 rule is
what keeps a lucky single-session fire out of the live demo.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audition_checkpoint import parse_log, score, known_good, QUESTIONS  # noqa: E402

LOG = """\
2026-08-08 10:00:00,000 INFO batched step took 81.2 ms
2026-08-08 10:00:05,000 INFO [RAG] model emitted ret token
2026-08-08 10:00:06,900 INFO Generated reference: The first World Cup...
2026-08-08 10:00:20,000 WARNING LM buffer empty
2026-08-08 10:00:20,400 WARNING LM buffer empty
2026-08-08 10:00:22,100 WARNING LM buffer empty
2026-08-08 10:00:40,000 INFO batched step took 79.8 ms
"""


def _session(n, overrides=None, voice=4):
    results = {q["id"]: {"fired": False, "passed": True, "notes": ""} for q in QUESTIONS}
    for qid in ("q04", "q05", "q06"):
        results[qid] = {"fired": True, "passed": True, "notes": ""}
    for qid, r in (overrides or {}).items():
        results[qid] = r
    return {"checkpoint": "t", "scaling": 2.0, "session": n, "voice": voice,
            "interrupt_yields": True, "results": results}


def test_parse_log_counts():
    m = parse_log(LOG)
    assert m["triggers"] == 1 and m["references"] == 1 and m["stall_lines"] == 3


def test_longest_stall_groups_adjacent_lines():
    # 10:00:20.0 → 10:00:20.4 (gap .4 merges) → 10:00:22.1 (gap 1.7 splits)
    assert parse_log(LOG)["longest_stall_sec"] == 0.4


def test_round_trip_pairs_trigger_with_next_reference():
    assert parse_log(LOG)["round_trips"] == [1.9]


def test_step_timings_extracted():
    assert parse_log(LOG)["step_ms"] == [81.2, 79.8]


def test_no_timestamps_falls_back_to_none():
    m = parse_log("LM buffer empty\nLM buffer empty\n")
    assert m["stall_lines"] == 2 and m["longest_stall_sec"] is None


def test_score_trigger_and_persona_rates():
    s = score([_session(1)], parse_log(LOG))
    assert s["trigger_rate"] == 1.0
    assert s["persona_rate"] == 1.0
    assert s["spurious_fires"] == 0
    assert s["voice_mean"] == 4.0


def test_score_counts_spurious_fires_on_nonfactual():
    bad = {"q02": {"fired": True, "passed": False, "notes": "retrieved own name"}}
    assert score([_session(1, bad)], parse_log(LOG))["spurious_fires"] == 1


def test_known_good_requires_two_of_three_sessions():
    miss = {"q06": {"fired": False, "passed": False, "notes": ""}}
    kg = known_good([_session(1), _session(2, miss), _session(3, miss)])
    assert "q05" in kg["demo"] and "q06" not in kg["demo"]
    assert "q06" in kg["avoid"]


def test_known_good_factual_needs_fire_and_pass():
    conf = {"q04": {"fired": False, "passed": True, "notes": "confabulated correctly"}}
    kg = known_good([_session(1, conf), _session(2, conf), _session(3, conf)])
    assert "q04" in kg["avoid"]


def test_persona_questions_listed_separately():
    kg = known_good([_session(1), _session(2), _session(3)])
    assert "q02" in kg["persona"] and "q02" not in kg["demo"]


def test_decline_may_fire_and_still_be_known_good():
    """Real declines are trained marked+referenced: fire-then-decline-gracefully is correct."""
    fired_decline = {"q07": {"fired": True, "passed": True, "notes": "fired, then declined"}}
    kg = known_good([_session(1, fired_decline), _session(2, fired_decline), _session(3, fired_decline)])
    assert "q07" in kg["demo"]


def test_round_trip_consumes_each_reference_once():
    log = """\
2026-08-08 10:00:05,000 INFO [RAG] model emitted ret token
2026-08-08 10:00:05,500 INFO [RAG] model emitted ret token
2026-08-08 10:00:06,900 INFO Generated reference: only one
"""
    assert parse_log(log)["round_trips"] == [1.9]


def test_backstop_lines_counted():
    log = (
        "12:00:01.00 INFO    [slot 0] [RAG] model emitted RAG token, triggering reference generation\n"
        "12:00:05.00 INFO    [slot 0] [Backstop] engaged\n"
        "12:00:09.00 INFO    [slot 0] [Backstop] engaged\n"
    )
    result = parse_log(log)
    assert result["backstops"] == 2
    assert result["backstop_rate"] == 2 / 3


def test_backstop_rate_none_when_no_events():
    result = parse_log("12:00:01.00 INFO    nothing relevant\n")
    assert result["backstops"] == 0
    assert result["backstop_rate"] is None


def test_cli_rejects_mixed_checkpoint_sessions(tmp_path):
    """A mistyped --sessions glob must fail loudly, not silently blend two checkpoints
    into one scorecard (the false 'checkpoint 500 regression' failure mode)."""
    import json as _json, subprocess, sys
    log = tmp_path / "serve.log"; log.write_text("")
    p1, p2 = tmp_path / "a_s1.json", tmp_path / "b_s1.json"
    p1.write_text(_json.dumps(_session(1)))
    mixed = _session(2); mixed["checkpoint"] = "OTHER"
    p2.write_text(_json.dumps(mixed))
    r = subprocess.run([sys.executable, str(REPO / "scripts/audition_checkpoint.py"),
                        "--log", str(log), "--sessions", str(p1), str(p2),
                        "--out-dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "mix" in (r.stderr + r.stdout)

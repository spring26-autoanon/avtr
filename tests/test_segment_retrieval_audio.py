import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from segment_retrieval_audio import (
    dedupe,
    normalize,
    snap_bounds,
    split_long,
    turn_mix,
    validate_segments,
    windows,
)


def _utts(spec):
    """spec: [(speaker, start, end)] -> utterance dicts with filler text."""
    return [{"speaker": s, "start": a, "end": b, "text": "word"} for s, a, b in spec]


def _seg(sid, start, end, turns):
    return {"id": sid, "start": start, "end": end, "turns": turns}


def _turn(speaker, start, end, kind="smalltalk", reference=None):
    return {"speaker": speaker, "start": start, "end": end, "kind": kind, "reference": reference}


def test_snap_moves_bounds_to_nearest_utterance_edges():
    utts = _utts([("JOSHUA", 10.0, 12.0), ("DANIELLE", 13.0, 20.0)])
    assert snap_bounds(10.4, 19.2, utts) == (10.0, 20.0)


def test_validate_drops_segment_shorter_than_min():
    utts = _utts([("JOSHUA", 0.0, 1.0), ("DANIELLE", 1.2, 2.0)])
    segs = [_seg("s0", 0.0, 2.0, [_turn("JOSHUA", 0.0, 1.0), _turn("DANIELLE", 1.2, 2.0)])]
    kept, problems = validate_segments(segs, utts, min_sec=4.0)
    assert kept == []
    assert any("too short" in p for p in problems)


def test_validate_drops_segment_with_no_danielle_turn():
    utts = _utts([("JOSHUA", 0.0, 9.0)])
    segs = [_seg("s0", 0.0, 9.0, [_turn("JOSHUA", 0.0, 9.0)])]
    kept, problems = validate_segments(segs, utts)
    assert kept == []
    assert any("no JOSHUA-question -> DANIELLE-answer span" in p for p in problems)


def test_validate_drops_overlapping_second_segment():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0),
                  ("JOSHUA", 8.0, 9.0), ("DANIELLE", 9.5, 20.0)])
    segs = [
        _seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0), _turn("DANIELLE", 2.5, 10.0)]),
        _seg("s1", 8.0, 20.0, [_turn("JOSHUA", 8.0, 9.0), _turn("DANIELLE", 9.5, 20.0)]),
    ]
    kept, problems = validate_segments(segs, utts)
    assert [s["id"] for s in kept] == ["s0"]
    assert any("overlap" in p for p in problems)


def test_validate_keeps_a_good_segment_unchanged():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0)])
    segs = [_seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0),
                                   _turn("DANIELLE", 2.5, 10.0, "grounded", "a passage")])]
    kept, problems = validate_segments(segs, utts)
    assert len(kept) == 1 and problems == []
    assert kept[0]["turns"][1]["reference"] == "a passage"


def test_split_long_cuts_at_joshua_turn_nearest_midpoint():
    turns = [
        _turn("JOSHUA", 0.0, 5.0), _turn("DANIELLE", 5.0, 60.0, "grounded", "ref A"),
        _turn("JOSHUA", 60.0, 65.0), _turn("DANIELLE", 65.0, 130.0, "grounded", "ref B"),
    ]
    utts = _utts([(t["speaker"], t["start"], t["end"]) for t in turns])
    halves = split_long(_seg("s0", 0.0, 130.0, turns), utts, max_sec=100.0)
    assert len(halves) == 2
    assert halves[0]["start"] == 0.0 and halves[0]["end"] == 60.0
    assert halves[1]["start"] == 60.0 and halves[1]["end"] == 130.0
    # each half keeps only the turns that fall inside it, with their own references
    assert [t["reference"] for t in halves[0]["turns"]] == [None, "ref A"]
    assert [t["reference"] for t in halves[1]["turns"]] == [None, "ref B"]


def test_validate_splits_over_long_segment_rather_than_dropping_it():
    turns = [
        _turn("JOSHUA", 0.0, 5.0), _turn("DANIELLE", 5.0, 60.0),
        _turn("JOSHUA", 60.0, 65.0), _turn("DANIELLE", 65.0, 130.0),
    ]
    utts = _utts([(t["speaker"], t["start"], t["end"]) for t in turns])
    kept, _ = validate_segments([_seg("s0", 0.0, 130.0, turns)], utts, max_sec=100.0)
    assert len(kept) == 2
    assert all(s["end"] - s["start"] <= 100.0 for s in kept)


def test_validate_reports_unclaimed_minutes_are_not_an_error():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0),
                  ("JOSHUA", 500.0, 502.0), ("DANIELLE", 502.5, 510.0)])
    segs = [_seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0), _turn("DANIELLE", 2.5, 10.0)])]
    kept, problems = validate_segments(segs, utts)
    assert len(kept) == 1
    assert problems == []


def test_validate_drops_over_long_segment_with_no_internal_joshua_turn():
    """A 130 s monologue cannot be split, so it must be reported rather than kept."""
    turns = [_turn("JOSHUA", 0.0, 5.0), _turn("DANIELLE", 5.0, 130.0)]
    utts = _utts([(t["speaker"], t["start"], t["end"]) for t in turns])
    kept, problems = validate_segments([_seg("s0", 0.0, 130.0, turns)], utts, max_sec=100.0)
    assert kept == []
    assert any("too long" in p for p in problems)


def test_validate_drops_segment_with_no_joshua_turn_at_all():
    utts = _utts([("DANIELLE", 0.0, 5.0), ("DANIELLE", 5.5, 10.0)])
    segs = [_seg("s0", 0.0, 10.0, [_turn("DANIELLE", 0.0, 5.0), _turn("DANIELLE", 5.5, 10.0)])]
    kept, problems = validate_segments(segs, utts)
    assert kept == []
    assert any("no JOSHUA-question -> DANIELLE-answer span" in p for p in problems)


def test_validate_trims_a_trailing_joshua_turn_rather_than_dropping():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 8.0), ("JOSHUA", 8.5, 10.0)])
    segs = [_seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0), _turn("DANIELLE", 2.5, 8.0),
                                   _turn("JOSHUA", 8.5, 10.0)])]
    kept, problems = validate_segments(segs, utts)
    assert len(kept) == 1, problems
    assert [t["speaker"] for t in kept[0]["turns"]] == ["JOSHUA", "DANIELLE"]
    assert kept[0]["end"] == 8.0          # edge followed the trim


def test_validate_orders_segments_by_start_time():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0),
                  ("JOSHUA", 20.0, 22.0), ("DANIELLE", 22.5, 30.0)])
    late = _seg("s1", 20.0, 30.0, [_turn("JOSHUA", 20.0, 22.0), _turn("DANIELLE", 22.5, 30.0)])
    early = _seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0), _turn("DANIELLE", 2.5, 10.0)])
    kept, _ = validate_segments([late, early], utts)
    assert [s["id"] for s in kept] == ["s0", "s1"]


def test_windows_overlap_so_boundary_dialogues_are_seen_whole():
    utts = [{"speaker": "JOSHUA", "start": float(t), "end": t + 0.5, "text": "w"}
            for t in range(0, 1500, 10)]
    ws = windows(utts, window_sec=720.0, overlap_sec=60.0)
    assert len(ws) >= 2
    # window 2 starts before window 1 ends
    assert ws[1][0]["start"] < ws[0][-1]["end"]
    # every utterance appears somewhere
    seen = {u["start"] for w in ws for u in w}
    assert seen == {u["start"] for u in utts}


def test_dedupe_drops_segments_repeated_across_window_overlap():
    segs = [
        {"id": "w0-1", "start": 700.0, "end": 750.0, "turns": []},
        {"id": "w1-0", "start": 700.3, "end": 750.2, "turns": []},
        {"id": "w1-1", "start": 800.0, "end": 850.0, "turns": []},
    ]
    out = dedupe(segs, tol=1.0)
    assert [s["start"] for s in out] == [700.0, 800.0]


def test_turn_mix_counts_only_her_turns():
    segs = [{
        "id": "s0", "start": 0.0, "end": 50.0,
        "turns": [
            {"speaker": "JOSHUA", "start": 0.0, "end": 2.0, "kind": "smalltalk", "reference": None},
            {"speaker": "DANIELLE", "start": 2.0, "end": 20.0, "kind": "grounded", "reference": "r"},
            {"speaker": "JOSHUA", "start": 20.0, "end": 22.0, "kind": "smalltalk", "reference": None},
            {"speaker": "DANIELLE", "start": 22.0, "end": 50.0, "kind": "smalltalk", "reference": None},
        ],
    }]
    mix = turn_mix(segs)
    assert mix == {"grounded": 1, "decline": 0, "persona": 0, "smalltalk": 1, "total": 2, "retrieval_share": 0.5}


def test_normalize_forces_joshua_turns_to_smalltalk_with_no_reference():
    raw = [{"start": 0.0, "end": 10.0, "turns": [
        {"speaker": "JOSHUA", "start": 0.0, "end": 2.0, "kind": "grounded", "reference": "nope"},
        {"speaker": "DANIELLE", "start": 2.0, "end": 10.0, "kind": "grounded", "reference": "yes"},
    ]}]
    out = normalize(raw, 0)
    assert out[0]["turns"][0]["kind"] == "smalltalk"
    assert out[0]["turns"][0]["reference"] is None
    assert out[0]["turns"][1]["reference"] == "yes"


def test_normalize_strips_references_from_smalltalk_and_rejects_bad_kinds():
    raw = [{"start": 0.0, "end": 10.0, "turns": [
        {"speaker": "JOSHUA", "start": 0.0, "end": 2.0, "kind": "smalltalk"},
        {"speaker": "DANIELLE", "start": 2.0, "end": 6.0, "kind": "smalltalk", "reference": "x"},
        {"speaker": "DANIELLE", "start": 6.0, "end": 10.0, "kind": "nonsense", "reference": "y"},
    ]}]
    out = normalize(raw, 0)
    assert out[0]["turns"][1]["reference"] is None
    assert out[0]["turns"][2]["kind"] == "smalltalk"
    assert out[0]["turns"][2]["reference"] is None


def test_normalize_ids_are_unique_across_windows():
    raw = [{"start": 0.0, "end": 10.0, "turns": []}]
    assert normalize(raw, 0)[0]["id"] != normalize(raw, 1)[0]["id"]


from segment_retrieval_audio import attach_real_turns, trim_to_speaker_bounds


def test_attach_real_turns_uses_utterances_not_llm_timestamps():
    """Whisper utterances are ground truth; the LLM only supplies labels."""
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0)])
    # LLM turn timings are deliberately wrong by a few tenths
    llm = [_turn("JOSHUA", 0.1, 2.2), _turn("DANIELLE", 2.4, 10.4, "grounded", "a passage")]
    turns = attach_real_turns(0.0, 10.0, llm, utts)
    assert [(t["speaker"], t["start"], t["end"]) for t in turns] == [
        ("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0),
    ]
    assert turns[1]["kind"] == "grounded"
    assert turns[1]["reference"] == "a passage"


def test_attach_real_turns_defaults_unlabelled_utterances_to_smalltalk():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0), ("DANIELLE", 11.0, 15.0)])
    llm = [_turn("DANIELLE", 2.5, 10.0, "grounded", "p")]
    turns = attach_real_turns(0.0, 15.0, llm, utts)
    assert [t["kind"] for t in turns] == ["smalltalk", "grounded", "smalltalk"]
    assert turns[2]["reference"] is None


def test_attach_real_turns_carries_the_utterance_text():
    utts = [{"speaker": "DANIELLE", "start": 0.0, "end": 5.0, "text": "the real words"}]
    turns = attach_real_turns(0.0, 5.0, [_turn("DANIELLE", 0.0, 5.0, "grounded", "p")], utts)
    assert turns[0]["text"] == "the real words"


def test_trim_drops_leading_danielle_and_trailing_joshua():
    turns = [_turn("DANIELLE", 0.0, 1.0), _turn("JOSHUA", 1.0, 2.0),
             _turn("DANIELLE", 2.0, 8.0), _turn("JOSHUA", 8.0, 9.0)]
    out = trim_to_speaker_bounds(turns)
    assert [t["speaker"] for t in out] == ["JOSHUA", "DANIELLE"]


def test_trim_returns_empty_when_no_valid_span_exists():
    assert trim_to_speaker_bounds([_turn("JOSHUA", 0.0, 1.0)]) == []
    assert trim_to_speaker_bounds([]) == []


def test_validate_trims_instead_of_dropping_a_bad_boundary():
    """The old behaviour discarded a good dialogue over a boundary rounding mismatch."""
    utts = _utts([("DANIELLE", 0.0, 1.0), ("JOSHUA", 2.0, 4.0),
                  ("DANIELLE", 5.0, 12.0), ("JOSHUA", 13.0, 14.0)])
    llm = [_turn("DANIELLE", 0.0, 1.0), _turn("JOSHUA", 2.0, 4.0),
           _turn("DANIELLE", 5.0, 12.0, "grounded", "p"), _turn("JOSHUA", 13.0, 14.0)]
    kept, problems = validate_segments([_seg("s0", 0.0, 14.0, llm)], utts)
    assert len(kept) == 1, problems
    assert kept[0]["turns"][0]["speaker"] == "JOSHUA"
    assert kept[0]["turns"][-1]["speaker"] == "DANIELLE"
    assert kept[0]["start"] == 2.0 and kept[0]["end"] == 12.0


from segment_retrieval_audio import clamp_to_window


def test_clamp_drops_segments_outside_the_window_shown_to_the_llm():
    """Gemini emits timestamps beyond its window; unclamped they snap to arbitrary
    places elsewhere in the recording and block legitimate segments as 'overlaps'."""
    segs = [
        {"id": "w1-0", "start": 700.0, "end": 760.0, "turns": []},   # inside
        {"id": "w1-1", "start": 2100.0, "end": 2160.0, "turns": []},  # far outside
    ]
    out = clamp_to_window(segs, 660.0, 1380.0)
    assert [s["id"] for s in out] == ["w1-0"]


def test_clamp_keeps_a_segment_straddling_the_edge_by_a_little():
    segs = [{"id": "w1-0", "start": 1370.0, "end": 1400.0, "turns": []}]
    out = clamp_to_window(segs, 660.0, 1380.0, tol=30.0)
    assert [s["id"] for s in out] == ["w1-0"]


def test_clamp_drops_inverted_or_empty_spans():
    segs = [{"id": "x", "start": 800.0, "end": 800.0, "turns": []},
            {"id": "y", "start": 900.0, "end": 850.0, "turns": []}]
    assert clamp_to_window(segs, 660.0, 1380.0) == []


def test_snap_bounds_only_considers_the_utterances_it_is_given():
    """Snapping must be window-local, or an out-of-range timestamp lands anywhere."""
    window = _utts([("JOSHUA", 700.0, 702.0), ("DANIELLE", 703.0, 710.0)])
    assert snap_bounds(701.0, 709.0, window) == (700.0, 710.0)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from filter_retrieval_segments import apply_verdicts, judgeable_turns


def _seg(sid, turns):
    return {"id": sid, "start": 0.0, "end": 50.0, "turns": turns}


def _t(speaker, kind, reference=None, text="said"):
    return {"speaker": speaker, "start": 0.0, "end": 1.0,
            "kind": kind, "reference": reference, "text": text}


def test_only_grounded_turns_are_judged():
    segs = [_seg("s0", [
        _t("JOSHUA", "smalltalk"),
        _t("DANIELLE", "grounded", "a passage"),
        _t("DANIELLE", "decline", "an unrelated passage"),
        _t("DANIELLE", "smalltalk"),
    ])]
    assert judgeable_turns(segs) == [("s0", 1)]


def test_grounded_turn_without_a_reference_is_not_judged():
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", None)])]
    assert judgeable_turns(segs) == []


def test_unfaithful_grounded_turn_is_demoted_not_deleted():
    segs = [_seg("s0", [
        _t("JOSHUA", "smalltalk"),
        _t("DANIELLE", "grounded", "wrong-topic passage"),
        _t("DANIELLE", "smalltalk"),
    ])]
    kept, dropped = apply_verdicts(segs, {"s0#1": False})
    assert len(kept) == 1
    # demoted to smalltalk so the clip survives but emits no <RAG>
    assert kept[0]["turns"][1]["kind"] == "smalltalk"
    assert kept[0]["turns"][1]["reference"] is None
    assert dropped == [{"segment": "s0", "turn": 1, "reason": "unfaithful"}]


def test_faithful_turn_is_untouched():
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", "good passage")])]
    kept, dropped = apply_verdicts(segs, {"s0#1": True})
    assert kept[0]["turns"][1]["kind"] == "grounded"
    assert kept[0]["turns"][1]["reference"] == "good passage"
    assert dropped == []


def test_segment_with_every_retrieval_turn_demoted_is_still_kept():
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", "bad")])]
    kept, _ = apply_verdicts(segs, {"s0#1": False})
    assert len(kept) == 1
    assert all(t["kind"] == "smalltalk" for t in kept[0]["turns"])


def test_apply_verdicts_does_not_mutate_the_input():
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", "bad")])]
    apply_verdicts(segs, {"s0#1": False})
    assert segs[0]["turns"][1]["kind"] == "grounded"
    assert segs[0]["turns"][1]["reference"] == "bad"


def test_turn_with_no_verdict_is_left_alone():
    """A judge call that errored must not silently demote a good turn."""
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", "passage")])]
    kept, dropped = apply_verdicts(segs, {})
    assert kept[0]["turns"][1]["kind"] == "grounded"
    assert dropped == []

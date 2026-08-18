"""persona -> retrieval conversion (no network; gen_reference is always injected).

Decision of record (2026-08-07 night): the 127 identity answers in the persona recording,
previously guarded AGAINST retrieval marking (tests/test_persona_guard.py), are now converted
TO retrieval turns — live evidence showed the retrieval trigger+grounding pipeline works
excellently while weights-based identity keeps failing. The loss mask makes mixing grounded
identity turns in with the unmarked long-form copy safe.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from mark_persona_retrieval import mark_persona  # noqa: E402


def _turn(kind, speaker="DANIELLE", start=0.0, end=1.0, reference=None, text="words"):
    return {"speaker": speaker, "start": start, "end": end,
            "kind": kind, "reference": reference, "text": text}


def test_persona_turns_become_grounded_with_reference():
    def fake_gen(question, answer):
        return "Reference: she said something."

    segment = {"id": "s0", "turns": [
        _turn("smalltalk", speaker="JOSHUA", text="What's your name?"),
        _turn("persona", text="I'm Danielle."),
    ]}
    out, stats = mark_persona([segment], fake_gen)

    turn = out[0]["turns"][1]
    assert turn["kind"] == "grounded"
    assert turn["reference"] == "Reference: she said something."
    assert stats["converted"] == 1
    assert stats["references_generated"] == 1


def test_non_persona_turns_pass_through_unchanged():
    def fake_gen(question, answer):
        raise AssertionError("gen_reference must not be called for non-persona turns")

    segment = {"id": "s1", "turns": [
        _turn("decline", reference="a passage"),
        _turn("grounded", reference="another passage"),
        _turn("smalltalk", speaker="JOSHUA", reference=None),
    ]}
    out, stats = mark_persona([segment], fake_gen)

    assert out[0]["turns"] == segment["turns"]
    assert stats["declines_untouched"] == 1
    assert stats["grounded_untouched"] == 1
    assert stats["converted"] == 0


def test_only_danielle_turns_convert():
    def fake_gen(question, answer):
        raise AssertionError("gen_reference must not be called for JOSHUA turns")

    segment = {"id": "s2", "turns": [
        _turn("persona", speaker="JOSHUA", text="something"),
    ]}
    out, stats = mark_persona([segment], fake_gen)

    assert out[0]["turns"][0]["kind"] == "persona"
    assert out[0]["turns"][0] == segment["turns"][0]
    assert stats["converted"] == 0


def test_failed_generation_leaves_turn_as_persona():
    calls = {"n": 0}

    def flaky_gen(question, answer):
        calls["n"] += 1
        raise RuntimeError("api down")

    segment = {"id": "s3", "turns": [
        _turn("persona", text="I'm Danielle."),
    ]}
    out, stats = mark_persona([segment], flaky_gen)

    assert out[0]["turns"][0]["kind"] == "persona"
    assert out[0]["turns"][0]["reference"] is None
    assert stats["skipped"] == 1
    assert stats["converted"] == 0
    assert calls["n"] == 2  # one retry


def test_reference_prefix_enforced():
    def bare_gen(question, answer):
        return "she lives in Tampa"

    segment = {"id": "s4", "turns": [
        _turn("persona", text="I live in Tampa."),
    ]}
    out, stats = mark_persona([segment], bare_gen)

    assert out[0]["turns"][0]["reference"] == "Reference: she lives in Tampa"
    assert stats["converted"] == 1

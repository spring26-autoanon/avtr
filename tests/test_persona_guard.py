"""Identity answers must never carry a <RAG> marker.

Tonight's Part 1 is ~20 min of Danielle stating specific facts about herself. The segmenter
prompt defines "grounded" as answering with specific external facts, so Gemini will label
those grounded and write reference passages — marking ~45 identity answers and teaching the
model to fire a retrieval whenever it is asked its name.

Decision of record (spec 3.2): persona lives in the weights, not in retrieval. A trigger that
fires ~50% of the time is a bad place to keep the model's own name.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cut_retrieval_clips import _retrieval_turns          # noqa: E402
from segment_retrieval_audio import normalize, turn_mix   # noqa: E402


def _turn(kind, speaker="DANIELLE", start=0.0, end=1.0, reference="a passage"):
    return {"speaker": speaker, "start": start, "end": end,
            "kind": kind, "reference": reference, "text": "words"}


def test_persona_is_a_recognised_kind():
    """It must survive normalize, not degrade to smalltalk — we need to count it."""
    out = normalize([{"start": 0.0, "end": 2.0, "turns": [_turn("persona")]}], 0)
    assert out[0]["turns"][0]["kind"] == "persona"


def test_persona_turns_carry_no_reference():
    """A reference would be precomputed into a tensor with no marker to attach it to."""
    out = normalize([{"start": 0.0, "end": 2.0, "turns": [_turn("persona")]}], 0)
    assert out[0]["turns"][0]["reference"] is None


def test_persona_never_becomes_a_retrieval_turn():
    """The whole point: no <RAG> on 'What's your name?'."""
    segment = {"turns": [_turn("smalltalk", speaker="JOSHUA"),
                         _turn("persona", start=1.0, end=2.0)]}
    assert _retrieval_turns(segment) == []


def test_grounded_still_becomes_a_retrieval_turn():
    """Guard must not suppress real retrieval."""
    segment = {"turns": [_turn("smalltalk", speaker="JOSHUA"),
                         _turn("grounded", start=1.0, end=2.0)]}
    assert len(_retrieval_turns(segment)) == 1


def test_turn_mix_counts_persona_separately():
    segments = [{"turns": [_turn("persona"), _turn("persona"), _turn("grounded")]}]
    mix = turn_mix(segments)
    assert mix["persona"] == 2
    assert mix["grounded"] == 1


def test_persona_is_excluded_from_retrieval_share():
    """retrieval_share sets RAG_TOKEN_WEIGHT. Counting 45 persona turns as retrieval would
    inflate it and pick the wrong weight."""
    segments = [{"turns": [_turn("persona")] * 8 + [_turn("grounded")] * 2}]
    assert turn_mix(segments)["retrieval_share"] == 0.2


def test_unknown_kinds_still_degrade_to_smalltalk():
    """The existing safety net must survive the change."""
    out = normalize([{"start": 0.0, "end": 2.0, "turns": [_turn("weird")]}], 0)
    assert out[0]["turns"][0]["kind"] == "smalltalk"
    assert out[0]["turns"][0]["reference"] is None


def test_joshua_turns_are_never_persona():
    """Only her turns get a kind; his are always smalltalk."""
    out = normalize(
        [{"start": 0.0, "end": 2.0, "turns": [_turn("persona", speaker="JOSHUA")]}], 0)
    assert out[0]["turns"][0]["kind"] == "smalltalk"


def test_the_prompt_defines_the_persona_label():
    """Gemini cannot emit a label it was never told about."""
    src = (REPO / "scripts/segment_retrieval_audio.py").read_text()
    assert '"persona"' in src
    assert "about herself" in src.lower() or "about yourself" in src.lower()

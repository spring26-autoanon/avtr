import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_utterances import format_transcript, merge_utterances


def test_groups_words_within_gap_into_one_utterance():
    ch0 = [["hello", [1.0, 1.2], "SPEAKER_MAIN"], ["there", [1.3, 1.5], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, [], gap=0.6)
    assert len(utts) == 1
    assert utts[0]["speaker"] == "DANIELLE"
    assert utts[0]["text"] == "hello there"
    assert utts[0]["start"] == 1.0
    assert utts[0]["end"] == 1.5


def test_splits_on_gap_larger_than_threshold():
    ch0 = [["hello", [1.0, 1.2], "SPEAKER_MAIN"], ["later", [3.0, 3.2], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, [], gap=0.6)
    assert [u["text"] for u in utts] == ["hello", "later"]


def test_two_channels_interleave_in_time_order():
    ch0 = [["yes", [2.0, 2.2], "SPEAKER_MAIN"]]
    ch1 = [["question", [1.0, 1.4], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, ch1)
    assert [(u["speaker"], u["text"]) for u in utts] == [
        ("JOSHUA", "question"),
        ("DANIELLE", "yes"),
    ]


def test_speaker_change_breaks_utterance_even_within_gap():
    ch0 = [["a", [1.0, 1.1], "SPEAKER_MAIN"], ["c", [1.4, 1.5], "SPEAKER_MAIN"]]
    ch1 = [["b", [1.2, 1.3], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, ch1, gap=0.6)
    assert [(u["speaker"], u["text"]) for u in utts] == [
        ("DANIELLE", "a"), ("JOSHUA", "b"), ("DANIELLE", "c"),
    ]


def test_strips_whisper_leading_spaces():
    ch0 = [[" hello", [1.0, 1.2], "SPEAKER_MAIN"], [" there", [1.3, 1.5], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, [])
    assert utts[0]["text"] == "hello there"


def test_format_transcript_is_timestamped_and_labelled():
    utts = [
        {"speaker": "JOSHUA", "start": 12.4, "end": 15.0, "text": "what is diwali"},
        {"speaker": "DANIELLE", "start": 19.1, "end": 24.0, "text": "the festival of lights"},
    ]
    text = format_transcript(utts)
    assert text.splitlines() == [
        "[12.4] JOSHUA: what is diwali",
        "[19.1] DANIELLE: the festival of lights",
    ]


def test_format_transcript_uses_raw_seconds_not_mm_ss():
    """The segmenter must echo absolute seconds back; MM:SS made it convert by hand and it
    got whole windows wrong. The printed number is the number it should return."""
    utts = [{"speaker": "DANIELLE", "start": 4805.2, "end": 4806.0, "text": "late"}]
    assert format_transcript(utts) == "[4805.2] DANIELLE: late"


def test_default_gap_merges_a_pause_that_would_split_one_answer():
    """Default is 1.5 s, measured from the real recording (see merge_utterances docstring):
    at 0.6 s the merge over-splits her turns by 37% against the speaker-change floor."""
    ch0 = [["answer", [1.0, 1.4], "SPEAKER_MAIN"], ["continues", [2.2, 2.6], "SPEAKER_MAIN"]]
    assert len(merge_utterances(ch0, [])) == 1              # 0.8 s pause -> still one turn
    assert len(merge_utterances(ch0, [], gap=0.6)) == 2     # old default would have split it


def test_default_gap_still_splits_a_real_turn_boundary():
    ch0 = [["done", [1.0, 1.4], "SPEAKER_MAIN"], ["new", [5.0, 5.4], "SPEAKER_MAIN"]]
    assert len(merge_utterances(ch0, [])) == 2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from pair_dialogue_stereo import parse_track_name


def test_parse_clay():
    assert parse_track_name("audioClayS11556527425_24khz.wav") == ("clays", "1", "1556527425")


def test_parse_danielle_idx2():
    assert parse_track_name("audioDanielleDeLosa21341305451_24khz.wav") == (
        "danielledelosa", "2", "1341305451",
    )


def test_parse_joshua():
    assert parse_track_name("audioJoshuaRhodes21411304343_24khz.wav") == (
        "joshuarhodes", "2", "1411304343",
    )


def test_parse_rejects_short_id():
    with pytest.raises(ValueError):
        parse_track_name("audioBob123_24khz.wav")


def test_parse_rejects_no_audio_prefix():
    with pytest.raises(ValueError):
        parse_track_name("ClayS11556527425_24khz.wav")

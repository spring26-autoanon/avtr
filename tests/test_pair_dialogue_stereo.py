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


import numpy as np
import soundfile as sf
from pair_dialogue_stereo import combine_to_stereo


def _mono_wav(path, seconds, value, sr=24000):
    n = int(seconds * sr)
    sf.write(str(path), np.full(n, value, dtype="float32"), sr, subtype="PCM_24")


def test_combine_left_is_main_right_is_partner(tmp_path):
    main = tmp_path / "main.wav"
    partner = tmp_path / "partner.wav"
    _mono_wav(main, 1.0, 0.5)
    _mono_wav(partner, 1.0, -0.25)
    stereo, sr, delta = combine_to_stereo(main, partner)
    assert sr == 24000
    assert stereo.shape == (24000, 2)
    assert delta == 0
    assert np.allclose(stereo[:, 0], 0.5, atol=1e-4)
    assert np.allclose(stereo[:, 1], -0.25, atol=1e-4)


def test_combine_truncates_to_shorter(tmp_path):
    main = tmp_path / "main.wav"
    partner = tmp_path / "partner.wav"
    _mono_wav(main, 2.0, 0.5)     # 48000 frames
    _mono_wav(partner, 1.5, -0.25)  # 36000 frames
    stereo, sr, delta = combine_to_stereo(main, partner)
    assert stereo.shape == (36000, 2)
    assert delta == 12000


def test_combine_rejects_wrong_sr(tmp_path):
    main = tmp_path / "main.wav"
    partner = tmp_path / "partner.wav"
    _mono_wav(main, 1.0, 0.5, sr=16000)
    _mono_wav(partner, 1.0, -0.25)
    with pytest.raises(SystemExit):
        combine_to_stereo(main, partner)


from pair_dialogue_stereo import carve_eval


def test_carve_splits_contiguously():
    stereo = np.arange(10 * 24000 * 2, dtype="float32").reshape(10 * 24000, 2)
    before, ev, after = carve_eval(stereo, 24000, eval_sec=2.0, center_frac=0.5)
    assert ev.shape[0] == 2 * 24000
    assert before.shape[0] == 4 * 24000   # centered: start at 5s-1s = 4s
    assert after.shape[0] == 4 * 24000
    assert np.array_equal(np.concatenate([before, ev, after]), stereo)


def test_carve_rejects_too_long():
    stereo = np.zeros((1 * 24000, 2), dtype="float32")
    with pytest.raises(SystemExit):
        carve_eval(stereo, 24000, eval_sec=2.0)

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


from pair_dialogue_stereo import discover_conversations, main


def _fake_source(dirpath):
    # conv A (no eval): main=danielle val 0.5, partner=clay val -0.25
    _mono_wav(dirpath / "audioDanielleDeLosa20000000001_24khz.wav", 1.0, 0.5)
    _mono_wav(dirpath / "audioClayS10000000001_24khz.wav", 1.0, -0.25)
    # conv B (eval target): main=danielle val 0.5, partner=joshua val -0.5
    _mono_wav(dirpath / "audioDanielleDeLosa10000000002_24khz.wav", 10.0, 0.5)
    _mono_wav(dirpath / "audioJoshuaRhodes20000000002_24khz.wav", 10.0, -0.5)


def test_discover_pairs_by_conv_and_identity(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _fake_source(src)
    convs = discover_conversations(src, "danielle")
    assert set(convs) == {"0000000001", "0000000002"}
    main_path, main_spk, partner_path, partner_spk = convs["0000000001"]
    assert main_spk == "danielledelosa" and partner_spk == "clays"


def test_main_writes_expected_files_and_channels(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _fake_source(src)
    main([
        "--src", str(src), "--dst", str(dst),
        "--main-name", "danielle",
        "--eval-conv", "0000000002", "--eval-sec", "2.0", "--eval-center-frac", "0.5",
    ])
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert names == [
        "danielle_clays.wav",
        "danielle_joshuarhodes_eval.wav",
        "danielle_joshuarhodes_train_a.wav",
        "danielle_joshuarhodes_train_b.wav",
    ]
    # channel assignment + format on the non-eval conv
    data, sr = sf.read(str(dst / "danielle_clays.wav"), always_2d=True)
    assert sr == 24000 and data.shape[1] == 2
    assert np.allclose(data[:, 0], 0.5, atol=1e-4)   # left = main
    assert np.allclose(data[:, 1], -0.25, atol=1e-4)  # right = partner
    assert sf.info(str(dst / "danielle_clays.wav")).subtype == "PCM_24"
    # eval slice length
    ev = sf.info(str(dst / "danielle_joshuarhodes_eval.wav"))
    assert ev.frames == 2 * 24000


REAL_SRC = Path(__file__).resolve().parents[1] / "finetune/data/datastereo/clean_moshi_audio_24khz"
REAL_DST = Path(__file__).resolve().parents[1] / "finetune/data/prepared_dialogue"


@pytest.mark.skipif(
    not (REAL_DST / "danielle_clays.wav").exists(),
    reason="run pair_dialogue_stereo.py on the real recordings first",
)
def test_real_outputs_format_and_channels():
    expected = {
        "danielle_clays.wav",
        "danielle_joshuarhodes_train_a.wav",
        "danielle_joshuarhodes_eval.wav",
        "danielle_joshuarhodes_train_b.wav",
        "danielle_leenatantawy.wav",
    }
    assert {p.name for p in REAL_DST.glob("*.wav")} == expected
    for name in expected:
        info = sf.info(str(REAL_DST / name))
        assert info.channels == 2, name
        assert info.samplerate == 24000, name
        assert info.subtype == "PCM_24", name

    # eval slice is exactly 600 s
    assert sf.info(str(REAL_DST / "danielle_joshuarhodes_eval.wav")).frames == 600 * 24000

    # Joshua split lengths sum to the original conversation (within 1 frame)
    orig = sf.info(str(REAL_SRC / "audioDanielleDeLosa11411304343_24khz.wav")).frames
    parts = sum(
        sf.info(str(REAL_DST / n)).frames
        for n in ("danielle_joshuarhodes_train_a.wav",
                  "danielle_joshuarhodes_eval.wav",
                  "danielle_joshuarhodes_train_b.wav")
    )
    assert abs(parts - orig) <= 1

    # left channel == Danielle source, right == Clay source (first 1 s, exact copy)
    out, _ = sf.read(str(REAL_DST / "danielle_clays.wav"), start=0, stop=24000, always_2d=True)
    dan, _ = sf.read(str(REAL_SRC / "audioDanielleDeLosa21556527425_24khz.wav"),
                     start=0, stop=24000, always_2d=True)
    clay, _ = sf.read(str(REAL_SRC / "audioClayS11556527425_24khz.wav"),
                      start=0, stop=24000, always_2d=True)
    assert np.array_equal(out[:, 0], dan[:, 0])
    assert np.array_equal(out[:, 1], clay[:, 0])

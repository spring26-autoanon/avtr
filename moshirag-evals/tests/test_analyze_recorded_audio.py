"""Tests for scripts/analyze_recorded_audio.py.

Deliberately bypasses ffmpeg/webm decoding (a real system dependency, not
mockable the way the rest of this project mocks torch/moshi) by writing
real, synthetic WAV files directly via the stdlib `wave` module and feeding
them straight to compute_rms_envelope()/find_acoustic_events() — the actual
signal-processing logic this script exists to get right. Parsing/alignment
helpers are tested against real text fixtures shaped like real server.log
lines and real recording filenames.
"""

import struct
import wave
from pathlib import Path

import pytest

from scripts.analyze_recorded_audio import (
    CONNECTED_LINE_RE,
    compute_rms_envelope,
    find_acoustic_events,
    find_websocket_connect_time,
    format_abs_s,
    parse_hhmmss_cs,
    parse_recording_start_offset,
    parse_server_events,
)

SAMPLE_RATE = 16000


def _write_wav(path: Path, segments: list[tuple[float, int]]) -> None:
    """segments: [(duration_s, amplitude), ...]. amplitude 0 = silence."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for duration_s, amplitude in segments:
            n = int(duration_s * SAMPLE_RATE)
            frames = struct.pack(f"<{n}h", *([amplitude] * n))
            wf.writeframes(frames)


def test_compute_rms_envelope_silence_is_near_zero(tmp_path):
    wav_path = tmp_path / "silence.wav"
    _write_wav(wav_path, [(0.5, 0)])
    envelope = compute_rms_envelope(wav_path)
    assert len(envelope) > 0
    assert all(rms < 5 for _, rms in envelope)


def test_compute_rms_envelope_tone_has_high_rms(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, [(0.5, 10000)])
    envelope = compute_rms_envelope(wav_path)
    assert all(rms > 5000 for _, rms in envelope)


def test_find_acoustic_events_detects_onset_after_silence(tmp_path):
    wav_path = tmp_path / "silence_then_tone.wav"
    _write_wav(wav_path, [(0.3, 0), (0.3, 10000)])
    envelope = compute_rms_envelope(wav_path)
    events = find_acoustic_events(envelope)
    onsets = [e for e in events if e.kind == "onset"]
    assert len(onsets) == 1
    # Onset should land close to the real silence/tone boundary (0.3s),
    # not drift by more than a couple of windows' worth of sustain-count delay.
    assert 0.25 < onsets[0].abs_s < 0.4


def test_find_acoustic_events_detects_offset(tmp_path):
    wav_path = tmp_path / "tone_then_silence.wav"
    _write_wav(wav_path, [(0.3, 10000), (0.3, 0)])
    envelope = compute_rms_envelope(wav_path)
    events = find_acoustic_events(envelope)
    offsets = [e for e in events if e.kind == "offset"]
    assert len(offsets) == 1


def test_find_acoustic_events_pure_silence_has_no_events(tmp_path):
    wav_path = tmp_path / "all_silence.wav"
    _write_wav(wav_path, [(0.5, 0)])
    envelope = compute_rms_envelope(wav_path)
    events = find_acoustic_events(envelope)
    assert events == []


def test_find_acoustic_events_flags_a_real_discontinuity(tmp_path):
    # Sustained moderate tone, then an abrupt jump to a much louder tone
    # mid-speech (not from silence) — the hard-splice-artifact proxy.
    wav_path = tmp_path / "discontinuity.wav"
    _write_wav(wav_path, [(0.3, 5000), (0.3, 30000)])
    envelope = compute_rms_envelope(wav_path)
    events = find_acoustic_events(envelope)
    discontinuities = [e for e in events if e.kind == "discontinuity"]
    assert len(discontinuities) >= 1


def test_parse_hhmmss_cs():
    assert parse_hhmmss_cs("06", "20", "12", "08") == 6 * 3600 + 20 * 60 + 12 + 0.08


def test_format_abs_s_roundtrip():
    s = 6 * 3600 + 20 * 60 + 12.08
    assert format_abs_s(s) == "06:20:12.08"


def test_connected_line_regex_matches_real_log_line():
    line = "06:19:20.67 INFO    new WebSocket client connected"
    m = CONNECTED_LINE_RE.match(line)
    assert m is not None
    assert m.groups() == ("06", "19", "20", "67")


def test_find_websocket_connect_time(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "06:18:50.17 INFO    Using Reference Encoder service at: http://localhost:8001\n"
        "06:19:20.67 INFO    new WebSocket client connected\n"
        "06:19:31.35 INFO    [Slot] acquired slot 0 (1/16)\n"
    )
    connect_s = find_websocket_connect_time(log)
    assert connect_s == 6 * 3600 + 19 * 60 + 20.67


def test_find_websocket_connect_time_raises_when_missing(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("06:18:50.17 INFO    Using Reference Encoder service at: http://localhost:8001\n")
    with pytest.raises(ValueError, match="No 'new WebSocket client connected'"):
        find_websocket_connect_time(log)


def test_parse_recording_start_offset():
    path = Path("moshirag-audio_startT6.234s-2026-07-28T06-38-14-000Z.webm")
    assert parse_recording_start_offset(path) == pytest.approx(6.234)


def test_parse_recording_start_offset_raises_without_tag():
    path = Path("moshirag-audio-2026-07-28T06-38-14-000Z.webm")
    with pytest.raises(ValueError, match="no _startT"):
        parse_recording_start_offset(path)


def test_parse_server_events_extracts_interesting_lines_only(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "06:19:20.67 INFO    new WebSocket client connected\n"
        "06:19:50.38 INFO    [VAD] User started speaking (history=['0.5'])\n"
        "06:19:51.66 INFO    [VAD] User stopped speaking (history=['0.7']), waiting\n"
        "06:19:52.26 INFO    [VAD] Waiting ended, switching to model\n"
        "06:19:52.30 INFO    Some irrelevant line that should be filtered out\n"
        "06:19:52.35 INFO    [Display Model] 'the'\n"
    )
    events = parse_server_events(log)
    assert len(events) == 4
    assert "irrelevant" not in " ".join(e.text for e in events)
    assert events[0].text.startswith("[VAD] User started speaking")


def test_parse_server_events_strips_ansi_color_codes(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("06:19:50.38 \x1b[32mINFO\x1b[0m    [VAD] User started speaking (history=['0.5'])\n")
    events = parse_server_events(log)
    assert len(events) == 1
    assert events[0].abs_s == 6 * 3600 + 19 * 60 + 50.38

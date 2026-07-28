#!/usr/bin/env python3
"""
Ground-truth acoustic validation for a demo session — cross-references the
client's saved conversation recording (demo/client/src/pages/Conversation/
hooks/useRecording.ts's "Save audio", which records the same
AudioWorkletNode output that drives the speakers, mixed with the user's own
mic input — see that file's docstring on audioStreamDestination) against
server.log's own event timestamps, using real waveform energy instead of
internal JS/Python counters.

Why this exists: every diagnostic built earlier in this investigation
(server-side text-token timestamps, the client's own buffer-occupancy
counters) measures a proxy for what the model *decided* to do, not what a
person actually heard. One of those proxies (a "delay" field in the client
diagnostic) was flatly measuring the wrong thing. This script instead
decodes the actual recorded audio and finds real onset/silence transitions
from its energy envelope — a signal that can't be fooled by a bug in this
project's own instrumentation.

Timing alignment (read this before trusting any output): the recording's
filename embeds `_startT<X>s` (see useRecording.ts's saveAudio()) — X is
seconds from the *client's* WebSocket-open moment to when recording started.
That client-side t_rel_s zero point is NOT the same as raw_events.jsonl's
own t_rel_s (that one is anchored to SessionLog construction / server
process startup, i.e. before model loading even begins — confirmed by
reading SessionLog.__init__'s self._t0 = time.perf_counter() directly, not
assumed). The one thing both the client and server genuinely share is the
real-world moment the WebSocket connection completes, which server.log logs
verbatim as "new WebSocket client connected" with a real HH:MM:SS
timestamp. So alignment here goes:

    recording_start_wallclock = server.log's "new WebSocket client
                                 connected" timestamp + the filename's X

and every acoustic event found in the recording is reported at
`recording_start_wallclock + position_in_recording_seconds`, directly
comparable to every other HH:MM:SS timestamp already in server.log.

Requires ffmpeg on PATH to decode the recording (webm/opus) to a raw WAV —
not a new Python dependency, a system binary. Everything after that uses
only the standard library (wave + audioop) so no new `uv add` is needed.

Usage:
  uv run scripts/analyze_recorded_audio.py \\
      demo/sessions/2026-07-28T06-38-14Z/ \\
      demo/sessions/2026-07-28T06-38-14Z/moshirag-audio_startT6.234s-2026-....webm
"""

from __future__ import annotations

import argparse
import audioop  # deprecated, slated for removal in Python 3.13 — this
                # project targets >=3.10 (currently 3.11); revisit if a
                # future move to 3.13+ is ever considered.
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

WINDOW_MS = 20
# An onset/offset must persist for at least this many consecutive windows
# to count — filters out single-window noise blips, not real speech.
MIN_SUSTAIN_WINDOWS = 3
# A window's RMS energy counts as "speaking" once it's at least this
# fraction of the whole recording's own peak RMS — self-calibrating per
# recording rather than a fixed dBFS threshold, since mic gain/volume
# varies by machine.
SPEECH_FRACTION_OF_PEAK = 0.08
# A discontinuity candidate: energy jumps by at least this much between
# adjacent windows *without* being a clean onset from silence — a proxy
# for an audible hard-splice artifact (see the "chronic jitter-buffer
# overrun" investigation's own description of drops as discontinuities).
DISCONTINUITY_JUMP_FRACTION = 0.5

FILENAME_START_TAG_RE = re.compile(r"_startT([0-9.]+)s")
CONNECTED_LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{2})\s+INFO\s+new WebSocket client connected")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SERVER_EVENT_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})\.(\d{2})\s+\S+\s+(.*)$"
)
# Event lines worth surfacing in the merged timeline — kept narrow and
# explicit rather than "every INFO line" so the output stays readable.
INTERESTING_EVENT_SUBSTRINGS = (
    "[VAD] User stopped speaking (",
    "[VAD] Waiting ended, switching to model",
    "[VAD] User started speaking",
    "[RAG] model emitted RAG token",
    "[Reference] Triggering retrieval",
    "[Reference] received reference text",
    "[Reference] ARC encoding received",
    "[Display Model] '",
    "[Display User] '",
)


@dataclass
class AcousticEvent:
    abs_s: float
    kind: str  # "onset", "offset", "discontinuity"
    detail: str


@dataclass
class ServerEvent:
    abs_s: float
    text: str


def parse_hhmmss_cs(h: str, m: str, s: str, cs: str) -> float:
    """HH:MM:SS.cc -> seconds since midnight. cs is centiseconds (2 digits,
    matching moshi-rag's own _ColorFormatter.formatTime — see server.py)."""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def find_websocket_connect_time(server_log_path: Path) -> float:
    text = server_log_path.read_text(errors="replace")
    for line in text.splitlines():
        line = ANSI_RE.sub("", line)
        m = CONNECTED_LINE_RE.match(line)
        if m:
            return parse_hhmmss_cs(*m.groups())
    raise ValueError(
        f"No 'new WebSocket client connected' line found in {server_log_path} — "
        "is this the right session, and did a client actually connect?"
    )


def parse_recording_start_offset(recording_path: Path) -> float:
    m = FILENAME_START_TAG_RE.search(recording_path.name)
    if not m:
        raise ValueError(
            f"{recording_path.name} has no _startT<X>s tag — this recording predates "
            "the timing-alignment fix (useRecording.ts's saveAudio()); re-record."
        )
    return float(m.group(1))


def decode_to_wav(recording_path: Path, wav_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. This script decodes the recording via ffmpeg "
            "(a system binary, not a new Python dependency) — install it wherever "
            "this script actually runs (e.g. `apt install ffmpeg` on the machine "
            "that has the .webm file)."
        )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(recording_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        check=True,
        capture_output=True,
    )


def compute_rms_envelope(wav_path: Path) -> list[tuple[float, float]]:
    """Returns [(offset_in_recording_s, rms), ...] at WINDOW_MS resolution."""
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        window_frames = max(1, int(sample_rate * WINDOW_MS / 1000))
        envelope: list[tuple[float, float]] = []
        frames_read = 0
        idx = 0
        while frames_read < n_frames:
            chunk = wf.readframes(window_frames)
            if not chunk:
                break
            rms = audioop.rms(chunk, sample_width)
            envelope.append((idx * WINDOW_MS / 1000.0, float(rms)))
            frames_read += window_frames
            idx += 1
    return envelope


def find_acoustic_events(envelope: list[tuple[float, float]]) -> list[AcousticEvent]:
    if not envelope:
        return []
    peak = max(rms for _, rms in envelope) or 1.0
    threshold = peak * SPEECH_FRACTION_OF_PEAK
    jump_threshold = peak * DISCONTINUITY_JUMP_FRACTION

    events: list[AcousticEvent] = []
    speaking = False
    sustain_count = 0
    pending_state = False

    for i, (t, rms) in enumerate(envelope):
        above = rms >= threshold
        if above != pending_state:
            pending_state = above
            sustain_count = 1
        else:
            sustain_count += 1

        if sustain_count == MIN_SUSTAIN_WINDOWS and pending_state != speaking:
            speaking = pending_state
            events.append(
                AcousticEvent(
                    abs_s=t,
                    kind="onset" if speaking else "offset",
                    detail=f"rms={rms:.0f} (peak={peak:.0f}, threshold={threshold:.0f})",
                )
            )

        if i > 0:
            prev_rms = envelope[i - 1][1]
            jump = abs(rms - prev_rms)
            # Only flag jumps while already speaking — a jump from silence
            # is just a normal onset, already captured above.
            if speaking and jump >= jump_threshold and sustain_count > MIN_SUSTAIN_WINDOWS:
                events.append(
                    AcousticEvent(
                        abs_s=t,
                        kind="discontinuity",
                        detail=f"jump={jump:.0f} (prev_rms={prev_rms:.0f}, rms={rms:.0f})",
                    )
                )

    return events


def parse_server_events(server_log_path: Path) -> list[ServerEvent]:
    events: list[ServerEvent] = []
    text = server_log_path.read_text(errors="replace")
    for line in text.splitlines():
        line = ANSI_RE.sub("", line)
        if not any(sub in line for sub in INTERESTING_EVENT_SUBSTRINGS):
            continue
        m = SERVER_EVENT_RE.match(line)
        if not m:
            continue
        h, mnt, s, cs, rest = m.groups()
        events.append(ServerEvent(abs_s=parse_hhmmss_cs(h, mnt, s, cs), text=rest.strip()))
    return events


def format_abs_s(abs_s: float) -> str:
    abs_s = abs_s % 86400
    h = int(abs_s // 3600)
    m = int((abs_s % 3600) // 60)
    s = abs_s % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", help="demo/sessions/<session_id>/ directory (needs server.log inside it)")
    parser.add_argument("recording", help="Path to the saved moshirag-audio_startT<X>s-....webm recording")
    parser.add_argument("--keep-wav", action="store_true", help="Keep the decoded .wav next to the recording (default: delete after analysis)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    recording_path = Path(args.recording)
    server_log_path = session_dir / "server.log"
    if not server_log_path.is_file():
        print(f"No server.log found in {session_dir}", file=sys.stderr)
        sys.exit(1)
    if not recording_path.is_file():
        print(f"Recording not found: {recording_path}", file=sys.stderr)
        sys.exit(1)

    connect_abs_s = find_websocket_connect_time(server_log_path)
    start_offset_s = parse_recording_start_offset(recording_path)
    recording_start_abs_s = connect_abs_s + start_offset_s

    print(f"WebSocket connected at:      {format_abs_s(connect_abs_s)}")
    print(f"Recording start offset:      +{start_offset_s:.3f}s")
    print(f"=> Recording starts at:      {format_abs_s(recording_start_abs_s)}")
    print()

    wav_path = recording_path.with_suffix(".wav")
    decode_to_wav(recording_path, wav_path)
    try:
        envelope = compute_rms_envelope(wav_path)
    finally:
        if not args.keep_wav:
            wav_path.unlink(missing_ok=True)

    if not envelope:
        print("Decoded recording is empty — nothing to analyze.", file=sys.stderr)
        sys.exit(1)

    acoustic_events = find_acoustic_events(envelope)
    server_events = parse_server_events(server_log_path)

    timeline: list[tuple[float, str]] = []
    for e in acoustic_events:
        marker = {"onset": "[ACOUSTIC] speech onset", "offset": "[ACOUSTIC] speech offset", "discontinuity": "[ACOUSTIC] possible glitch"}[e.kind]
        timeline.append((recording_start_abs_s + e.abs_s, f"{marker} ({e.detail})"))
    for se in server_events:
        timeline.append((se.abs_s, se.text))

    timeline.sort(key=lambda x: x[0])

    print(f"n_acoustic_events={len(acoustic_events)}  n_server_events={len(server_events)}")
    print(f"Recording duration: {envelope[-1][0]:.1f}s")
    print()
    print("=== Merged timeline (server.log events + real acoustic onsets/offsets/glitches) ===")
    for abs_s, text in timeline:
        print(f"{format_abs_s(abs_s)}  {text}")


if __name__ == "__main__":
    main()

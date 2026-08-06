#!/usr/bin/env python3
"""Merge two channels of Whisper word alignments into speaker-tagged utterances.

annotate.py writes {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]} per channel.
Channel 0 is Danielle (the cloned voice), channel 1 is her partner. We interleave both in
time order and group consecutive same-speaker words separated by less than `gap` seconds.

Both channels are labelled SPEAKER_MAIN on disk because annotate.py transcribes whichever
channel it is pointed at as the main speaker; the real speaker identity comes from which
FILE the alignments were read out of, which is why this module takes them as two arguments.
"""
import json

DANIELLE = "DANIELLE"
JOSHUA = "JOSHUA"


def _words(alignments, speaker):
    out = []
    for word, (start, end), _spk in alignments:
        text = word.strip()
        if not text:
            continue
        out.append({"speaker": speaker, "start": float(start), "end": float(end), "text": text})
    return out


def merge_utterances(ch0, ch1, gap: float = 0.6):
    """Interleave both channels in time order, grouping same-speaker words < `gap` apart."""
    words = _words(ch0, DANIELLE) + _words(ch1, JOSHUA)
    words.sort(key=lambda w: (w["start"], w["speaker"]))

    utts = []
    for w in words:
        if utts and utts[-1]["speaker"] == w["speaker"] and w["start"] - utts[-1]["end"] < gap:
            utts[-1]["text"] += " " + w["text"]
            utts[-1]["end"] = w["end"]
        else:
            utts.append(dict(w))
    return utts


def _stamp(t: float) -> str:
    """Seconds -> MM:SS.s. Minutes are not wrapped at 60 — the recording runs to 85 min."""
    minutes = int(t) // 60
    return f"{minutes:02d}:{t - 60 * minutes:04.1f}"


def format_transcript(utts) -> str:
    return "\n".join(f"[{_stamp(u['start'])}] {u['speaker']}: {u['text']}" for u in utts)


def load_alignments(path):
    with open(path) as f:
        return json.load(f)["alignments"]

#!/usr/bin/env python3
"""Stage 3 — segment the retrieval recording into per-dialogue units with per-turn labels.

Reads the two Whisper transcripts (channel 0 = Danielle, channel 1 = Joshua), merges them
into a speaker-tagged timestamped transcript, and asks Gemini to mark dialogue boundaries
and label each of HER turns grounded / decline / smalltalk. Grounded and decline turns get
a reference passage — the "retrieved document" the trainer encodes via the :8001 ARC encoder.

Per-turn (not per-segment) labelling is what lets one clip teach the mode switch
chit-chat -> factual -> chit-chat, which is the Stage 3 goal.

LLM timestamps are never trusted: every boundary snaps to a real utterance edge and is then
validated (monotonic, non-overlapping, starts on JOSHUA, ends on DANIELLE, 4-100 s).

    export GEMINI_API_KEY=...
    python3 scripts/segment_retrieval_audio.py \
        --ch0 finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
        --ch1 finetune/data/prepared_retrieval/danielle_joshuarhodes.ch1.json \
        --out replay/retrieval_segments.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_utterances import format_transcript, load_alignments, merge_utterances  # noqa: E402

MAX_SEC = 100.0
MIN_SEC = 4.0


def snap_bounds(start, end, utts):
    """Snap a proposed [start, end] to the nearest real utterance edges."""
    starts = [u["start"] for u in utts]
    ends = [u["end"] for u in utts]
    return (
        min(starts, key=lambda s: abs(s - start)),
        min(ends, key=lambda e: abs(e - end)),
    )


def clamp_to_window(segments, win_start, win_end, tol=30.0):
    """Drop segments whose bounds fall outside the window the LLM was actually shown.

    Gemini regularly emits timestamps beyond its window. Unclamped, snap_bounds matches them
    against the whole recording, so a hallucinated time lands somewhere arbitrary and then
    blocks legitimate segments there under the no-overlap rule — in one run this cost a
    29-minute hole. `tol` allows a small straddle at the edges.
    """
    out = []
    for seg in segments:
        if seg["end"] <= seg["start"]:
            continue
        if seg["start"] < win_start - tol or seg["end"] > win_end + tol:
            continue
        out.append(seg)
    return out


def _turns_in(turns, start, end):
    return [t for t in turns if t["start"] >= start - 1e-6 and t["end"] <= end + 1e-6]


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def attach_real_turns(start, end, llm_turns, utts):
    """Build a segment's turns from the REAL utterances in [start, end].

    The LLM's timestamps are only advisory — being off by tenths of a second used to push a
    turn outside the snapped bounds and get an otherwise good dialogue discarded. Whisper's
    utterances are ground truth for timing and text; the LLM contributes only `kind` and
    `reference`, matched to each utterance by greatest temporal overlap. An utterance the LLM
    said nothing about defaults to smalltalk, which is the safe direction: no <RAG>.
    """
    turns = []
    for u in utts:
        if u["start"] < start - 1e-6 or u["end"] > end + 1e-6:
            continue
        kind, reference = "smalltalk", None
        if u["speaker"] == "DANIELLE":
            best, best_ov = None, 0.0
            for t in llm_turns:
                if t["speaker"] != "DANIELLE":
                    continue
                ov = _overlap(u["start"], u["end"], t["start"], t["end"])
                if ov > best_ov:
                    best, best_ov = t, ov
            if best is not None:
                kind = best.get("kind", "smalltalk")
                reference = best.get("reference") if kind != "smalltalk" else None
        turns.append({"speaker": u["speaker"], "start": u["start"], "end": u["end"],
                      "kind": kind, "reference": reference, "text": u["text"]})
    return turns


def trim_to_speaker_bounds(turns):
    """Trim to the first JOSHUA turn and the last DANIELLE turn.

    A unit must open on a question and close on her answer. Trimming rather than dropping
    keeps the dialogue when the LLM merely over-reached at one end.
    """
    first = next((i for i, t in enumerate(turns) if t["speaker"] == "JOSHUA"), None)
    last = next((i for i in range(len(turns) - 1, -1, -1)
                 if turns[i]["speaker"] == "DANIELLE"), None)
    if first is None or last is None or last <= first:
        return []
    return turns[first:last + 1]


def split_long(segment, utts, max_sec=MAX_SEC):
    """Split an over-long segment at the internal JOSHUA turn nearest its midpoint.

    Each half keeps only the turns inside it, with their own kinds and references.
    Recurses so a very long segment yields as many pieces as needed. A segment with no
    internal JOSHUA turn cannot be split and is returned as-is for the validator to drop.
    """
    start, end = segment["start"], segment["end"]
    if end - start <= max_sec:
        return [segment]

    mid = (start + end) / 2.0
    candidates = [
        t for t in segment["turns"]
        if t["speaker"] == "JOSHUA" and start < t["start"] < end
    ]
    if not candidates:
        return [segment]
    cut = min(candidates, key=lambda t: abs(t["start"] - mid))["start"]
    if cut <= start or cut >= end:
        return [segment]

    left = {**segment, "id": segment["id"] + "a", "start": start, "end": cut,
            "turns": _turns_in(segment["turns"], start, cut)}
    right = {**segment, "id": segment["id"] + "b", "start": cut, "end": end,
             "turns": _turns_in(segment["turns"], cut, end)}
    return split_long(left, utts, max_sec) + split_long(right, utts, max_sec)


def _check_one(seg, min_sec, max_sec):
    """Return a problem string, or None if the segment is usable."""
    dur = seg["end"] - seg["start"]
    if dur < min_sec:
        return f"{seg['id']}: too short ({dur:.1f}s < {min_sec}s)"
    if dur > max_sec:
        return f"{seg['id']}: too long ({dur:.1f}s > {max_sec}s) and unsplittable"
    if not any(t["speaker"] == "DANIELLE" for t in seg["turns"]):
        return f"{seg['id']}: no DANIELLE turn"
    if not seg["turns"] or seg["turns"][0]["speaker"] != "JOSHUA":
        return f"{seg['id']}: does not start on a JOSHUA turn"
    if seg["turns"][-1]["speaker"] != "DANIELLE":
        return f"{seg['id']}: does not end on a DANIELLE turn"
    return None


def validate_segments(segments, utts, max_sec=MAX_SEC, min_sec=MIN_SEC):
    """Snap, split, order and filter. Returns (kept, problems).

    Audio claimed by no segment is fine and is not reported as a problem — retakes and
    false starts are expected. The caller reports the unclaimed total separately.
    """
    problems = []
    snapped = []
    for seg in segments:
        s, e = snap_bounds(seg["start"], seg["end"], utts)
        if e <= s:
            problems.append(f"{seg['id']}: empty after snapping")
            continue
        turns = trim_to_speaker_bounds(attach_real_turns(s, e, seg["turns"], utts))
        if not turns:
            problems.append(f"{seg['id']}: no JOSHUA-question -> DANIELLE-answer span inside")
            continue
        # the trim may have moved the edges in; follow them
        snapped.append({**seg, "start": turns[0]["start"], "end": turns[-1]["end"],
                        "turns": turns})

    snapped.sort(key=lambda s: s["start"])

    kept = []
    last_end = float("-inf")
    for seg in snapped:
        if seg["start"] < last_end - 1e-6:
            problems.append(f"{seg['id']}: overlaps the previous segment; dropped")
            continue
        for piece in split_long(seg, utts, max_sec):
            problem = _check_one(piece, min_sec, max_sec)
            if problem:
                problems.append(problem)
                continue
            kept.append(piece)
            last_end = piece["end"]
    return kept, problems


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "turns": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "speaker": {"type": "STRING"},
                                "start": {"type": "NUMBER"},
                                "end": {"type": "NUMBER"},
                                "kind": {"type": "STRING"},
                                "reference": {"type": "STRING"},
                                "text": {"type": "STRING"},
                            },
                            "required": ["speaker", "start", "end", "kind"],
                        },
                    },
                },
                "required": ["start", "end", "turns"],
            },
        },
    },
    "required": ["segments"],
}

PROMPT = """You are given a timestamped transcript of a real recorded conversation between
JOSHUA (asks questions) and DANIELLE (a warm, knowledgeable assistant who answers them).
Each line begins with [SECONDS] — the absolute start time of that turn, in seconds from the
beginning of the recording. Every start/end you return MUST be copied verbatim from these
bracketed numbers. Do not convert, rescale, or restart them at zero: a unit beginning at the
line "[1543.2] JOSHUA: ..." has start 1543.2, not 0 and not 25.7.

Split the transcript into DIALOGUE UNITS. A unit is a self-contained stretch of conversation:
it begins with a JOSHUA turn and ends with a DANIELLE turn. Prefer units that contain a MODE
CHANGE — casual chat that turns into a factual question, or a factual answer that returns to
casual chat — because those teach the model to switch. Do not split a follow-up away from the
answer it follows up on. Units must not overlap. Aim for 20-90 seconds. Leave retakes, false
starts and dead air out of every unit.

COVERAGE MATTERS: this transcript is roughly 12 minutes of near-continuous conversation, so
expect to emit on the order of 10-25 units — one per topic or sub-topic. Work through the
WHOLE transcript from the first timestamp to the last; do not stop after the first few. Every
stretch of real conversation should belong to some unit. A long thread on one topic should be
split into several units at natural question boundaries rather than emitted as one long unit
or skipped.

Then label EACH of DANIELLE's turns:
  "grounded"  - she answers a question using specific external facts.
  "decline"   - she says she does not know / cannot answer, instead of guessing.
  "smalltalk" - casual conversation needing no external facts.
Label every JOSHUA turn "smalltalk" with no reference.

IMPORTANT: many of DANIELLE's turns are QUESTIONS she asks JOSHUA to keep the conversation
going ("Is there one you've heard of recently?"), or warm acknowledgements of what he just
said. Those are "smalltalk", never "grounded" — she is asking or reacting, not answering from
facts. Only label a turn "grounded" when she is stating specific external information.
A turn that both states facts AND ends with a follow-up question is still "grounded".

For each "grounded" turn write a "reference": a 2-5 sentence encyclopedic passage, in neutral
reference-work prose, that CONTAINS the facts she states. Include some surrounding detail she
did NOT mention, so the passage reads like a retrieved document rather than a restatement of
her answer. Never invent facts that contradict her.

For each "decline" turn write a "reference" about a RELATED BUT DIFFERENT topic — the passage
a retrieval system would have wrongly surfaced. It must not answer the question.

For "smalltalk" turns omit "reference".

Copy each turn's spoken words into "text". Every start and end must be one of the bracketed
second-values shown above, copied exactly. The first unit in this excerpt should begin at or
near {win_start:.1f} and the last should end at or near {win_end:.1f}.

TRANSCRIPT:
{transcript}
"""


def call_gemini(prompt: str, model: str, key: str, schema=SCHEMA, retries: int = 4):
    """Same stdlib REST pattern as scripts/gen_recording_dialogues.py."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json",
                             "responseSchema": schema},
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            return json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
        except urllib.error.HTTPError as e:
            # 4xx other than rate-limiting will never succeed on retry (a retired model
            # 404s even though it is still listed by the models endpoint) — fail fast.
            detail = e.read()[:300].decode(errors="replace")
            if 400 <= e.code < 500 and e.code != 429:
                raise RuntimeError(f"gemini {e.code} for model '{model}': {detail}") from e
            last = e
            print(f"  [retry {attempt+1}/{retries}] HTTP {e.code}: {detail}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last = e
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini failed: {last}")


def windows(utts, window_sec=720.0, overlap_sec=60.0):
    """Slice utterances into overlapping time windows so no dialogue is cut by a window edge."""
    if not utts:
        return []
    end = max(u["end"] for u in utts)
    step = window_sec - overlap_sec
    out = []
    start = 0.0
    while start < end:
        stop = start + window_sec
        chunk = [u for u in utts if u["start"] >= start and u["start"] < stop]
        if chunk:
            out.append(chunk)
        start += step
    return out


def dedupe(segments, tol=1.0):
    """Drop segments repeated across a window overlap (same start within `tol` seconds)."""
    out = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        if out and abs(seg["start"] - out[-1]["start"]) <= tol:
            continue
        out.append(seg)
    return out


def turn_mix(segments):
    """Count HER turns by kind. retrieval_share sets RAG_TOKEN_WEIGHT (see the spec)."""
    counts = {"grounded": 0, "decline": 0, "smalltalk": 0}
    for seg in segments:
        for t in seg["turns"]:
            if t["speaker"] != "DANIELLE":
                continue
            counts[t["kind"]] = counts.get(t["kind"], 0) + 1
    total = sum(counts.values())
    share = (counts["grounded"] + counts["decline"]) / total if total else 0.0
    return {**counts, "total": total, "retrieval_share": round(share, 4)}


def normalize(raw_segments, window_index):
    """Gemini output -> our segment dicts, with ids and reference defaults.

    Enforces the invariants the prompt asks for but cannot guarantee: only DANIELLE turns
    carry a kind other than smalltalk, only grounded/decline keep a reference, and an
    unrecognised kind degrades to smalltalk (no <RAG>) rather than poisoning the data.
    """
    out = []
    for i, seg in enumerate(raw_segments):
        turns = []
        for t in seg.get("turns", []):
            kind = t.get("kind", "smalltalk")
            if kind not in ("grounded", "decline", "smalltalk"):
                kind = "smalltalk"
            speaker = "DANIELLE" if t.get("speaker", "").upper().startswith("D") else "JOSHUA"
            if speaker == "JOSHUA":
                kind = "smalltalk"
            reference = t.get("reference") or None
            if kind == "smalltalk":
                reference = None
            turns.append({"speaker": speaker, "start": float(t["start"]),
                          "end": float(t["end"]), "kind": kind, "reference": reference,
                          "text": t.get("text", "")})
        out.append({"id": f"ret-{window_index:02d}-{i:03d}",
                    "start": float(seg["start"]), "end": float(seg["end"]), "turns": turns})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch0", required=True, help="channel-0 (Danielle) alignments json")
    ap.add_argument("--ch1", required=True, help="channel-1 (Joshua) alignments json")
    ap.add_argument("--out", default="replay/retrieval_segments.jsonl")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--window-sec", type=float, default=720.0)
    ap.add_argument("--overlap-sec", type=float, default=60.0)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (use a rotated key; never pass it on the command line)")

    utts = merge_utterances(load_alignments(args.ch0), load_alignments(args.ch1))
    if not utts:
        raise SystemExit("no utterances — are both transcripts non-empty?")
    print(f"{len(utts)} utterances, {max(u['end'] for u in utts) / 60:.1f} min")

    raw = []
    wins = windows(utts, args.window_sec, args.overlap_sec)
    for wi, chunk in enumerate(wins):
        win_start, win_end = chunk[0]["start"], chunk[-1]["end"]
        print(f"  window {wi + 1}/{len(wins)}  [{win_start / 60:.1f}-{win_end / 60:.1f} min]",
              flush=True)
        prompt = PROMPT.format(transcript=format_transcript(chunk),
                               win_start=win_start, win_end=win_end)
        resp = call_gemini(prompt, args.model, key)
        proposed = normalize(resp.get("segments", []), wi)
        # Clamp and snap against THIS window only — an out-of-window timestamp must not be
        # matched against utterances elsewhere in the recording.
        inside = clamp_to_window(proposed, win_start, win_end)
        if len(inside) < len(proposed):
            print(f"    {len(proposed) - len(inside)} of {len(proposed)} segments fell "
                  f"outside the window; dropped", flush=True)
        raw += [{**s, "_utts": chunk} for s in inside]

    # snap each segment against its own window's utterances, then validate globally
    prepared = []
    for seg in raw:
        chunk = seg.pop("_utts")
        s, e = snap_bounds(seg["start"], seg["end"], chunk)
        prepared.append({**seg, "start": s, "end": e})

    kept, problems = validate_segments(dedupe(prepared), utts)

    claimed = sum(s["end"] - s["start"] for s in kept)
    total = max(u["end"] for u in utts)
    mix = turn_mix(kept)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for seg in kept:
            f.write(json.dumps(seg) + "\n")

    for p in problems:
        print(f"  DROP {p}")
    print(f"\nwrote {args.out}: {len(kept)} segments, {len(problems)} dropped")
    print(f"kept {claimed / 60:.1f} min of {total / 60:.1f} min "
          f"({(total - claimed) / 60:.1f} min unclaimed)")
    print(f"her turns: {mix}")
    print(f"-> retrieval_share {mix['retrieval_share']:.2f}: "
          f"use RAG_TOKEN_WEIGHT 15 if smalltalk+decline >= 30% of her turns, else 8-10")


if __name__ == "__main__":
    main()

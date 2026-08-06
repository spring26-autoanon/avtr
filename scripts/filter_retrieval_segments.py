#!/usr/bin/env python3
"""Stage 3 — topic screen over the generated reference passages.

References here are reverse-engineered from what Danielle actually said, so they should
almost always be on topic; this exists solely to catch Gemini writing a passage about the
WRONG SUBJECT. It is not a completeness or accuracy grader.

That distinction was learned the hard way. A first version asked "is every factual claim
supported?" and rejected 34% of grounded turns — including a Diwali passage that covered
every fact she stated. It was failing on Whisper's garbled proper nouns ("dais" for diyas,
"gurupra" for Gurpurab) and on trivial paraphrase. The prompt now screens for subject
mismatch and explicitly discounts transcription noise.

A segment with an off-topic reference is DROPPED WHOLE. An earlier version demoted the turn
to smalltalk to save the audio, which is backwards: it leaves her giving a specific factual
answer with no <RAG>, training the confabulation this pipeline exists to prevent.

Only `grounded` turns are judged — a `decline` reference is deliberately unrelated and
`smalltalk` has none.

    export GEMINI_API_KEY=...
    python3 scripts/filter_retrieval_segments.py \
        --in replay/retrieval_segments.jsonl \
        --out replay/retrieval_segments.filtered.jsonl \
        --dropped replay/retrieval_dropped.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_retrieval_audio import call_gemini  # noqa: E402

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"supported": {"type": "BOOLEAN"}},
    "required": ["supported"],
}

JUDGE_PROMPT = """A retrieval system surfaced this PASSAGE, and the assistant then gave this
SPOKEN ANSWER. Decide whether the passage is a plausible source for the answer.

PASSAGE:
{reference}

SPOKEN ANSWER:
{answer}

Answer {{"supported": false}} ONLY if the passage is about a DIFFERENT SUBJECT than the
answer — i.e. it could not have been what she was reading from.

Answer {{"supported": true}} in every other case, including when:
  - the answer mentions details the passage omits, or vice versa;
  - the wording differs, or she paraphrases loosely;
  - proper nouns are garbled. The answer is a rough speech-to-text transcript, so
    "dais" for diyas, "gurupra" for Gurpurab, "young capur" for Yom Kippur are
    TRANSCRIPTION NOISE, not factual errors. Judge the subject matter, not the spelling;
  - the answer is a brief conversational follow-up on the passage's topic.

You are screening for a retrieval system that fetched the wrong document — not grading
completeness or accuracy.
"""


def judgeable_turns(segments):
    """[(segment_id, turn_index)] for grounded turns that have a reference."""
    out = []
    for seg in segments:
        for i, t in enumerate(seg["turns"]):
            if t["speaker"] == "DANIELLE" and t["kind"] == "grounded" and t.get("reference"):
                out.append((seg["id"], i))
    return out


def apply_verdicts(segments, verdicts):
    """Drop segments containing an off-topic reference. Returns (kept, dropped_records).

    An earlier version DEMOTED the offending turn to smalltalk to keep the audio. That is
    not the safe direction: it leaves her giving a specific factual answer with no <RAG> and
    no conditioning, which trains exactly the confabulation this pipeline exists to prevent.
    Losing the clip teaches nothing wrong; keeping a mislabelled one teaches something wrong.

    A turn with no verdict (its judge call errored) keeps its segment — dropping on an API
    failure would quietly drain retrieval out of the dataset.
    """
    kept, dropped = [], []
    for seg in segments:
        bad = [i for i in range(len(seg["turns"])) if verdicts.get(f"{seg['id']}#{i}") is False]
        if bad:
            dropped += [{"segment": seg["id"], "turn": i, "reason": "reference not on topic"}
                        for i in bad]
            continue
        kept.append({**seg, "turns": [dict(t) for t in seg["turns"]]})
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/retrieval_segments.jsonl")
    ap.add_argument("--out", default="replay/retrieval_segments.filtered.jsonl")
    ap.add_argument("--dropped", default="replay/retrieval_dropped.jsonl")
    ap.add_argument("--model", default="gemini-flash-latest")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (use a rotated key)")

    segments = [json.loads(line) for line in open(args.inp) if line.strip()]
    by_id = {s["id"]: s for s in segments}

    verdicts = {}
    targets = judgeable_turns(segments)
    for n, (sid, ti) in enumerate(targets, 1):
        turn = by_id[sid]["turns"][ti]
        prompt = JUDGE_PROMPT.format(reference=turn["reference"], answer=turn.get("text", ""))
        try:
            resp = call_gemini(prompt, args.model, key, schema=VERDICT_SCHEMA)
        except RuntimeError as e:
            print(f"  WARN {sid}#{ti} judge failed, keeping as-is: {e}")
            continue
        verdicts[f"{sid}#{ti}"] = bool(resp.get("supported", True))
        if n % 20 == 0 or n == len(targets):
            print(f"  judged {n}/{len(targets)}")

    kept, dropped = apply_verdicts(segments, verdicts)

    with open(args.out, "w") as f:
        for seg in kept:
            f.write(json.dumps(seg) + "\n")
    with open(args.dropped, "w") as f:
        for rec in dropped:
            f.write(json.dumps(rec) + "\n")

    pct = 100 * len(dropped) / max(len(targets), 1)
    print(f"wrote {args.out}: {len(kept)}/{len(segments)} segments kept; "
          f"{len(dropped)} off-topic references -> {args.dropped}")
    if pct > 10:
        print(f"WARNING: {pct:.0f}% of grounded turns rejected. This screen should fire "
              f"rarely — a high rate means the judge is grading completeness rather than "
              f"subject match. Inspect the dropped turns before training on this.")


if __name__ == "__main__":
    main()

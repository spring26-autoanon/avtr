#!/usr/bin/env python3
"""Stage 3 — faithfulness filter over the generated reference passages.

References here are reverse-engineered from what Danielle actually said, so they should
almost always be faithful; this catches Gemini writing a passage about the wrong topic.

An unfaithful grounded turn is DEMOTED to smalltalk (no <RAG>, no reference) rather than
deleting the clip: the audio is still good natural conversation in her voice, and keeping
it preserves the surrounding turn-taking. Only `grounded` turns are judged — a `decline`
reference is deliberately unrelated and `smalltalk` has none.

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

JUDGE_PROMPT = """Here is a passage and a spoken answer.

PASSAGE:
{reference}

SPOKEN ANSWER:
{answer}

Is every factual claim in the spoken answer supported by the passage? Answer with
{{"supported": true}} or {{"supported": false}}. Casual phrasing, greetings and
follow-up questions in the answer do not count as factual claims.
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
    """Demote turns whose verdict is False. Returns (kept_segments, dropped_records).

    A turn with no verdict (its judge call errored) is left alone — silently demoting on
    an API failure would quietly drain the retrieval signal out of the dataset.
    """
    kept, dropped = [], []
    for seg in segments:
        turns = [dict(t) for t in seg["turns"]]
        for i, t in enumerate(turns):
            key = f"{seg['id']}#{i}"
            if verdicts.get(key) is False:
                t["kind"] = "smalltalk"
                t["reference"] = None
                dropped.append({"segment": seg["id"], "turn": i, "reason": "unfaithful"})
        kept.append({**seg, "turns": turns})
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

    print(f"wrote {args.out}: {len(kept)} segments, {len(dropped)} turns demoted -> {args.dropped}")


if __name__ == "__main__":
    main()

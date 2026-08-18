#!/usr/bin/env python3
"""Path A Stage 2/3 — produce a human recording script from the replay dialogues.

Danielle records ONLY the assistant answers (her turns); user questions stay as generic TTS.
Writes:
  replay/recording_script.txt      — numbered answers for her to read aloud
  replay/recording_manifest.jsonl  — {n, id, kind, text} to map each recording back to its example

Reads replay/dialogue.filtered.jsonl. Answers are read in order; we later auto-segment/align a
continuous recording using the known text (forced alignment), so she doesn't need one file per
answer — just a clear pause between numbers.

    python scripts/make_recording_script.py
"""
import argparse
import json
import os

HEADER = """DANIELLE — VOICE RECORDING SCRIPT ({n} lines)
====================================================================
HOW TO RECORD
- Same mic / quiet room as your dialogue recordings, if possible.
- Read each numbered line naturally, the way you'd say it out loud to a person.
- Leave a clear ~1 second pause between numbers (helps us split the audio).
- You do NOT need to say the numbers, and you do NOT need one file per line —
  one continuous recording (or a few, in order) is fine. We align by the text.
- If you flub a line, just pause and read that whole line again; we'll keep the good take.
- These are short assistant-style answers. A few (the "I don't have that info..." ones)
  are meant to sound like politely not knowing — read them that way.
====================================================================

"""


def ans_of(rec: dict) -> str:
    return next(t["text"] for t in rec["turns"] if t["speaker"] == "assistant").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/dialogue.filtered.jsonl")
    ap.add_argument("--script", default="replay/recording_script.txt")
    ap.add_argument("--manifest", default="replay/recording_manifest.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.script)), exist_ok=True)

    with open(args.script, "w") as s, open(args.manifest, "w") as m:
        s.write(HEADER.format(n=len(rows)))
        for i, rec in enumerate(rows, 1):
            text = ans_of(rec)
            s.write(f"{i:04d}.  {text}\n\n")
            m.write(json.dumps({"n": i, "id": rec["id"], "kind": rec["kind"], "text": text},
                               ensure_ascii=False) + "\n")

    print(f"wrote {args.script} and {args.manifest}  ({len(rows)} answers)")


if __name__ == "__main__":
    main()

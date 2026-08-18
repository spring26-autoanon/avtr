#!/usr/bin/env python3
"""Path A Stage 3 — natural multi-turn recording script for a 2-person session.

Turns each QA item into a natural, SPOKEN, multi-turn dialogue between:
  USER      — a friendly person asking (the partner in the recording), on channel 1
  DANIELLE  — the warm assistant answering (SPEAKER_MAIN), on channel 0

Crucially the dialogues are NON-TERMINAL: after Danielle's grounded answer she adds a natural
follow-up question / continuation and the conversation keeps going — this fixes the empty-buffer
stalls (the old 2-turn "answer then stop" data taught her to stop after answering). Her factual
turn stays faithful to the reference so we can still precompute the reference tensor + place ⟨ret⟩.

Outputs:
  replay/recording_dialogues.txt   — human-readable script the two speakers read from
  replay/recording_dialogues.jsonl — structured {id,kind,reference,turns,grounded_turn} for
                                      post-recording processing (align → pair → precompute → ⟨ret⟩)

Gemini REST (stdlib). Set GEMINI_API_KEY. Resumable/de-duped by id.

    python scripts/gen_recording_dialogues.py --in replay/qa.jsonl --limit 120
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# Randomized conversation shapes so the dataset isn't templated on one follow-up pattern.
SHAPES = [
    "Very brief: Danielle's answer plus a single short follow-up line, then it ends naturally. 3-4 turns.",
    "Short: the answer, then one quick follow-up exchange, then wrap. About 4-5 turns.",
    "Medium: Danielle answers, then a couple of natural follow-up exchanges; mix who asks. 5-6 turns.",
    "The USER is curious and asks TWO or THREE related follow-up questions after the first answer; "
    "Danielle answers each naturally. 6-8 turns.",
    "Danielle drives: after answering she asks the USER a follow-up question and they chat about it "
    "for a couple of turns. 5-7 turns.",
    "A longer, wandering chat: the answer leads into a related tangent with several back-and-forth "
    "turns. 7-9 turns.",
]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "turns": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "speaker": {"type": "STRING"},   # "USER" or "DANIELLE"
                    "text": {"type": "STRING"},
                },
                "required": ["speaker", "text"],
            },
        },
        "grounded_turn": {"type": "INTEGER"},        # index of Danielle's factual answer turn
    },
    "required": ["turns", "grounded_turn"],
}

_COMMON = (
    "Script a NATURAL, SPOKEN conversation for a voice-assistant recording. Two speakers: USER "
    "(a friendly person) and DANIELLE (a warm, knowledgeable assistant). Sound like real people "
    "talking out loud — contractions, easy phrasing, no markdown, no lists. Alternate USER/DANIELLE, "
    "starting with USER. IMPORTANT: the conversation must NOT dead-end on Danielle's answer — she "
    "keeps it going with follow-ups. VARY the rhythm so no two dialogues feel templated. "
    "SHAPE FOR THIS ONE: {shape} "
)


def build_prompt(kind: str, question: str, reference: str, answer: str, shape: str) -> str:
    if kind == "grounded":
        body = (
            f"USER asks (in their own natural words): \"{question}\". DANIELLE answers using ONLY "
            f"facts from this REFERENCE, staying faithful to it: \"{reference}\". Then she keeps the "
            "conversation going (a follow-up question to USER or a related aside), USER replies, and "
            "Danielle responds warmly. Set grounded_turn to the index (0-based) of Danielle's turn "
            "that contains the factual answer."
        )
    elif kind == "decline":
        body = (
            f"USER asks: \"{question}\". Danielle does NOT have that information (her only note is the "
            f"unrelated reference: \"{reference}\"), so she gracefully says she can't answer that one "
            "rather than guessing, then pivots and keeps the conversation going naturally. Set "
            "grounded_turn to the index of Danielle's declining turn."
        )
    else:  # smalltalk
        body = (
            f"A casual exchange starting from: \"{question}\". No facts needed, just warm natural "
            "small talk that flows for a few turns. Set grounded_turn to -1."
        )
    return _COMMON.format(shape=shape) + body


def call_gemini(prompt: str, model: str, key: str, retries: int = 4):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9, "responseMimeType": "application/json",
                             "responseSchema": SCHEMA},
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            return json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last = e
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini failed: {last}")


def load_done(path):
    ids = set()
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l:
                try:
                    ids.add(json.loads(l)["id"])
                except Exception:
                    pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/qa.jsonl")
    ap.add_argument("--out-jsonl", default="replay/recording_dialogues.jsonl")
    ap.add_argument("--out-script", default="replay/recording_dialogues.txt")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (source .env).")

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = load_done(args.out_jsonl)

    jf = open(args.out_jsonl, "a")
    n = 0
    for rec in rows:
        if rec["id"] in done:
            continue
        try:
            d = call_gemini(build_prompt(rec["kind"], rec["question"], rec.get("reference", ""),
                                         rec.get("answer", ""), random.choice(SHAPES)), args.model, key)
        except RuntimeError as e:
            print(f"  skip {rec['id']} ({e})", file=sys.stderr)
            continue
        out = {"id": rec["id"], "kind": rec["kind"], "reference": rec.get("reference", ""),
               "turns": d["turns"], "grounded_turn": d.get("grounded_turn", -1)}
        jf.write(json.dumps(out, ensure_ascii=False) + "\n")
        jf.flush()
        n += 1
        if n % 10 == 0:
            print(f"  {n} dialogues", file=sys.stderr)
    jf.close()

    # rebuild the readable script from the full jsonl
    dialogues = [json.loads(l) for l in open(args.out_jsonl) if l.strip()]
    with open(args.out_script, "w") as s:
        s.write("DANIELLE — RAG CONVERSATION RECORDING SCRIPT ({} dialogues)\n".format(len(dialogues)))
        s.write("=" * 68 + "\n")
        s.write("Two people record this together: a PARTNER reads the USER lines, Danielle reads\n"
                "the DANIELLE lines. Talk naturally, like a real conversation — it's fine to\n"
                "improvise wording. Leave a short pause between each numbered dialogue.\n")
        s.write("=" * 68 + "\n\n")
        for i, d in enumerate(dialogues, 1):
            s.write(f"--- {i:04d}  [{d['kind']}] ---\n")
            for t in d["turns"]:
                s.write(f"  {t['speaker']}: {t['text']}\n")
            s.write("\n")
    print(f"wrote {args.out_jsonl} ({len(dialogues)} total) and {args.out_script}")


if __name__ == "__main__":
    main()

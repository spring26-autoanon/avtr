#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 4 Step 1: generate grounded QA triples for RAG replay data.

Emits diverse {question, reference, answer} triples that teach the domain-agnostic behavior
"use the reference": the assistant answers a spoken question STRICTLY from a short retrieved
reference (the "Reference: ..." text the serve-time Gemini would surface). Three kinds:

  grounded  — factual question; ~40-60 word reference; answer faithful to the reference only.
  decline   — reference is IRRELEVANT to the question; answer declines / does not hallucinate.
  smalltalk — casual chit-chat; NO reference (empty); answer is casual, no retrieval.

Output: JSONL, one triple per line:
  {"id","kind","domain","question","reference","answer"}

The reference here is TEXT. Task 5 later POSTs it to the :8001 ARC encoder to get the
`reference_with_time` tensor the trainer now consumes (verified via the data plumbing).

Uses Gemini (REST, stdlib only — no deps). Set GEMINI_API_KEY in the env (use a freshly
ROTATED key — do not paste keys on the command line). Model + mix are configurable.

    export GEMINI_API_KEY=...          # rotated key
    python scripts/gen_replay_qa.py --n 300 --out replay/qa.jsonl

Resumable: appends and de-dupes against questions already in --out.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

DOMAINS = [
    "world geography", "space and astronomy", "human biology and health", "world history",
    "cooking and food science", "everyday physics", "computers and the internet", "music and instruments",
    "animals and nature", "art and painters", "world literature", "basic economics and money",
    "sports and games", "weather and climate", "chemistry in daily life", "famous inventions",
    "languages and words", "mythology and folklore", "plants and gardening", "the human brain",
    "transportation and vehicles", "architecture and landmarks", "film and television", "oceans and marine life",
    "mathematics in everyday life", "first aid and safety", "world religions and traditions", "energy and electricity",
    "psychology basics", "the solar system", "ancient civilizations", "modern technology and gadgets",
]

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "question": {"type": "STRING"},
            "reference": {"type": "STRING"},
            "answer": {"type": "STRING"},
        },
        "required": ["question", "reference", "answer"],
    },
}

_COMMON = (
    "You are generating spoken-conversation training data for a friendly voice assistant. "
    "Questions must sound natural said aloud. Answers must be conversational and spoken-style, "
    "1-3 short sentences, no markdown, no lists, no citations. "
)


def build_prompt(kind: str, domain: str, k: int) -> str:
    if kind == "grounded":
        return (
            _COMMON
            + f"Produce {k} diverse, self-contained factual QA items in the domain of '{domain}'. "
            "For each: a natural spoken 'question'; a concise 'reference' of about 40-60 words giving "
            "the facts needed to answer (the snippet a retrieval system would surface); and an 'answer' "
            "that is STRICTLY faithful to the reference and states nothing beyond it. Vary difficulty, "
            "phrasing, and subtopics so items don't overlap. Return a JSON array of "
            "{question, reference, answer}."
        )
    if kind == "decline":
        return (
            _COMMON
            + f"Produce {k} items about '{domain}' where the 'reference' (about 40-60 words) is a real "
            "factual snippet on a DIFFERENT, unrelated topic than the 'question', so it does NOT contain "
            "the answer. The 'answer' must gracefully say it doesn't have that information rather than "
            "guessing or hallucinating (e.g., admits the reference doesn't cover it). Keep answers warm "
            "and brief. Return a JSON array of {question, reference, answer}."
        )
    if kind == "smalltalk":
        return (
            _COMMON
            + f"Produce {k} casual small-talk exchanges loosely themed around '{domain}' that need NO "
            "external facts (greetings, opinions, feelings, banter, encouragement). Set 'reference' to an "
            "empty string. The 'answer' is a casual, warm reply with no factual claims that would need "
            "retrieval. Return a JSON array of {question, reference, answer}."
        )
    raise ValueError(kind)


def call_gemini(prompt: str, model: str, key: str, temperature: float, retries: int = 4) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
        },
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last = e
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode()[:300]
                except Exception:
                    pass
            print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__}: {e} {detail}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after {retries} tries: {last}")


def load_existing_questions(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)["question"].strip().lower())
                    except Exception:
                        pass
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="total triples to generate (target)")
    ap.add_argument("--out", default="replay/qa.jsonl")
    ap.add_argument("--model", default="gemini-flash-latest",
                    help="Gemini model id (e.g. gemini-flash-latest, gemini-3.5-flash-lite)")
    ap.add_argument("--per-call", type=int, default=8, help="triples requested per API call")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--grounded", type=float, default=0.7, help="fraction grounded")
    ap.add_argument("--decline", type=float, default=0.15, help="fraction irrelevant-reference/decline")
    ap.add_argument("--smalltalk", type=float, default=0.15, help="fraction no-retrieval smalltalk")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set. export a ROTATED key first (do not put it on the command line).")

    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seen = load_existing_questions(args.out)
    start = len(seen)
    kinds = (["grounded"] * int(args.grounded * 100)
             + ["decline"] * int(args.decline * 100)
             + ["smalltalk"] * int(args.smalltalk * 100)) or ["grounded"]

    written = 0
    with open(args.out, "a") as out:
        while start + written < args.n:
            kind = random.choice(kinds)
            domain = random.choice(DOMAINS)
            k = min(args.per_call, args.n - start - written)
            try:
                items = call_gemini(build_prompt(kind, domain, k), args.model, key, args.temperature)
            except RuntimeError as e:
                print(f"skipping batch ({e})", file=sys.stderr)
                continue
            for it in items:
                q = str(it.get("question", "")).strip()
                a = str(it.get("answer", "")).strip()
                ref = str(it.get("reference", "")).strip()
                if not q or not a:
                    continue
                if q.lower() in seen:
                    continue
                if kind == "smalltalk":
                    ref = ""
                elif not ref:
                    continue  # grounded/decline must have a reference
                seen.add(q.lower())
                rec = {"id": f"{kind}-{start + written:05d}", "kind": kind,
                       "domain": domain, "question": q, "reference": ref, "answer": a}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if start + written >= args.n:
                    break
            print(f"  {start + written}/{args.n}  (+{len(items)} {kind}/{domain})", file=sys.stderr)

    print(f"done: {written} new, {start + written} total in {args.out}")


if __name__ == "__main__":
    main()

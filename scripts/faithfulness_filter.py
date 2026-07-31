#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 4.3: faithfulness filter for replay dialogues.

Safety net: for each GROUNDED dialogue, an LLM judges whether the assistant answer is fully
supported by its reference — i.e. makes no factual claim that's absent from or contradicts the
reference. Unsupported ones are dropped so the model never learns to "retrieve then hallucinate."
`decline` and `smalltalk` pass through unchanged (decline is constructed to refuse; smalltalk has
no reference).

Batched Gemini calls (stdlib only). Set GEMINI_API_KEY (rotated). Writes a filtered JSONL and
a dropped-items report.

    python scripts/faithfulness_filter.py --in replay/dialogue.jsonl --out replay/dialogue.filtered.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

JUDGE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "supported": {"type": "BOOLEAN"},
            "reason": {"type": "STRING"},
        },
        "required": ["index", "supported", "reason"],
    },
}

JUDGE_PROMPT = (
    "You are a strict faithfulness judge for a retrieval assistant. For each item you are given a "
    "REFERENCE and an ANSWER. Decide whether the ANSWER is FULLY supported by the REFERENCE: it may "
    "only state facts present in (or directly entailed by) the reference, and must not contradict it "
    "or add outside facts. Paraphrase and omission are fine; new specific claims are not. Return a "
    "JSON array with one object per item: {index, supported (boolean), reason (short)}.\n\nITEMS:\n"
)


def call_judge(items: list, model: str, key: str, retries: int = 4) -> list:
    payload = "\n".join(
        f"[{i}] REFERENCE: {it['reference']}\n    ANSWER: {it['answer']}" for i, it in enumerate(items)
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": JUDGE_PROMPT + payload}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                             "responseSchema": JUDGE_SCHEMA},
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            verdicts = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
            return {v["index"]: v for v in verdicts}
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last = e
            print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"judge call failed after {retries} tries: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/dialogue.jsonl")
    ap.add_argument("--out", default="replay/dialogue.filtered.jsonl")
    ap.add_argument("--report", default="replay/dropped.jsonl")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set. source .env with a rotated key first.")

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    grounded = [r for r in rows if r["kind"] == "grounded"]

    # Judge grounded in batches; map answer = last assistant turn's text.
    def ans(r):
        return next(t["text"] for t in r["turns"] if t["speaker"] == "assistant")

    verdicts = {}
    for i in range(0, len(grounded), args.batch):
        chunk = grounded[i:i + args.batch]
        items = [{"reference": r["reference"], "answer": ans(r)} for r in chunk]
        try:
            local = call_judge(items, args.model, key)
        except RuntimeError as e:
            print(f"  batch {i} failed, keeping all in it ({e})", file=sys.stderr)
            local = {j: {"supported": True, "reason": "judge-failed-keep"} for j in range(len(chunk))}
        for j, r in enumerate(chunk):
            verdicts[r["id"]] = local.get(j, {"supported": True, "reason": "missing-verdict-keep"})
        print(f"  judged {min(i + args.batch, len(grounded))}/{len(grounded)} grounded", file=sys.stderr)

    kept, dropped = [], []
    for r in rows:
        if r["kind"] != "grounded":
            kept.append(r)
            continue
        v = verdicts.get(r["id"], {"supported": True})
        if v.get("supported", True):
            kept.append(r)
        else:
            dropped.append({**r, "_drop_reason": v.get("reason", "")})

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.report, "w") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)}/{len(rows)}  (dropped {len(dropped)} unfaithful grounded)")
    print(f"  -> {args.out}   dropped report -> {args.report}")


if __name__ == "__main__":
    main()

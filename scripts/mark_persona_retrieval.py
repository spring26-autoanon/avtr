#!/usr/bin/env python3
"""persona -> retrieval: convert identity answers into grounded (retrieval) turns.

Decision of record (2026-08-07 night): the 127 identity answers in the persona recording
(`replay/persona_segments.filtered.jsonl`), originally guarded AGAINST retrieval marking (see
`tests/test_persona_guard.py`), are now converted TO retrieval turns. Live evidence showed the
retrieval trigger+grounding pipeline works excellently while weights-based identity keeps
failing. The loss mask makes this safe alongside the unmarked long-form copy: a marker-free
window never trains the <ret> logit (see the RAG-token loss mask notes), so mixing grounded
identity turns in with the rest of the persona data does not risk teaching a spurious retrieval
trigger on ordinary conversation.

For every DANIELLE turn with kind == "persona", this script asks Gemini to write a short
"Reference: ..." passage derived from HER OWN answer text (and the preceding question, for
context), then flips kind -> "grounded" and reference -> that passage. Every other turn
(JOSHUA turns, decline turns, already-grounded turns, smalltalk) passes through unchanged.

    export GEMINI_API_KEY=...
    python3 scripts/mark_persona_retrieval.py \
        --in  replay/persona_segments.filtered.jsonl \
        --out replay/persona_segments.retrieval.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DEFAULT = "gemini-flash-latest"

REFERENCE_PROMPT = """{question_part}Danielle's answer: "{answer}"

Write one line starting with "Reference: " followed by at most 50 words of third-person
factual statements about Danielle extracted from her answer above. Only facts she states; no
meta commentary; plain text.
"""


def build_reference_prompt(question_text: str, answer_text: str) -> str:
    question_part = f'Preceding question: "{question_text}"\n' if question_text else ""
    return REFERENCE_PROMPT.format(question_part=question_part, answer=answer_text)


def call_gemini_text(prompt: str, model: str, key: str, retries: int = 4) -> str:
    """Same stdlib REST pattern as scripts/segment_retrieval_audio.py:call_gemini, but for a
    plain-text response (no JSON schema) since we just want one "Reference: ..." line."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            return resp["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            # 4xx other than rate-limiting will never succeed on retry — fail fast.
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


def make_gemini_reference_fn(model: str, key: str):
    """Build a gen_reference(question_text, answer_text) -> str closure bound to Gemini."""
    def gen_reference(question_text: str, answer_text: str) -> str:
        prompt = build_reference_prompt(question_text, answer_text)
        return call_gemini_text(prompt, model, key)
    return gen_reference


def mark_persona(segments, gen_reference):
    """Pure transform: DANIELLE 'persona' turns -> 'grounded' with a self-derived reference.

    `gen_reference(question_text, answer_text) -> str` is injected so this needs no network
    access to test. Retries once per turn on failure; if both attempts raise, the turn is left
    as `persona` (unconverted) rather than crashing the whole run. Every other turn (JOSHUA
    turns, decline, already-grounded, smalltalk) is passed through unchanged.

    Returns (new_segments, stats) where stats counts converted / skipped / declines_untouched /
    grounded_untouched / references_generated.
    """
    stats = {
        "converted": 0,
        "skipped": 0,
        "declines_untouched": 0,
        "grounded_untouched": 0,
        "references_generated": 0,
    }
    out_segments = []
    for seg in segments:
        turns = seg.get("turns", [])
        new_turns = []
        for i, t in enumerate(turns):
            if t.get("speaker") != "DANIELLE" or t.get("kind") != "persona":
                if t.get("speaker") == "DANIELLE" and t.get("kind") == "decline":
                    stats["declines_untouched"] += 1
                elif t.get("speaker") == "DANIELLE" and t.get("kind") == "grounded":
                    stats["grounded_untouched"] += 1
                new_turns.append(t)
                continue

            question_text = turns[i - 1].get("text", "") if i > 0 else ""
            answer_text = t.get("text", "")

            reference = None
            for _attempt in range(2):
                try:
                    reference = gen_reference(question_text, answer_text)
                    break
                except Exception:
                    reference = None
                    continue

            if reference is None:
                stats["skipped"] += 1
                new_turns.append(t)
                continue

            reference = " ".join(reference.split())  # strip newlines
            if not reference.startswith("Reference: "):
                reference = "Reference: " + reference

            new_turns.append({**t, "kind": "grounded", "reference": reference})
            stats["converted"] += 1
            stats["references_generated"] += 1
        out_segments.append({**seg, "turns": new_turns})
    return out_segments, stats


def load_jsonl(path):
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    return segments


def write_jsonl(path, segments):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for seg in segments:
            f.write(json.dumps(seg) + "\n")


def count_persona_kinds(segments):
    """Dry-run counting only — no API calls."""
    counts = {"persona": 0, "decline": 0, "grounded": 0}
    for seg in segments:
        for t in seg.get("turns", []):
            if t.get("speaker") == "DANIELLE" and t.get("kind") in counts:
                counts[t["kind"]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="replay/persona_segments.filtered.jsonl")
    ap.add_argument("--out", default="replay/persona_segments.retrieval.jsonl")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="count only, no API calls")
    args = ap.parse_args()

    segments = load_jsonl(args.in_path)

    if args.dry_run:
        counts = count_persona_kinds(segments)
        print(f"[dry-run] {args.in_path}: {len(segments)} segments")
        print(f"[dry-run] persona turns to convert: {counts['persona']}")
        print(f"[dry-run] decline turns (would pass through): {counts['decline']}")
        print(f"[dry-run] grounded turns (would pass through): {counts['grounded']}")
        print("[dry-run] no API calls made")
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (use a rotated key; never pass it on the command line)")

    gen_reference = make_gemini_reference_fn(args.model, key)
    out_segments, stats = mark_persona(segments, gen_reference)
    write_jsonl(args.out, out_segments)

    print(f"wrote {args.out}")
    print(f"persona -> grounded converted: {stats['converted']}")
    print(f"skipped (generation failed twice, left as persona): {stats['skipped']}")
    print(f"decline turns untouched: {stats['declines_untouched']}")
    print(f"grounded turns untouched: {stats['grounded_untouched']}")
    print(f"total references generated: {stats['references_generated']}")


if __name__ == "__main__":
    main()

"""Rewrite vague persona references with bio-grounded concrete facts.

The first-pass references were written by Gemini from Danielle's answers alone,
before the template carried her bio. ~22% are vague/circular ("Danielle liked
something about Dublin") and at least one is factually wrong ("director of
visual platforms"). Every such turn trains fire -> receive-nothing-useful.

This pass rewrites every grounded reference against her actual bio; declines
are kept untouched. Output is a NEW manifest (v2) — the original is never
modified. Clips, ids, paths, kinds all pass through unchanged, so the
downstream precompute/verify/train pipeline is identical.

Usage:
    export GEMINI_API_KEY=...   # or source .env
    python scripts/rewrite_references.py \
        --in  replay/persona_rag_all_manifest.jsonl \
        --out replay/persona_rag_all_manifest_v2.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mark_persona_retrieval import call_gemini_text  # same stdlib REST client

MODEL_DEFAULT = "gemini-flash-latest"

BIO = """- Danielle DeLosa, 35, born 25 February 1991. Lives in Tampa, Florida. From Brooklyn, New York.
- Work: Director of Digital Platforms, legal industry. Owns the strategy and roadmap for a single governed source of truth for company data, supporting reporting and AI capability. Also optimises digital capabilities and runs the AI adoption strategy.
- Travel: mostly Europe. Favourite is Ireland, especially Dublin, the Cliffs of Moher and the Giant's Causeway. Dream trip is the puffin migration in Iceland.
- Fighting: tae kwon do, wrestling, krav maga, bare knuckle boxing. Started in middle school and competed for many years.
- Music: heavy metal, hardcore and punk. Travels for festivals, went to Hellfest in France this year. Favourite bands are Minor Threat, Hatebreed, Emmure, H2O and Deftones. Bucket list show is System of a Down.
- Tattoos: over twenty, including a full sleeve, mostly American traditional or Japanese traditional. Has a travel sleeve with a tattoo from every new country she visits.
- Plants: over sixty house plants. Her first was a six inch monstera from Ikea, now six feet tall after eight years. Favourite is her finger leaf philodendron, also called philodendron goeldii."""

PROMPT = """You maintain reference documents for a retrieval-augmented chatbot that answers as Danielle.

Danielle's bio (the only source of truth):
{bio}

Below is an existing reference document. Rewrite it as ONE line of concrete, factual content about Danielle on the SAME topic, using only facts from the bio (plus any concrete personal fact already present in the original that the bio omits). No vagueness ("liked something", "confirmed that she mentioned"), no meta commentary, no markdown. Maximum 50 words. Output exactly one line starting with "Reference: ".

Existing reference: {ref}"""


def needs_prefix(text: str) -> str:
    text = text.strip().replace("\n", " ")
    if not text.lower().startswith("reference:"):
        text = "Reference: " + text
    return text


def rewrite_manifest(rows, gen):
    """Pure-ish core: rows -> (new_rows, stats). gen(ref_text) -> new text or raises."""
    stats = {"rewritten": 0, "declines_kept": 0, "failed_kept_original": 0}
    out = []
    for row in rows:
        new_refs = []
        for kind, ref in zip(row.get("kinds", []), row.get("references", [])):
            if kind == "decline":
                new_refs.append(ref)
                stats["declines_kept"] += 1
                continue
            try:
                new_refs.append(needs_prefix(gen(ref)))
                stats["rewritten"] += 1
            except Exception:
                new_refs.append(ref)
                stats["failed_kept_original"] += 1
        new_row = dict(row)
        new_row["references"] = new_refs
        out.append(new_row)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (source .env)")

    rows = [json.loads(l) for l in open(args.inp)]

    def gen(ref):
        return call_gemini_text(PROMPT.format(bio=BIO, ref=ref), args.model, key)

    new_rows, stats = rewrite_manifest(rows, gen)

    with open(args.out, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    total_refs = stats["rewritten"] + stats["declines_kept"] + stats["failed_kept_original"]
    print(f"rows: {len(new_rows)}, refs: {total_refs}")
    print(f"rewritten: {stats['rewritten']}, declines kept: {stats['declines_kept']}, "
          f"failed (original kept): {stats['failed_kept_original']}")
    # show three before/afters for eyeballing
    shown = 0
    for old, new in zip(rows, new_rows):
        for k, o, n in zip(old["kinds"], old["references"], new["references"]):
            if k != "decline" and o != n and shown < 3:
                print(f"\nBEFORE: {str(o)[:110]}\nAFTER:  {str(n)[:110]}")
                shown += 1


if __name__ == "__main__":
    main()

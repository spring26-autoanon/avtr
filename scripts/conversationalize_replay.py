#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 4.2: turn QA triples into 2-turn stereo-ready dialogues.

Deterministic restructure (no LLM, no cost). Each {question, reference, answer} becomes a
short 2-speaker exchange with retrieval semantics set per kind:

  turns[0] = user (channel 1)      : the question
  turns[1] = assistant (channel 0) : the answer, with ret_before = whether ⟨ret⟩ fires first

Retrieval per kind (mirrors serve):
  grounded  -> retrieval=True : ⟨ret⟩ fires, the reference IS used, answer is faithful to it.
  decline   -> retrieval=True : ⟨ret⟩ fires, reference is fetched but IRRELEVANT, answer declines.
  smalltalk -> retrieval=False: no ⟨ret⟩, no reference, casual reply.

Output JSONL (one dialogue per line):
  {"id","kind","retrieval","reference","turns":[{"speaker","text","channel","ret_before"},...]}

`reference` (text) is what Task 5 POSTs to the :8001 ARC encoder for the reference_with_time
tensor. `ret_before` on the assistant turn marks where Task 5 places the ⟨ret⟩/rag_token in the
inner-monologue alignment (at the start of the grounded answer). Richer turn-taking
(backchannels/interruptions) is a Stage-3 enrichment; this minimal 2-turn form is enough to
teach the retrieval behavior for Stage 2b.

    python scripts/conversationalize_replay.py --in replay/qa.jsonl --out replay/dialogue.jsonl
"""
import argparse
import json

USER_CHANNEL = 1       # partner / user
ASSISTANT_CHANNEL = 0  # Danielle / main (SPEAKER_MAIN)

RETRIEVES = {"grounded": True, "decline": True, "smalltalk": False}


def to_dialogue(rec: dict) -> dict:
    kind = rec["kind"]
    retrieval = RETRIEVES[kind]
    reference = rec["reference"] if retrieval else ""
    return {
        "id": rec["id"],
        "kind": kind,
        "domain": rec.get("domain", ""),
        "retrieval": retrieval,
        "reference": reference,
        "turns": [
            {"speaker": "user", "channel": USER_CHANNEL,
             "text": rec["question"], "ret_before": False},
            {"speaker": "assistant", "channel": ASSISTANT_CHANNEL,
             "text": rec["answer"], "ret_before": retrieval},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/qa.jsonl")
    ap.add_argument("--out", default="replay/dialogue.jsonl")
    args = ap.parse_args()

    n = 0
    with open(args.inp) as f, open(args.out, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["kind"] not in RETRIEVES:
                raise SystemExit(f"unknown kind {rec['kind']!r} in {rec.get('id')}")
            out.write(json.dumps(to_dialogue(rec), ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} dialogues -> {args.out}")


if __name__ == "__main__":
    main()

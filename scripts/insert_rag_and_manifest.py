#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 5: insert the ⟨ret⟩ marker + build the replay train manifest.

After annotate.py writes <id>.json (Whisper alignments of her channel) next to each replay wav:
- For each RETRIEVAL example (grounded/decline), insert a ["<RAG>", [start-0.1, start], "SPEAKER_MAIN"]
  alignment just before her first word, so the interleaver emits rag_token_id (4) at the answer
  start and the model learns to trigger retrieval. (smalltalk examples get nothing.)
- Write replay/train.jsonl ({path, duration}) over all rendered wavs, for mixing with the
  dialogue data.

Idempotent (won't double-insert). Runs on the box (needs the annotate .json files present).

    ~/fork-venv/bin/python scripts/insert_rag_and_manifest.py \
        --render replay/render_manifest.jsonl --audiodir replay/audio --out replay/train.jsonl
"""
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="replay/render_manifest.jsonl")
    ap.add_argument("--audiodir", default="replay/audio")
    ap.add_argument("--out", default="replay/train.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.render) if l.strip()]
    train, n_rag, n_missing = [], 0, 0
    for r in rows:
        wav = os.path.join(args.audiodir, f"{r['id']}.wav")
        jf = os.path.join(args.audiodir, f"{r['id']}.json")
        if not os.path.exists(jf):
            print(f"  WARN no alignment json for {r['id']} (annotate incomplete?)")
            n_missing += 1
            continue
        data = json.load(open(jf))
        al = data.get("alignments", [])
        if r.get("retrieval") and al and not (al and al[0][0] == "<RAG>"):
            first_start = min(a[1][0] for a in al)
            rag = ["<RAG>", [max(0.0, first_start - 0.1), first_start], "SPEAKER_MAIN"]
            data["alignments"] = [rag] + al
            json.dump(data, open(jf, "w"))
            n_rag += 1
        train.append({"path": os.path.abspath(wav), "duration": r["duration_sec"]})

    with open(args.out, "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")

    print(f"inserted <RAG> into {n_rag} retrieval examples; {n_missing} missing json")
    print(f"wrote {args.out} with {len(train)} entries")


if __name__ == "__main__":
    main()

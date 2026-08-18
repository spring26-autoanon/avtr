#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 6: mix the dialogue + replay manifests for the Stage-2b retrain.

Concatenates the voice-dialogue train manifest with the replay train manifest into one
train.jsonl. `--dialogue-repeat` duplicates the dialogue entries to rebalance (the dialogue
files are long / many windows, the replay files short / one window each) — the replay:dialogue
ratio is the voice-vs-retrieval dial; tune it if voice smears or retrieval stays weak.

Reports total duration per source so you can see the effective balance.

    ~/fork-venv/bin/python scripts/mix_manifests.py \
        --dialogue finetune/data/prepared_dialogue/train.jsonl \
        --replay replay/train.jsonl --out replay/stage2b_train.jsonl --dialogue-repeat 3
"""
import argparse
import json


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def dur(rows):
    return sum(r.get("duration", 0) for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", default="replay/stage2b_train.jsonl")
    ap.add_argument("--dialogue-repeat", type=int, default=1)
    args = ap.parse_args()

    d = load(args.dialogue)
    r = load(args.replay)
    mixed = d * args.dialogue_repeat + r

    with open(args.out, "w") as f:
        for e in mixed:
            f.write(json.dumps(e) + "\n")

    dd = dur(d) * args.dialogue_repeat
    rr = dur(r)
    tot = dd + rr or 1
    print(f"dialogue: {len(d)}x{args.dialogue_repeat} entries, {dd/60:.1f} min ({100*dd/tot:.0f}%)")
    print(f"replay:   {len(r)} entries, {rr/60:.1f} min ({100*rr/tot:.0f}%)")
    print(f"wrote {args.out}  ({len(mixed)} entries, {tot/60:.1f} min total)")


if __name__ == "__main__":
    main()

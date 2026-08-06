#!/usr/bin/env python3
"""Stage 3 — cut per-dialogue clips from the retrieval master, with per-turn <RAG> markers.

For each validated segment: cut the stereo audio, slice the master's channel-0 alignments to
the segment and rebase them to zero, then insert a "<RAG>" alignment before the first word of
EACH grounded/decline turn. interleaver._tokenize maps "<RAG>" to raw token 4, so the model
learns to emit the retrieval trigger at the moment a factual question is answered — not once
at the head of the clip, which is what the synthetic terminal Q->A data taught.

The i-th <RAG> marker corresponds to the i-th entry of the manifest's "references" list.
train.py pairs them positionally, so a segment where they would disagree is skipped.

    python3 scripts/cut_retrieval_clips.py \
        --master finetune/data/prepared_retrieval/danielle_joshuarhodes.wav \
        --alignments finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
        --segments replay/retrieval_segments.filtered.jsonl \
        --outdir replay/retrieval --manifest replay/retrieval_manifest.jsonl
"""
import argparse
import json
from pathlib import Path

import soundfile as sf

TARGET_SR = 24000
RETRIEVAL_KINDS = ("grounded", "decline")


def slice_alignments(alignments, start, end):
    """Words fully inside [start, end), rebased so the clip begins at 0.0."""
    out = []
    for word, (ws, we), speaker in alignments:
        if ws >= start and we <= end:
            out.append([word, [round(ws - start, 3), round(we - start, 3)], speaker])
    return out


def insert_rag_markers(alignments, retrieval_turn_starts):
    """Insert a <RAG> alignment before the first word at or after each retrieval turn start.

    Timestamps are clip-relative. Each marker occupies the 0.1 s before the word it precedes,
    clamped at 0.0. Turn starts are sorted first: the emitted markers must be in chronological
    order because train.py pairs the i-th marker with the i-th reference.
    """
    if not retrieval_turn_starts:
        return alignments

    anchors = []
    for turn_start in sorted(retrieval_turn_starts):
        for i, (_w, (ws, _we), _s) in enumerate(alignments):
            if ws >= turn_start - 1e-6:
                anchors.append(i)
                break

    out = []
    for i, item in enumerate(alignments):
        for _ in range(anchors.count(i)):
            word_start = item[1][0]
            out.append(["<RAG>", [round(max(0.0, word_start - 0.1), 3), word_start],
                        "SPEAKER_MAIN"])
        out.append(item)
    return out


def _retrieval_turns(segment):
    """Her retrieval turns, in time order — one per QUESTION, not one per fragment.

    A retrieval trigger corresponds to a question being answered. If JOSHUA has not spoken
    since her last retrieval turn, she did not retrieve again: her single answer was merely
    split into fragments by the pause-merge, and Gemini labelled each fragment `grounded`.
    Emitting a <RAG> per fragment would train the model to fire ⟨ret⟩ repeatedly while it is
    still speaking one reply — over-triggering baked straight into the data. One real clip
    had four triggers inside a single Gurpurab answer, one of them mid-sentence.

    Only a JOSHUA turn opens a new retrieval opportunity; her own smalltalk aside does not.
    """
    out = []
    joshua_since = True          # the segment always opens on a JOSHUA turn
    for t in sorted(segment["turns"], key=lambda x: x["start"]):
        if t["speaker"] == "JOSHUA":
            joshua_since = True
            continue
        if t["kind"] in RETRIEVAL_KINDS and t.get("reference"):
            if joshua_since:
                out.append(t)
                joshua_since = False
    return out


def cut_all(master_path, alignments, segments, outdir, manifest_path):
    """Cut every segment. Returns the list of manifest rows written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(master_path))
    if info.samplerate != TARGET_SR or info.channels != 2:
        raise SystemExit(f"{master_path}: expected 2ch/{TARGET_SR} Hz, got "
                         f"{info.channels}ch/{info.samplerate} Hz")

    rows = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        clip_alignments = slice_alignments(alignments, start, end)
        if not clip_alignments:
            print(f"  SKIP {seg['id']}: no Danielle words inside")
            continue

        turns = _retrieval_turns(seg)
        turn_starts = [t["start"] - start for t in turns]
        references = [t["reference"] for t in turns]
        kinds = [t["kind"] for t in turns]

        marked = insert_rag_markers(clip_alignments, turn_starts)
        n_markers = sum(1 for w in marked if w[0] == "<RAG>")
        if n_markers != len(references):
            print(f"  SKIP {seg['id']}: {n_markers} <RAG> markers vs {len(references)} references")
            continue

        frames_start = int(round(start * TARGET_SR))
        frames_stop = int(round(end * TARGET_SR))
        audio, _sr = sf.read(str(master_path), start=frames_start, stop=frames_stop,
                             always_2d=True)
        wav_path = outdir / f"{seg['id']}.wav"
        sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_24")

        with open(outdir / f"{seg['id']}.json", "w") as f:
            json.dump({"alignments": marked}, f, ensure_ascii=False)

        rows.append({
            "id": seg["id"],
            "path": str(wav_path.resolve()),
            "duration_sec": audio.shape[0] / TARGET_SR,
            "references": references,
            "kinds": kinds,
        })

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--alignments", required=True)
    ap.add_argument("--segments", default="replay/retrieval_segments.filtered.jsonl")
    ap.add_argument("--outdir", default="replay/retrieval")
    ap.add_argument("--manifest", default="replay/retrieval_manifest.jsonl")
    args = ap.parse_args()

    with open(args.alignments) as f:
        alignments = json.load(f)["alignments"]
    segments = [json.loads(line) for line in open(args.segments) if line.strip()]

    rows = cut_all(args.master, alignments, segments, args.outdir, args.manifest)

    total = sum(r["duration_sec"] for r in rows)
    n_ret = sum(len(r["references"]) for r in rows)
    multi = sum(1 for r in rows if len(r["references"]) > 1)
    print(f"\nwrote {len(rows)} clips ({total / 60:.1f} min) -> {args.outdir}")
    print(f"{n_ret} retrieval turns; {multi} clips have more than one")
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()

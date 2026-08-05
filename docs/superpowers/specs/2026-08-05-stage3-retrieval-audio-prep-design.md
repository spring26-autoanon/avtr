# Stage 3 — real retrieval-audio prep design

Date: 2026-08-05
Status: approved, ready for implementation planning

## Problem

Stage 3 (see `docs/moshi-rag-experiments.md`) calls for retrieval to be trained
*inside* natural conversation, in Danielle's real voice, so the model stops
trading persona for retrieval along the checkpoint axis.

The retrieval half is now recorded: `replay/QuestionAudio/` holds two isolated
mono tracks — `audioDanielleDeLosa2…_24khz.wav` and
`audioJoshuaRhodes1…_24khz.wav` — each 84.65 min, 24 kHz, PCM_24, byte-identical
in length and therefore already time-aligned.

Two facts shape the whole design:

1. **The trainer loads one reference tensor per WAV file.**
   `interleaver.py:315` reads a sibling `<stem>.ref.safetensors`, while
   `dataset.py:48-55` chunks long files into `duration_sec` (100 s) windows. An
   85-minute file can carry exactly one reference, so it must be cut into one
   clip per dialogue.
2. **The recording is not verbatim from `replay/recording_dialogues.jsonl`.**
   Danielle used that file as a loose guideline and wrote her own script, which
   was not saved. The 300 scripted `reference` passages therefore cannot be
   matched to what she actually said, and no written script exists to segment
   against — only audio.

## Approach

Transcribe the master once, have an LLM segment the transcript into dialogue
units and write a reference passage per unit, then cut audio clips at those
boundaries while **slicing and offsetting the existing alignments** instead of
re-transcribing each clip. Whisper runs once over 85 minutes rather than ~100
times over clips; the offset-slice is the same operation `interleaver.py:297-303`
already performs when windowing long files.

Rejected alternatives:

- **VAD/silence segmentation on Joshua's channel.** Over-segments multi-turn
  dialogues into Q→A fragments, destroying the multi-turn retrieval structure
  that is the entire point of Stage 3. Still needs an LLM for references, so it
  saves little.
- **Fixed 100 s windows, one reference each.** Window edges land mid-dialogue and
  references misalign, re-teaching "conditioning is noise" — the Stage 2b failure
  this work exists to avoid.

### Accepted costs

**References are reverse-engineered.** At serve time retrieval supplies a real
document and the model answers from it; here only the answer exists, so the
passage is written backwards from it. This risks a passage that merely restates
her answer, making training easier than live conditions. Mitigated by prompting
for a natural encyclopedic passage containing surrounding detail she did not
say, so the model must still select from the passage. This is strictly weaker
than the synthetic set, where the passage came first, and is the price of
unscripted recording.

**`decline` and `smalltalk` counts are whatever she happened to record.** In the
synthetic set, `decline` meant *an irrelevant passage was supplied and she
declined rather than hallucinate* — those examples are why trial 4's
`checkpoint_000400` declined gracefully instead of confabulating. Classification
here is from the transcript, and the resulting mix is reported. If declines come
out thin, top up from the surviving synthetic replay clips.

## Data flow

```
replay/QuestionAudio/audio{Danielle…,Joshua…}_24khz.wav   (2 × 84.65 min mono, aligned)
  │
  ├─[local] pair_dialogue_stereo.py --no-eval
  │     └─► finetune/data/prepared_retrieval/danielle_joshuarhodes_qa.wav  (L=Danielle, R=Joshua)
  │           ⇢ rsync to A100 (one ~730 MB upload)
  │
  ├─[A100] annotate.py --channel 0                          → …qa.json      (her words)
  ├─[A100] annotate.py --channel 1 --out-suffix .ch1.json    → …qa.ch1.json  (Joshua's words)
  │           ⇢ pull both JSON back (small)
  │
  ├─[local] segment_retrieval_audio.py            (Gemini)
  │     └─► replay/retrieval_segments.jsonl
  │         [{id, start, end, kind, reference, turns}]
  │           ⇢ push segments.jsonl (small)
  │
  ├─[A100] cut_retrieval_clips.py
  │     └─► replay/retrieval/<id>.wav        (stereo clip, ≤100 s)
  │         replay/retrieval/<id>.json       (ch0 alignments, offset-sliced, <RAG> prepended)
  │         replay/retrieval_manifest.jsonl  (render_manifest-compatible)
  │
  ├─[A100] precompute_references.py  (:8001 ARC encoder)  → <id>.ref.safetensors
  │
  └─[A100] mix_manifests.py ⊕ finetune/data/prepared_dialogue/train.jsonl
        └─► train, RAG_TOKEN_WEIGHT ≈ 15
```

Pairing and segmentation are CPU/API work and stay on the Mac. Annotation and
reference-encoding need CUDA and the `:8001` service, so they run on the A100.
Clip cutting happens on the box, so after the first upload only small JSON files
cross the wire.

## Components

### Changes to existing files

**`scripts/pair_dialogue_stereo.py`** — add `--no-eval` to skip the eval carve.
The script currently requires an `--eval-conv` and exits if it is not found; this
is a single conversation we want kept whole. Eval continues to come from the
natural set's existing `danielle_joshuarhodes_eval.wav`.

**`annotate.py`** — `run()` hardcodes `channel=0` (line 188) and
`out_file = path.with_suffix(".json")` (line 176). Add `--channel` and
`--out-suffix` flags and thread them through. `process_one` already accepts a
`channel` argument, so this is plumbing only. Without `--out-suffix` the
channel-1 pass would clobber the channel-0 transcript.

**`scripts/mix_manifests.py`** — add `--replay-repeat` alongside the existing
`--dialogue-repeat`, so the retrieval side can be upweighted to reach 50/50.

### New: `scripts/segment_retrieval_audio.py`

Input: the two alignment JSONs. Output: `replay/retrieval_segments.jsonl`.

1. **Merge to utterances.** Group word-level alignments into utterances (words
   with <0.6 s gaps, break on speaker change), producing a speaker-tagged
   timestamped transcript:
   ```
   [00:12.4] JOSHUA:   hey danielle, can you tell me what the main theme of diwali is
   [00:19.1] DANIELLE: at its heart it's the festival of lights, celebrating…
   ```
2. **Segment via Gemini** in ~12-minute windows with 1 minute of overlap, so a
   dialogue straddling a window edge is still seen whole. Segments are deduped by
   start time. Reuses the Gemini client pattern from
   `scripts/gen_recording_dialogues.py`.
3. **Per unit Gemini returns** `start`, `end`, `kind`, `reference`:
   - `grounded` — a 2–5 sentence encyclopedic passage supporting her answer,
     deliberately including surrounding detail she did not say.
   - `decline` — a passage on a related but different topic (reference present
     but unhelpful).
   - `smalltalk` — no reference.
4. **Snap and validate.** LLM timestamps are not trusted. Every boundary snaps to
   a real utterance boundary from the transcript, then:
   - monotonic and non-overlapping (overlaps merged or dropped);
   - each segment starts on a Joshua turn and ends on a Danielle turn;
   - ≥4 s long and contains ≥1 Danielle word;
   - ≤100 s (`duration_sec`); longer segments split at the internal Joshua turn
     nearest the midpoint, both halves keeping the same `kind` and `reference`,
     each receiving its own `<RAG>`.
   Audio claimed by no segment (retakes, false starts, chatter) is unused; the
   script reports how many minutes were dropped.
5. **Faithfulness pass**, reusing the `scripts/faithfulness_filter.py` pattern:
   Gemini judges whether her answer is supported by the generated reference.
   Failures go to `replay/retrieval_dropped.jsonl`. Because references derive
   from answers this should mostly pass; it exists to catch wrong-topic
   hallucinations.

### New: `scripts/cut_retrieval_clips.py`

Input: master stereo WAV, `retrieval_segments.jsonl`, master ch0 alignments.
Output: per-clip WAV + JSON + a `render_manifest`-compatible manifest.

- Cut `[start, end]` from the master stereo WAV, preserving 24 kHz / PCM_24.
- Slice the ch0 alignments to the segment and subtract `start` from every
  timestamp.
- For `grounded` and `decline`, prepend
  `["<RAG>", [first_start-0.1, first_start], "SPEAKER_MAIN"]` — identical to
  `scripts/insert_rag_and_manifest.py`, applied after offsetting, idempotent.
- Emit `replay/retrieval_manifest.jsonl` with the fields
  `precompute_references.py` expects (`id`, `retrieval`, `reference`,
  `duration_sec`), so that script runs unchanged and does not need to know the
  audio is real now.

### Reused unchanged

`scripts/precompute_references.py`, `scripts/build_manifest.py`, and the
reference-conditioning training surgery already applied for Stage 2b.

## Training

New config `example/moshika_rag_stage3.yaml`, copied from
`example/moshika_rag_stage2b.yaml`:

- **Fresh LoRA**, not continued from a Stage 2b checkpoint — retrieval must be in
  the loss from step 0.
- `RAG_TOKEN_WEIGHT=15`, down from trial 4's 25, because retrieval now occurs
  inside natural conversation rather than in isolated formal Q&A.
- Mix targeted at roughly 50/50 by duration. The natural train set is 135.3 min
  (`danielle_clays` 59.1 + `..._train_a` 38.1 + `..._train_b` 38.1; the 10.0 min
  eval file is excluded). Against 84.65 min of retrieval, `--replay-repeat 2`
  gives 169.3 vs 135.3 min ≈ **56/44** — the closest whole-number repeat to
  parity, and the ratio remains the voice-vs-retrieval dial if either side is
  weak. Duration is the right dial because the loader chunks by duration, so
  minutes ≈ training windows.
- Eval from the existing `danielle_joshuarhodes_eval.wav`.
- Checkpoints every 100 steps.

## Verification

Gate before spending GPU training time:

1. Every clip WAV has a non-empty sibling `.json`.
2. Every non-smalltalk clip has a `.ref.safetensors`.
3. Segment counts reported by kind, with total minutes kept vs. dropped.
4. Listen to a few cut clips and confirm boundaries land at dialogue starts.

Post-training, judge **both** voice and retrieval per the experiment log: a
factual probe answerable only from the corpus, and — the specific Stage 3 target
— a factual question asked *after* several turns of chit-chat, which previously
primed persona mode and produced confabulation.

# Stage 3 — real retrieval-audio prep design

Date: 2026-08-05
Status: approved, ready for implementation planning

## Success criterion

**One session, both modes, fluid switching.** The model must hold casual
conversation in Danielle's voice, retrieve when a question needs external
knowledge, and move between the two within a single conversation — without
having to be primed into "retrieval mode" by opening with a factual question.

This is the target every decision below serves. It rules out the Stage 2b
outcome where retrieval and persona lived at different checkpoints, and it also
rules out its mirror image: firing `⟨ret⟩` during chit-chat.

## Problem

Stage 3 (see `docs/moshi-rag-experiments.md`) calls for retrieval to be trained
*inside* natural conversation, in Danielle's real voice, so the model stops
trading persona for retrieval along the checkpoint axis.

The retrieval half is now recorded: `replay/QuestionAudio/` holds two isolated
mono tracks — `audioDanielleDeLosa2…_24khz.wav` and
`audioJoshuaRhodes1…_24khz.wav` — each 84.65 min, 24 kHz, PCM_24, byte-identical
in length and therefore already time-aligned.

Three facts shape the design:

1. **The trainer loads one reference tensor per WAV file.**
   `interleaver.py:315` reads a sibling `<stem>.ref.safetensors`, while
   `dataset.py:48-55` chunks long files into `duration_sec` (100 s) windows. An
   85-minute file can carry exactly one reference, so it must be cut into clips.
2. **The recording is not verbatim from `replay/recording_dialogues.jsonl`.**
   Danielle used that file as a loose guideline and wrote her own script, which
   was not saved. The 300 scripted `reference` passages cannot be matched to what
   she actually said, and no written script exists to segment against.
3. **The existing retrieval plumbing assumes one retrieval turn per clip.**
   Every synthetic replay clip was a terminal Q→A pair, so
   `insert_rag_and_manifest.py` prepends a single `<RAG>` before her first word,
   and `train.py:272-273` injects the reference at `hit[0]` — the *first*
   `⟨ret⟩` frame only. Real multi-turn recordings break both assumptions.

## Approach

Transcribe the master once, have an LLM segment the transcript into dialogue
units **and label each of her turns individually**, then cut audio clips at those
boundaries while **slicing and offsetting the existing alignments** instead of
re-transcribing each clip. Whisper runs once over 85 minutes rather than ~100
times over clips; the offset-slice is the same operation `interleaver.py:297-303`
already performs when windowing long files.

Rejected alternatives:

- **VAD/silence segmentation on Joshua's channel.** Over-segments multi-turn
  dialogues into Q→A fragments, destroying the multi-turn structure that the
  success criterion depends on. Still needs an LLM for references.
- **Fixed 100 s windows, one reference each.** Window edges land mid-dialogue and
  references misalign, re-teaching "conditioning is noise" — the Stage 2b failure
  this work exists to avoid.

### Accepted costs

**References are reverse-engineered.** At serve time retrieval supplies a real
document and the model answers from it; here only the answer exists, so the
passage is written backwards from it. This risks a passage that merely restates
her answer, making training easier than live conditions. Mitigated by prompting
for a natural encyclopedic passage containing surrounding detail she did not
say, so the model must still select. Strictly weaker than the synthetic set,
where the passage came first, and the price of unscripted recording.

**Turn kinds are whatever she happened to record.** In the synthetic set,
`decline` meant *an irrelevant passage was supplied and she declined rather than
hallucinate* — those examples are why trial 4's `checkpoint_000400` declined
gracefully instead of confabulating, and `smalltalk` was the no-retrieval
negative class (`gen_replay_qa.py:10`,
`conversationalize_replay.py:32`). Classification here is from the transcript,
and the resulting turn-level mix is reported. If declines come out thin, top up
from the surviving synthetic replay clips.

## Data flow

```
replay/QuestionAudio/audio{Danielle…,Joshua…}_24khz.wav   (2 × 84.65 min mono, aligned)
  │
  ├─[local] pair_dialogue_stereo.py --no-eval
  │     └─► finetune/data/prepared_retrieval/danielle_joshuarhodes_qa.wav  (L=Danielle, R=Joshua)
  │           ⇢ rsync to A100 (one ~730 MB upload)
  │
  ├─[A100] annotate.py --channel 0                        → …qa.json      (her words)
  ├─[A100] annotate.py --channel 1 --out-suffix .ch1.json  → …qa.ch1.json  (Joshua's words)
  │           ⇢ pull both JSON back (small)
  │
  ├─[local] segment_retrieval_audio.py            (Gemini)
  │     └─► replay/retrieval_segments.jsonl
  │         [{id, start, end, turns:[{speaker,start,end,kind,reference}]}]
  │           ⇢ push segments.jsonl (small)
  │
  ├─[A100] cut_retrieval_clips.py
  │     └─► replay/retrieval/<id>.wav        (stereo clip, ≤100 s)
  │         replay/retrieval/<id>.json       (ch0 alignments, offset-sliced, N × <RAG>)
  │         replay/retrieval_manifest.jsonl  (one row per clip, references as a list)
  │
  ├─[A100] precompute_references.py  (:8001 ARC encoder) → <id>.ref.safetensors (N tensors)
  │
  └─[A100] train, train_data = "retrieval:0.5,dialogue:0.5"
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
`--out-suffix` flags and thread them through; `process_one` already accepts
`channel`. Without `--out-suffix` the channel-1 pass would clobber the
channel-0 transcript.

**`scripts/precompute_references.py`** — write **N** reference tensors per clip
(keys `reference_0 … reference_{N-1}`), in the same order as the clip's `<RAG>`
markers, instead of a single `reference` key. Single-reference clips still write
`reference_0`.

**`finetune/data/interleaver.py`** — `_load_reference_tensor` (lines 17-25)
currently returns `load_file(ref_path)["reference"]`. Return an ordered **list**
of tensors read from the `reference_i` keys. `Sample.reference_tensor` and
`Batch.reference_tensors` become lists-of-lists; the collate at lines 51-56
already just gathers them and needs no change.

**`train.py`** (lines 254-279, originally written by
`scripts/apply_rag_positioning.py`) — the injection loop takes
`start = int(hit[0])`, i.e. the first `⟨ret⟩` frame only. Replace with a loop
over **all** hits, pairing the i-th hit with the i-th reference tensor, and
clamping each span to `min(start + T_ref, next_hit)` so a long reference cannot
bleed over the following retrieval turn. If the hit count and reference count
disagree, log and fall back to injecting only the pairs that match, so one bad
clip cannot corrupt a batch. `apply_rag_positioning.py` is updated so the patch
stays reproducible on a fresh checkout.

Rationale: without this, every retrieval turn after the first in a clip receives
no conditioning while the loss still demands a grounded answer — training
confabulation directly.

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
   dialogue straddling a window edge is still seen whole; segments deduped by
   start time. Reuses the Gemini client pattern from
   `scripts/gen_recording_dialogues.py`.
3. **Label per turn, not per segment.** Each of *her* turns gets its own `kind`:
   - `grounded` — answers a factual question; carries its own 2–5 sentence
     encyclopedic reference passage, deliberately including surrounding detail
     she did not say, so the model must select rather than parrot.
   - `decline` — carries a passage on a related but different topic (reference
     present but unhelpful).
   - `smalltalk` — no reference, no `<RAG>`.

   A segment keeps a summary label for reporting only. Per-turn labelling is what
   makes the success criterion reachable: a clip that runs chit-chat → factual
   question → grounded answer → chit-chat teaches the *switch*, which is exactly
   the behavior being bought. Segments containing a mode transition are preferred
   over minimal Q→A units, subject to the 100 s cap.
4. **Snap and validate.** LLM timestamps are not trusted. Every boundary snaps to
   a real utterance boundary from the transcript, then:
   - monotonic and non-overlapping (overlaps merged or dropped);
   - each segment starts on a Joshua turn and ends on a Danielle turn;
   - ≥4 s long and containing ≥1 Danielle word;
   - ≤100 s (`duration_sec`); longer segments split at the internal Joshua turn
     nearest the midpoint, each half keeping the turn labels and references that
     fall inside it.
   Audio claimed by no segment (retakes, false starts, chatter) is unused; the
   script reports how many minutes were dropped.
5. **Faithfulness pass**, reusing the `scripts/faithfulness_filter.py` pattern:
   Gemini judges whether each grounded answer is supported by its generated
   reference. Failures go to `replay/retrieval_dropped.jsonl`. Because references
   derive from answers this should mostly pass; it catches wrong-topic
   hallucinations.

### New: `scripts/cut_retrieval_clips.py`

Input: master stereo WAV, `retrieval_segments.jsonl`, master ch0 alignments.
Output: per-clip WAV + JSON + manifest.

- Cut `[start, end]` from the master stereo WAV, preserving 24 kHz / PCM_24.
- Slice the ch0 alignments to the segment and subtract `start` from every
  timestamp.
- Insert `["<RAG>", [first_start-0.1, first_start], "SPEAKER_MAIN"]` before the
  first word of **each** grounded or decline turn — possibly several per clip,
  wherever they fall. This replaces `insert_rag_and_manifest.py`'s
  prepend-one-at-the-front logic, which would otherwise teach the model to fire
  `⟨ret⟩` at the start of a conversation rather than when a factual question
  arrives.
- **The i-th `<RAG>` must correspond to the i-th reference**; the manifest lists
  references in marker order and the script asserts the counts agree before
  writing.
- Emit `replay/retrieval_manifest.jsonl` with `id`, `references` (ordered list),
  `duration_sec`, and the per-turn kinds.

### Reused unchanged

`scripts/build_manifest.py`, and the reference-conditioning training surgery
already applied for Stage 2b (`apply_replay_training_surgery.py`,
`apply_replay_data_plumbing.py`).

`scripts/mix_manifests.py` is **not** used. See below.

## Training

New config `example/moshika_rag_stage3.yaml`, copied from
`example/moshika_rag_stage2b.yaml`:

- **Fresh LoRA**, not continued from a Stage 2b checkpoint — retrieval must be in
  the loss from step 0.
- **Sampling weights, not manifest duplication.** `dataset.py:86-141` parses
  `train_data: "a.jsonl:0.5,b.jsonl:0.5"` into normalized sampling
  probabilities. Set the retrieval and natural manifests to `0.5/0.5`. This is
  exact rather than rounded to a whole-number repeat, decoupled from each side's
  duration, retunable by editing one line, and matches what trials 3 and 4 did —
  keeping results comparable. For reference the underlying durations are 84.65
  min retrieval vs 135.3 min natural train audio (`danielle_clays` 59.1 +
  `train_a` 38.1 + `train_b` 38.1; the 10.0 min eval file excluded).
- **`RAG_TOKEN_WEIGHT` is set after segmentation, not before.** The meaningful
  denominator is `⟨ret⟩`-bearing turns as a share of all her turns — a 90 s clip
  with one retrieval and six chit-chat turns is mostly *negative* evidence for
  triggering. Trial 4 used 25 and under-triggered; the Stage 3 plan proposed 15.
  If declines + smalltalk reach ≥30% of her turns, 15 is reasonable; if she
  recorded mostly straight factual Q&A, use 8–10. Prefer moving this lever over
  the sampling ratio: the weight targets only the trigger token, while the ratio
  moves voice, conversational style, and fact memorization at once.
- Eval from the existing `danielle_joshuarhodes_eval.wav`.
- Checkpoints every 100 steps.

Note on exposure: at 0.5 sampling, 800 steps × batch 8 draws ~3200 retrieval
windows from ~100 clips, ~32 views each. Trial 4 drew ~2400 from 127 clips
(~19 each) out of only 32 min of synthetic audio. Stage 3 has 2.6× the retrieval
material and it is real voice in real context, so the same nominal ratio buys
considerably more diversity.

## Verification

Gate before spending GPU training time:

1. Every clip WAV has a non-empty sibling `.json`.
2. For every clip, the number of `<RAG>` markers equals the number of reference
   tensors in its `.ref.safetensors`.
3. Turn-level mix reported: grounded / decline / smalltalk as a share of her
   turns, plus total minutes kept vs. dropped. This sets `RAG_TOKEN_WEIGHT`.
4. Listen to a few cut clips and confirm boundaries land at dialogue starts.

Post-training, judge against the success criterion rather than either mode
alone:

1. **Switch into retrieval:** several turns of chit-chat, *then* a factual
   question answerable only from the corpus. This is the case trial 4 failed —
   chit-chat primed persona mode and it confabulated.
2. **Switch back out:** after a successful retrieval, return to casual
   conversation and confirm she does not stay in formal-assistant register
   (trial 4's `checkpoint_000600` failure).
3. **No spurious triggering:** a full casual conversation with no factual
   questions should emit no `⟨ret⟩`.
4. **Voice throughout:** she should sound like herself in both modes.

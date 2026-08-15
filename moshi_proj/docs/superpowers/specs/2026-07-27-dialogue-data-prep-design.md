# Dialogue data prep for the Path-B (fork) voice retrain

_Design doc — 2026-07-27. Read alongside `NEXT_STEPS.md` Part 1 and `CLAUDE.md`._

## Problem

The first LoRA run cloned the target voice but monologued, because the training
data was single-speaker monologue with a **silent** user channel (Option A). The
fix, established in `NEXT_STEPS.md` Part 1, is to retrain on **two-speaker
dialogue** with each speaker isolated on one stereo channel. We now have that
data. This doc specifies how to turn the raw per-speaker mono tracks into the
stereo WAVs + manifests + transcripts the fork trainer consumes, and draws the
line between what happens on the Mac vs. the A100.

## Input data

Six mono tracks in `finetune/data/datastereo/clean_moshi_audio_24khz/`, all
**24 kHz / 24-bit / mono**. Filenames encode `audio<Speaker><idx><conv_id>_24khz.wav`
where **`conv_id` = the last 10 digits** before `_24khz`. They pair into three
conversations (matched frame counts within a pair → recorded simultaneously,
already time-aligned):

| conv_id      | left (main)        | right (partner)  | length   | frames (L / R)         |
| ------------ | ------------------ | ---------------- | -------- | ---------------------- |
| `1556527425` | Danielle De Losa   | Clay S           | 59.1 min | 85068000 / 85067400    |
| `1411304343` | Danielle De Losa   | Joshua Rhodes    | 86.2 min | 124169100 / 124169100  |
| `1341305451` | Danielle De Losa   | Leena Tantawy    | 1.1 min  | 1641300 / 1641300      |

**Danielle De Losa is the only speaker present in all three conversations**, so
she is the voice to clone. Total ≈ 2.4 hr of dialogue — within the "2–5 hr across
many partners" target from `NEXT_STEPS.md`.

## Key decisions (confirmed with user 2026-07-27)

1. **Channel assignment:** Danielle → **left (ch 0)** = `SPEAKER_MAIN`, the cloned
   voice, the only channel transcribed. Partners (Clay / Joshua / Leena) →
   **right (ch 1)**, audio only. `annotate.py` hard-codes `channel=0`, and Moshi's
   Inner Monologue models only the main speaker's text, so left **must** be the
   voice.
2. **Eval set:** carve a **~10 min (600 s) slice from the middle** of the 86-min
   Joshua conversation into a separate eval WAV (~6 windows → a stable `eval_loss`
   curve; the last run logged NaN from a too-small eval set). Middle rather than
   end so the eval isn't skewed by low-activity wind-down. The two remaining
   Joshua segments (before / after the slice) both go to train as separate files —
   **not** spliced, so no jump-cut lands inside a 100 s window. Everything else
   trains.
3. **No manual clip segmentation** — see "Windowing" below.

## Why the papers back this

- **Base Moshi §4.2 (Fisher):** the exact recipe — "pairs of participants,
  **recorded with separate channels per speaker**"; sample one as the "first (and
  main) speaker" whose stream carries Moshi's voice. → Danielle = main = left.
- **Inner Monologue §3.4.4:** the text stream models **only the main speaker's**
  transcription (they explicitly do *not* transcribe the user's flux), via Whisper
  **word-level timestamps** + padding tokens. → transcribe ch 0 only; exactly what
  `annotate.py` produces.
- **All audio resampled to 24 kHz.** → already are.
- **moshi-rag §4.1:** "fixed speaker as Moshi's voice, randomly sample another
  speaker" → fixed Danielle, varying partners.
- **Out of scope:** moshi-rag's RAG-specific data prep (⟨ret⟩-token placement via
  forced alignment, retrieval-delay augmentation) belongs to Part 4, not this
  Path-B voice run.

## Windowing — why the long files stay whole

The trainer windows each file internally, so we do **not** pre-cut into short
clips:

- `dataset.py:52` — each file is chopped into non-overlapping **`duration_sec`
  (=100 s)** windows on a fixed grid (0, 100, 200 …); the 86-min train file →
  ~46 windows, 59-min Clay → ~35, all shuffled together.
- `interleaver.py:272-277` — the alignment `.json` is sliced to
  `[start_sec, start_sec + duration_sec]` per window and timestamps rebased, so
  **one whole-file `.json` is correct**.

Consequences accepted: window boundaries can fall mid-turn (inherent to the
pipeline; was fine for the last run). Turn-aware or overlapping windows would be a
future `duration_sec`/windowing change, not a re-prep. Annotation of the 86-min
file is within `annotate.py`'s `dur > 3600*4` guard (4 hr); Whisper uses `auditok`
VAD for anything >10 s.

## Components

### 1. `scripts/pair_dialogue_stereo.py` (new)

Replaces the mono→silent-right logic of `prepare_stereo.py` (which no longer
applies — kept for reference/history).

Responsibilities:

- Scan `--src` for `*_24khz.wav`; parse each into `(speaker, participant_idx,
  conv_id)`, `conv_id` = last 10 digits before `_24khz`.
- Group by `conv_id`; assert exactly 2 tracks per group and that the main speaker
  is present in every group.
- Assign channels **by identity, not the 1/2 index** (that index is inconsistent —
  Danielle is "2" in two convs, "1" in the third): `--main-name danielle`
  (case-insensitive substring) → left; the other → right.
- Read both mono tracks, assert `sr == 24000`, **truncate to the shorter length**
  (handles the 600-frame / 25 ms mismatch in the Clay conv), `column_stack([main,
  partner])`. Print the per-conv frame delta so track sync can be eyeballed.
- **Eval carve:** `--eval-conv 1411304343 --eval-sec 600` (centered by default;
  optional `--eval-center-frac 0.5`) cuts a 600 s slice from the middle at frame
  boundaries, yielding three pieces: `<...>_train_a.wav` (before),
  `<...>_eval.wav` (middle), `<...>_train_b.wav` (after). The two train pieces are
  emitted separately, never concatenated.
- Write `PCM_24 / 24 kHz` stereo to `--dst finetune/data/prepared_dialogue/`
  (new dir; leaves the old monologue run in `prepared/` intact).
- Snake_case output names.

Outputs (5 files):
`danielle_clay.wav`, `danielle_joshua_train_a.wav`, `danielle_joshua_eval.wav`,
`danielle_joshua_train_b.wav`, `danielle_leena.wav`.
Joshua conv (5173.7 s) → train_a ≈ 2286.9 s (~38.1 min), eval = 600 s (10 min),
train_b ≈ 2286.9 s (~38.1 min). Train total ≈ 136 min, eval 10 min.

### 2. `scripts/build_manifest.py` (reused unchanged)

`--eval-file danielle_joshua_eval.wav` routes that file to `eval.jsonl` and all
others to `train.jsonl`; `all.jsonl` gets everything (used to drive annotation).
**Runs on the A100** — manifest paths are absolute and machine-specific.

### 3. `example/moshika_rag_voice.yaml` (small edit)

Point `train_data` / `eval_data` at `finetune/data/prepared_dialogue/`. Leave
hyperparameters as-is; add a one-line note that ~2.4 hr (vs the old 58 min) may
warrant more `max_steps` (a training-time call, tracked separately, not part of
data prep).

## Data flow — who runs what

| Step                                        | Where           | Tool                        |
| ------------------------------------------- | --------------- | --------------------------- |
| Pair mono → stereo, carve eval slice        | **Mac (now)**   | `pair_dialogue_stereo.py`   |
| rsync `prepared_dialogue/` → box            | Mac → A100      | rsync                       |
| Build manifests (absolute paths)            | **A100**        | `build_manifest.py`         |
| Transcribe ch 0 → sibling `.json`           | **A100 (CUDA)** | `annotate.py all.jsonl`     |
| Verify every wav has non-empty `.json`      | A100            | check                       |
| Train (fork, full LoRA on plain moshika)    | A100            | `torchrun -m train …`       |

The `.json` transcripts are **not** producible on the Mac — `annotate.py:87` calls
`.cuda()`, and 2.4 hr of audio on CPU Whisper would take ~half a day. That, and
manifest building, are A100 steps.

## Verification

Built into the script and a pytest against the real files:

- Each output: 2-channel / 24 kHz / PCM_24.
- Durations sum correctly: `danielle_joshua_train_a` + `danielle_joshua_eval` +
  `danielle_joshua_train_b` ≈ original Joshua length (within one frame).
- **Left-channel RMS matches the Danielle source; right matches the partner** —
  the make-or-break guard against swapped channels.
- One group per conv, exactly 2 tracks each, Danielle in all three.
- Manual: listen to ~30 s of one stereo output to confirm the two tracks are
  actually time-synced (matched frame counts are strong but not conclusive
  evidence; the RMS check won't catch a constant start-offset).

## Out of scope

- Generating `.json` transcripts / manifests locally (A100 steps).
- Hyperparameter retuning for the larger dataset (separate follow-up).
- moshi-rag RAG-specific data (⟨ret⟩ tokens, retrieval delays) — Part 4.
- Turn-aware / overlapping windowing.
- Serving the resulting adapter on moshika-rag (Part 3).

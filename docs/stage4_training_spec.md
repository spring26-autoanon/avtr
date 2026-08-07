# Stage 4 training spec — everything needed to run tomorrow

Companion to `docs/superpowers/specs/2026-08-07-stage4-preserve-and-personalize-design.md`
(the *why*). This document is the *what and how*: every hyperparameter, its source, the data
pipeline, the code changes, and the order to execute in.

---

## 0. Sources of truth

| Source | What it governs |
|---|---|
| `base-moshi-paper.pdf` Table 1 | Moshi's own fine-tuning phases — LR, batch, steps, acoustic delay |
| `moshi-rag-paper.pdf` §3–4 | retrieval mechanism, lead/body/tail, `⟨ret⟩` placement, delay sampling (Eq. 3), reference dropout, data scale |
| `README.md` | moshi-finetune's own recommended LoRA settings |
| Our measurements | runway, marker density, stall causes, base-model control |

### Hyperparameters across all three sources

| | Moshi paper (Fisher / instruct FT) | moshi-rag paper | README | Stage 3 (ours) |
|---|---|---|---|---|
| Learning rate | **2e-6** | **2e-6** | **2e-6** | 4e-6 ❌ |
| Batch | 40 min / 2.7 h audio | 32 | 16 | 8 |
| Steps | 10k / 30k | 100k | 2000 | 800 |
| LR schedule | cosine | — | OneCycle (`pct_start`) | OneCycle |
| Optimizer | AdamW, wd 0.1 | — | AdamW, wd 0.1 | AdamW, wd 0.1 |
| Trainable | all | all except ref encoder | LoRA ≤128 | LoRA rank 64 |
| Frame rate | 12.5 Hz | 12.5 Hz | — | 12.5 Hz |
| Acoustic delay | 1 | — | (loader) | (loader) |

**Three independent sources say 2e-6 for fine-tuning. Every run we have done used 4e-6.**
That is the single most defensible change available, and it costs nothing.

Our batch of 16 × 100 s = 0.44 h of audio, against Moshi's 2.7 h for instruct fine-tuning —
6× smaller, and not closable on one GPU. Worth knowing when reading loss curves.

---

## 1. Track A — maximum voice clone (plain moshika)

**Objective:** the best-sounding Danielle achievable. No retrieval to protect, so maximise
adaptation. This is the safe competition deliverable.

**Config:** `example/moshika_voice_max.yaml`

| param | value | source |
|---|---|---|
| `hf_repo_id` | `kyutai/moshika-pytorch-bf16` | plain moshika, female voice |
| `config_path` | **omitted** | plain moshika ships no `config.json`; setting it loads the wrong architecture |
| `lora.rank` | 128 | README recommended |
| `lora.scaling` | 2.0 | README |
| `lora.ft_embed` | false | README default |
| `batch_size` | 16 | README |
| `max_steps` | 2000 | README |
| `duration_sec` | 100 | README |
| `optim.lr` | **2.0e-6** | all three sources |
| `optim.weight_decay` | 0.1 | README |
| `optim.pct_start` | 0.05 | README |
| `first_codebook_weight_multiplier` | 100.0 | repo default — first codebook is semantic |
| `text_padding_weight` | 0.5 | repo default |
| `ckpt_freq` | 200 | 10 checkpoints |
| `do_eval` | false | eval set is 6 windows < batch 16 → NaN |

**Data:** dialogue + **marker-stripped** retrieval clips + the new session.

> ⚠️ `interleaver._tokenize` maps `"<RAG>"` to raw token id 4. That is the rag token on
> moshika-rag but an **ordinary token on plain moshika** — marked clips would inject
> nonsense. Use `scripts/strip_rag_markers.py` output. Already generated locally:
> `replay/retrieval_nomarkers/`, 78 clips, 72.3 min, 0 markers, no reference tensors.

**Expected memory:** Stage 3 peaked at 26.4 GB with rank 64 / batch 8. Rank 128 / batch 16
roughly doubles both — budget **50–60 GB of 80**. Watch the first ten steps; halve
`batch_size` if it OOMs.

**Runtime:** ~15 s/step at batch 8; batch 16 will be slower per step but fewer tokens wasted.
Budget **5–7 hours** for 2000 steps. Start this one first.

---

## 2. Track B — RAG, sequenced

**Do not change more than one thing per run.** Stage 3's lesson: a checkpoint was called a
regression from a single session and the call was wrong.

### Run 4a — mask the negative + delay fix + reference dropout

Three changes, but they address two *different* failures and neither substitutes for the
other, so they ship together.

**(i) Mask `⟨ret⟩` from the text loss on unlabelled windows — fixes TRIGGERING.**

`loss.py` is plain cross-entropy over the text stream. On any window whose target never
contains token 4, cross-entropy pushes **P(⟨ret⟩) down at every position**. We have 135
markers against thousands of marker-free windows, so the dominant gradient is "never emit
this token." `RAG_TOKEN_WEIGHT=25` upweights the 135 positives against that much larger
force — which is exactly why it moved triggering from 0% to ~50% and stalled there.

The fix follows from what we actually know:

- **Labelled clip** — ground truth is known everywhere: fire at the marker, nowhere else.
  Train normally, positives and negatives.
- **Unlabelled window** — we never verified whether retrieval was warranted. Asserting
  "don't retrieve here" is an unsupported claim. Say nothing instead.

```
if this example has no reference tensor:
    logits[..., rag_token_id] = -inf      # before cross_entropy
```

Masking to `-inf` drops the token from the softmax denominator, so no gradient pushes it
down. Per-example, keyed on `batch.reference_tensors[b] is None`, which train.py already has.

**This largely subsumes the dialogue-mining work (§3.3).** If marker-free windows no longer
suppress, marker density stops being the binding constraint — all 135 min of dialogue stays
in for voice and persona at no cost to triggering, and mining becomes optional polish.

⚠️ Masking also removes the only signal teaching the model *not* to over-trigger. Base
moshika-rag already over-fires (it retrieved on "what's your favourite colour"). The real
recorded declines are the counterweight; watch criterion 8 in the audition.

**(ii) Delay + (iii) dropout — fix GROUNDING**, not triggering. Detailed in §4.1.

Everything else held at Stage 3 values (rank 64, batch 8, 800 steps, lr 4e-6) so the run
still isolates *these* changes from the hyperparameter question, which is 4b's job.

**Config:** `example/moshika_rag_stage4a.yaml`. Every Stage 3 hyperparameter held (rank 64,
batch 8, 800 steps, lr 4e-6) so the run isolates the injection change.

| param | value | source |
|---|---|---|
| `hf_repo_id` | `kyutai/moshika-rag-pytorch-bf16` | |
| `config_path` | `checkpoints/moshika-rag/config.spike.json` | `reference_with_time` in the fuser, not a module — avoids the ARC meta-crash |
| `lora.rank` / `scaling` | 64 / 2.0 | unchanged from Stage 3 |
| `batch_size` / `max_steps` | 8 / 800 | unchanged |
| `optim.lr` | 4.0e-6 | **deliberately unchanged** so 4a is single-variable |
| `RAG_TOKEN_WEIGHT` | **25** (env, not config) | 15 never fired; 25 fires ~50% |
| delay `d'` | Eq. 3, per example | moshi-rag paper |
| reference dropout | 0.2 | moshi-rag paper |

### Run 4b — minimal intervention

**Config:** `example/moshika_rag_stage4b.yaml`. Only after 4a is auditioned.

| param | 4a | 4b | why |
|---|---|---|---|
| `lora.rank` | 64 | **32** | one halving from the only evidenced value |
| `max_steps` | 800 | **400** | voice imprints by ~300; damage accrues after |
| `batch_size` | 8 | **16** | headroom exists; README uses 16 |
| `optim.lr` | 4.0e-6 | **2.0e-6** | all three sources |
| `ckpt_freq` | 100 | 50 | same checkpoint granularity |

⚠️ Total adaptation (≈ lr × steps) is **~1/4 of Stage 3**. If the voice fails to imprint,
relax that before touching rank: try 4e-6 × 400, or 2e-6 × 800.

### Run 4c — data ratio

**Only run this if 4a's masking fails to restore triggering.** With the negative masked,
marker density stops being the binding constraint, and every option below costs voice data
to buy something masking already bought. Options in preference order:

1. **Mined dialogue with markers** — keeps all 135 min of voice *and* raises marker density.
2. **0.8 retrieval / 0.2 dialogue** — cuts the marker-free signal 60% without discarding the
   voice or the persona content in the dialogue.
3. **Retrieval-only** — cleanest marker density, but discards Clay (the only second
   conversation partner) and the persona-rich casual turns. Highest cost to voice.

---

## 3. Data pipeline

### 3.1 New recording (2026-08-07 evening)

Brief: `docs/recording_brief_danielle.md`. Expect ~35–40 min: persona (~20), declines (~10),
interruption/overlap (~5–10). Isolated tracks, her on channel 0, 24 kHz, PCM_24.

Same pipeline as the retrieval master:

```bash
# Mac
python3 scripts/pair_dialogue_stereo.py \
  --src <new recording dir> --dst finetune/data/prepared_persona \
  --main-name danielle --no-eval

rsync -avP finetune/data/prepared_persona/*.wav \
  wb-gpu-a1ultra2g:~/moshi-finetune/finetune/data/prepared_persona/

# box — both channels
~/fork-venv/bin/python scripts/build_manifest.py \
  --wav-dir finetune/data/prepared_persona --out-dir finetune/data/prepared_persona --no-eval

CUDA_VISIBLE_DEVICES=0 setsid nohup ~/fork-venv/bin/python annotate.py \
  finetune/data/prepared_persona/all.jsonl -l --whisper_model medium --lang en \
  --channel 0 --out-suffix .json > ~/annotate_persona_ch0.log 2>&1 &
# wait, then --channel 1 --out-suffix .ch1.json

# Mac — segment, filter, cut (now with lead/body labelling)
source .env
python3 scripts/segment_retrieval_audio.py --ch0 ... --ch1 ... --out replay/persona_segments.jsonl
python3 scripts/filter_retrieval_segments.py --in ... --out ... --dropped ...
python3 scripts/cut_retrieval_clips.py --master ... --alignments ... --segments ... \
  --outdir replay/persona --manifest replay/persona_manifest.jsonl
```

### 3.2 Existing data — disposition

Which directory goes into which run. **Getting this wrong is silent** — nothing crashes, the
run just trains on the wrong thing for five hours.

| directory | what it is | min | markers | Track A | Track B |
|---|---|---|---|---|---|
| `finetune/data/prepared_dialogue/` | 4 stereo, natural dialogue (Joshua + Clay) | 135.3 | 0 | ✅ | ✅ unmarked |
| `replay/retrieval/` | 78 clips, `<RAG>` + refs, 3 s tails | 72.3 | 119 | ❌ | ✅ |
| `replay/retrieval_nomarkers/` | same audio, markers stripped, no refs | 72.3 | 0 | ✅ | ❌ |
| synthetic declines (in `replay/retrieval/`) | Stage-2b, marked + ref | 3.2 | 16 | ❌ | ✅ retire once real declines land |
| `finetune/data/prepared/` | 9 monologues, **silent right channel** | 57.8 | 0 | ❌ | ❌ |
| new recording (tonight) | persona / declines / overlap | ~35–40 | TBD | ✅ all | split, see below |

**Three traps:**

1. **`finetune/data/prepared/` must never be used again.** Those are the original Option-A
   monologues with a silent user channel — the data that taught the model the user never
   speaks and produced the monologuing failure. Still on disk; just keep it out of every
   manifest.
2. **The two retrieval directories are the same 72 minutes and are not interchangeable.**
   `interleaver._tokenize` maps `<RAG>` to raw token id 4 — the rag token on moshika-rag, an
   ordinary word token on plain moshika. Marked clips on Track A inject nonsense into the
   text stream; stripped clips on Track B have no reference to condition on.
3. **The dialogue goes into Track B unmarked, and that is now correct** rather than merely
   tolerable — see §4.1b. Before the loss mask it was 135 min of trigger suppression.

**Splitting tonight's recording when you cut it:**

| part | ~min | markers | why |
|---|---|---|---|
| 1 — persona | 20 | **none** | she is not retrieving her own name. Pure voice + identity; masked by §4.1b at zero cost to the trigger |
| 2 — declines | 10 | **marked + reference** | the retrieve-then-find-nothing case. With the diffuse negative signal masked away, these are now the **main** counterweight against over-triggering — check the count after cutting |
| 3 — interruption/overlap | 5–10 | none | turn-taking and barge-in |

### 3.2b Launch commands

```bash
# Track A — plain moshika. No env vars: token 4 is an ordinary token here.
uv run torchrun --nproc-per-node 1 -m train example/moshika_voice_max.yaml

# Track B run 4a — all four switches. Every one defaults to OFF.
RAG_TOKEN_WEIGHT=25 MASK_UNLABELLED_RAG=1 RAG_DELAY=1 RAG_REF_DROPOUT=0.2 \
  uv run torchrun --nproc-per-node 1 -m train example/moshika_rag_stage4a.yaml
```

| env var | run 4a | what it does |
|---|---|---|
| `RAG_TOKEN_WEIGHT` | 25 | upweight the ⟨ret⟩ positives (Stage 3, unchanged) |
| `MASK_UNLABELLED_RAG` | 1 | drop the ⟨ret⟩ logit from the loss on reference-free examples (§4.1b) |
| `RAG_DELAY` | 1 | inject at `hit + d'` per Eq. 3 instead of at the hit (§4.1) |
| `RAG_REF_DROPOUT` | 0.2 | skip injection for a turn; the marker itself stays trainable |

⚠️ **All four default to off**, so a forgotten variable silently reproduces Stage 3 for three
hours. Check the launch line before walking away. They are env vars rather than config keys
because the configs are committed and these change per run — the same reason
`RAG_TOKEN_WEIGHT` was one in Stage 3.

Delay and dropout draw from a `random.Random(args.seed)` stream created in `_train`, separate
from the global RNG, so toggling them does not shift data order or init and runs stay
comparable.

### 3.3 Mine the natural dialogue — keep it, don't discard it

**Decision made: the dialogue stays in training.** It is 135 minutes of her voice and — as
the transcripts show — it is **persona-rich**: 109 turns of 30+ words covering her music
taste, her job, Tampa, her cat, her opinions. Discarding it would throw away the best answer
to the "I'm just a chatbot" problem. The only issue is that it is marker-free and therefore
suppresses the trigger, so the fix is to **add markers where they belong**, matching the
density moshi-rag was trained with.

**Step 1 — count (local, no GPU, do first).** Channel-0 transcripts are already local.

```bash
# Gemini scans her 639 turns and flags the ones that state external facts
# or describe herself; count them.
```

Decision rule: **≥150 markable → do the full labelling.** 50–150 → mark only the strongest
and reweight to 0.8/0.2. **<50 → skip mining**, use option 2 or 3 from §2.

**Step 2 — channel 1 (GPU, only if step 1 justifies it).** The dialogue files have only
channel 0 annotated. The segmenter needs Joshua's/Clay's side to know what question a turn
answers.

```bash
for f in danielle_clays danielle_joshuarhodes_train_a danielle_joshuarhodes_train_b; do
  CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python annotate.py \
    finetune/data/prepared_dialogue/all.jsonl -l --whisper_model medium --lang en \
    --channel 1 --out-suffix .ch1.json
done
```

**Step 3 — segment and label** with the same pipeline, then cut with 3 s tails.

⚠️ **Bar for marking is higher here than for the recorded clips.** Marking a turn that is not
genuinely knowledge-dependent trains spurious triggering — the base model's existing flaw.
When in doubt, leave it unmarked.

### 3.4 Lead/body/tail labelling — required for the delay fix

Eq. 3 samples `d'` bounded by `d_lead`, so every retrieval turn needs to know where its lead
ends. Extend `segment_retrieval_audio.py`: Gemini already sees her turn text and the
reference it wrote — ask additionally for the index of the first grounded word. Word
timings then give `d_lead`.

**Measured baseline:** her median gap to content is **0.73 s** in casual conversation and
**0.43 s** in Q&A. Retrieval takes **1.7–3.4 s**. Only ~30% of turns leave ≥2 s using the
reference-word method. The delay fix cannot ground a turn whose lead is shorter than the
lookup — hence tomorrow's "start general, land specific" instruction.

---

## 4. Code changes

### 4.1 `finetune/data/reference_injection.py` — delay + dropout

Currently `start = hits[i]`. Change to:

```
start = hits[i] + d'          # d' in frames at 12.5 Hz
```

with `d'` sampled per example per Eq. 3:

```
d' = U(0, d_lead)             if d_lead < 2.0s  or  p < 0.2
     U(1.0, d_lead - 1.0)     otherwise           , p ~ U(0,1)
```

Requirements:
- Clamp each span at the next `⟨ret⟩` and at sequence end (existing behaviour).
- A delay exceeding the remaining answer means **no injection** — a valid "retrieval arrived
  too late" training case, not an error.
- **20% dropout**: skip injection entirely. The paper uses a learnable `h_dropout`, which a
  LoRA cannot add; omission is the approximation.
- `d_lead` comes from the manifest, per retrieval turn.
- Keep it deterministic under a seed for reproducibility.

**⚠️ Measured limitation — the delay barely engages on the current corpus.** Eq. 3 bounds
`d'` by `d_lead`, and her measured leads are 0.43-1.2 s, so the sampled delay is 0.2-0.6 s
against a real serve latency of 1.7-3.4 s. Simulated over the 119 real markers:

| d_lead | mean d' | % of draws >= 1.7 s |
|---|---|---|
| 0.43 s (her Q&A median) | 0.21 s | **0%** |
| 1.20 s (her overall median, our default) | 0.60 s | **0%** |
| 3.0 s | 1.50 s | 32% |
| 4.0 s | 2.00 s | 63% |

So the delay fix and the recording brief are **the same fix from two directions**: the code
can only train delays as long as the runway in the audio. The brief now asks for ~3 s of
general talk before the specific fact, with this reasoning attached. Until longer leads
exist, expect 4a's delay contribution to be small — do not read a null result as the delay
being wrong. The answer windows are not the constraint: median 18.7 s after each marker
(p10 6.3 s), so nothing is being suppressed for lack of room.

**Status: DONE** (`f518674`, `eaa2200`). `sample_delay_sec(d_lead, rng)` implements Eq. 3;
`build_reference_condition` gained `sample_delay`, `dropout`, `leads`, `default_lead`, `rng`.
15 tests in `tests/test_reference_delay.py`, 171 in the suite.

Two things to know:

- **`d_lead` falls back to `DEFAULT_LEAD_SEC = 1.2`** — her measured median — for any turn the
  manifest hasn't labelled. So the delay works *now*, before §4.2 lead labelling exists, and
  sharpens once it does. At a 1.2 s lead the delay is U(0, 1.2), mean 0.6 s ≈ 7 frames.
- **Dropout is per retrieval turn, not per example**, which gives more independent samples
  from a small corpus. A dropped turn keeps its ⟨ret⟩ in the text stream, so the trigger is
  still trained; only the conditioning is withheld. The paper's learnable `h_dropout` is the
  part a LoRA cannot add — omission is the approximation, as noted above.

### 4.1b `finetune/loss.py` — mask the rag token on unlabelled windows

Add a per-example boolean (`has_reference`) to `compute_loss_with_mask`. Where it is False,
set the rag-token logit to `-inf` before `F.cross_entropy`, so the token is excluded from the
softmax and receives no downward gradient.

Interaction with `rag_token_weight`: the 25× upweight still applies on labelled examples. It
may be able to come *down* once the negative pressure is gone — a cheap follow-up ablation,
not a change to make in the same run.

Tests: masked example contributes no gradient at the rag logit; unmasked example is
unchanged; a labelled example still trains both the marker position and the non-marker
positions; the mask is per-example, not per-batch.

### 4.2 `scripts/segment_retrieval_audio.py` — lead labelling

Add `lead_end_word_index` per grounded/decline turn to the schema and to `normalize`.
`cut_retrieval_clips.py` then writes `d_lead` (seconds) into the manifest.

### 4.3 `scripts/apply_rag_positioning.py`

Must be updated in lockstep, or a fresh checkout silently reverts to zero-delay. Verified
by the existing "patcher is inert / emits byte-identical code" check.

### 4.4 Audition harness (new) — build before running anything

Replays a fixed question list and reports numbers instead of impressions:

- `⟨ret⟩` fires / factual questions asked
- whether the answer contains the reference's key facts
- longest empty-buffer gap (stall)
- seconds from turn start to first content word (lead), vs her 0.43–0.73 s baseline
- `batched step` timings, to catch GPU contention masquerading as a model fault

Stage 3 lost hours to session variance. This makes each comparison minutes instead of a night.

---

## 5. Serving

```bash
# encoder — GPU 1, own tab, leave running
CUDA_VISIBLE_DEVICES=1 ~/fork-venv/bin/python -m moshi.server_conditioner \
  --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \
  --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
  --conditioner reference_with_time --cuda-device 0 --port 8001

# RAG server — GPU 0
source ~/.moshi_env
export REFERENCE_ENCODER_URL=http://localhost:8001    # REQUIRED: also selects the 676-slot
                                                      # serve model, matching the raw adapter
MOSHI_LORA_SCALING=2.0 \
MOSHI_LORA_WEIGHT=<ckpt>/consolidated/lora.safetensors \
CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python -m moshi.server \
  --hf-repo kyutai/moshika-rag-pytorch-bf16 --stt-wait-time 2.0 --port 8998 2>&1 | tee ~/serve.log

# Track A (plain moshika) — no encoder, no REFERENCE_ENCODER_URL
MOSHI_LORA_WEIGHT=<ckpt>/consolidated/lora.safetensors \
CUDA_VISIBLE_DEVICES=1 ~/fork-venv/bin/python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 --stt-wait-time 2.0 --port 8999 2>&1 | tee ~/serveA.log
```

**Non-obvious, all learned the hard way:**

- Use the **raw** `lora.safetensors`, not `lora_padded`. `REFERENCE_ENCODER_URL` selects
  `skip_conditioners=["reference_with_time"]` → 676 slots, which a Path-A adapter already
  fills exactly.
- **Serve on an idle GPU.** Sharing with training pushed steps 80 ms → 110 ms, starving the
  audio buffer and producing crackle that mimics a model fault.
- **Drop `--gradium-stt`.** The hosted STT drops the session after 35–75 s; local DSM does not.
- **`MOSHI_LORA_SCALING=2.0`.** 1.5 improved triggering but cost filler and prosody, and broke
  grounding by removing the delay cover. Not the lever.
- **Kill the server between checkpoints** — it holds ~43 GB and does not release it.
- A "maximum capacity" page has three times been a **stale service worker**, not real
  capacity. Check `grep -c "acquired slot"` in the log; zero means the browser never reached
  the server. Use a private window.
- Persona template lives at
  `~/fork-venv/lib/python3.12/site-packages/moshi/llm/reference_prompt_template_simplified.txt`
  (source of truth: `serving/`). Loaded at startup — restart the server after editing.

---

## 6. Audition protocol

Same script every time, three sessions per checkpoint, because triggering is stochastic.

1. "Hi, how are you?"
2. "What's your name?"
3. "What do you do for work?"
4. "Can you tell me about the World Cup?" *(vague — trial 4 failed this)*
5. "Who won the first World Cup?" *(specific)*
6. "What's the capital of Australia?" *(outside the training corpus)*
7. "What's the weather in Dublin right now?" *(should decline)*
8. "Do you like playing soccer?" *(personal — should NOT trigger)*

Then:

```bash
grep -cE "RAG. model emitted" ~/serve.log      # triggers
grep "Generated reference" ~/serve.log          # what was surfaced
grep -c "LM buffer empty" ~/serve.log           # stall pressure
```

**Pass criteria**, judged against the base-model control rather than against Stage 3:

| # | criterion | base | Stage 3 | target |
|---|---|---|---|---|
| 1 | triggers on factual Qs | 8/8 | ~4/8 | ≥6/8 |
| 2 | answer matches the reference | yes | yes when it fires | yes |
| 3 | lead before the fact | — | ~0.4 s | ≥1.5 s |
| 4 | longest stall | none | 2–18 s | <3 s |
| 5 | yields when interrupted | yes | untested | yes |
| 6 | says Danielle / the job | n/a | "I'm an AI chatbot" | correct |
| 7 | sounds like her, with filler | n/a | yes at scaling 2.0 | yes |

**3 and 7 must not be traded away for 1 and 2.** That trade is what scaling 1.5 made, and it
was the wrong one.

---

## 7. Execution order for tomorrow

**Before the recording** (local, no GPU):
1. ~~Loss mask (§4.1b), delay + dropout (§4.1)~~ — **DONE**, `63caa2d` / `f518674` /
   `eaa2200`, 171 tests pass. Lead labelling (§4.2) is still open but no longer blocking:
   the delay falls back to the measured 1.2 s median.
2. Build the audition harness (§4.4).
3. Dialogue mining count (§3.3 step 1) — now *optional*, and only informs 4c. Skip it if
   time is short; masking is the primary fix for the same failure.

**Launch early, it is the long pole:**
4. **Track A** on the marker-stripped clips + dialogue — 5–7 hours, and it is the safe
   deliverable. Does not depend on any of the code changes above.

**After the recording lands (evening):**
5. Pair → upload → annotate both channels → segment → cut the persona/decline audio.
6. Fold into the retrieval manifest; re-run `precompute_references` and the gate.
7. **Run 4a** (~3 h) — loss mask + delay fix + dropout.
8. Audition 4a with the harness. Only then decide on 4b and 4c.

**Standing rules:** back up checkpoints to the Mac as they land; stop the box when idle; keep
the `:8001` encoder up for the whole session.

# Demo Sprint Design — 2026-08-07 → Sunday 2026-08-10

Three days to two live demos: **Track A** (plain moshika + voice LoRA — the safe, best-voice
deliverable) and **Track B** (moshika-rag + voice LoRA — her voice *and* retrieval). Approach
chosen: **everything launches tonight** (Track A + runs 4a and 4b in parallel), because a run
done tonight buys a full extra iteration day, and worst case there is a Saturday-night slot to
run again.

Companion documents: `docs/HANDOFF-2026-08-07.md` (state of the world),
`docs/stage4_training_spec.md` (hyperparameters, §-references below),
`docs/superpowers/plans/2026-08-07-tonight-recording-pipeline.md` (Tasks 1–5, already written).

## Decisions of record (from this session)

1. **Best checkpoint wins.** Whatever 4a/4b checkpoint audits best gets demoed, with the demo
   script drawn from its known-good question list. Checkpoint selection is weighted **voice /
   persona fidelity and absence of stalls first, retrieval trigger rate second.**
2. **Demo runs live from the GPU boxes, with a backup screen-capture video of each demo
   recorded Saturday.** Boxes stay up on the jupyter keep-alive pattern from tonight through
   the demo — stockout insurance outweighs idle cost this week.
3. **The user runs every GPU command** (standing rule). All box steps are paste-able handoffs.
4. **Tonight's recording is sent raw.** No noise reduction, normalization, limiting, silence
   trimming, fades, or concatenation — see "Recording intake" below.

## Findings from the papers/logs sweep (what we'd missed)

**F1 — We serve past the paper's retrieval-delay cliff.** moshi-rag paper Fig. 6b: response
accuracy collapses once retrieval delay exceeds **1.5 s**. Our measured round trip is
**1.7–3.4 s** — entirely past the cliff. The paper waits only **0.5 s** for ASR after ⟨ret⟩;
we serve with `--stt-wait-time 2.0`. Saturday serving work: instrument the round trip in
serve.log, then step `--stt-wait-time` 2.0 → 1.0 → 0.5 and re-audition. Attacks grounding
failures and perceived timeouts **with no retraining**.

**F2 — Trigger rate tracks user-speech intelligibility.** Fig. 6a: firing declines predictably
as user WER rises. Part of Stage 3's "stochastic firing" is likely mic/intelligibility
variance. Consequences: audition uses fixed phrasings, same mic, quiet room; demo uses a
directional/headset mic.

**F3 — Noise robustness was trained into base Moshi and our fine-tune erodes it.** Base-moshi
paper §4.4: instruct fine-tuning augments the **user's channel** with random gain (−24 to
+15 dB, 50%), added background noise (30%), simulated echo of Moshi's own voice into the user
mic (scale 0–0.2, delay 100–500 ms), and reverb. Our LoRA trains on pristine channels — the
likely source of "easily distracted by noise" being *worse* than base moshika-rag.
Consequences: (i) demo with headphones — echo robustness is trained-in behaviour we dilute;
(ii) identified Saturday-night lever: user-channel gain+noise augmentation in the data loader,
gated by env var (e.g. `NOISE_AUG=1`), copying the paper's recipe. Not on the critical path.

**F4 — Interruption vs. distraction are separable behaviours** (Full-Duplex-Bench, Table 11):
respond to genuine interruption; resume through backchannels/background speech. Taught by the
paper's v2/v3 subsets; tonight's Parts 3 and 6 are our equivalents. **Part 3 upgraded from
"if time" to protected** (Danielle notified by text; cut order is now 6, then 7).

**F5 — A demoable Track B floor already exists.** Stage 3 run 2 ckpts 400/500 (her voice,
grounded answers when it fires ~50%, best-session stalls 2.1 s) and `runs_backup/`
(`ragw25_000600` — reliable retrieval, formal voice). Sunday can never be empty-handed.

**F6 — Confirmations:** ⟨ret⟩ belongs before the first lead word (our cutter matches); the
paper's pre-RAG filler = the brief's "start general"; the paper drops greetings 30% of the
time for user-first robustness — Part 4 openings fill the same hole from the data side.

## Recording intake (new constraints discovered)

The session is recorded on **Zoom** with per-participant audio files, and past deliveries were
post-processed. For tonight:

- **Raw files only.** Her cleaning pipeline (noise reduction, normalize, peak limiting,
  silence shortening, edge fades, concatenation with 0.7 s gaps) is banned for this use:
  denoising artifacts contaminate the voice; normalize/limiting flatten the emotional dynamics
  the brief asks for; **silence trimming and take-concatenation destroy cross-track timing** —
  `pair_dialogue_stereo.py` assumes sample-aligned tracks, and pauses/runway are the timing
  signal itself. Benign steps (DC offset, 70 Hz high-pass, 24 kHz resample) are unnecessary —
  the pipeline handles format.
- **Zoom settings:** separate audio file per participant ON; both wear **headphones** (stops
  Zoom's echo cancellation ducking mics during overlap — Part 3 depends on this); "Original
  sound" / noise suppression OFF if available.
- **Pre-session test take:** 60 s of deliberate talking-over, confirm both voices survive on
  both tracks before recording Part 1.
- **Open question to answer during intake:** were the previous sessions silence-trimmed? If
  yes, the measured 0.43–1.2 s leads are partly an editing artifact, her natural runway is
  longer than we think, and tonight's raw audio helps the delay fix more than projected.
  Record the answer in the experiments log either way.

## Day-by-day

### Today (Thu) — before the recording (local, ~3–4 h)

1. Text Danielle: raw-files instruction + Part 3 protected (drafted, sent by the user).
2. **Persona guard** — Task 1 of the tonight plan, TDD, ~30 min. Hard blocker for Pass B:
   without it Gemini labels her ~45 identity answers `grounded` and marks them.
3. **Audition harness** (§ below) — the rest of the afternoon.
4. **User provisions `wb-gpu-training`** in parallel (handoff blocks): fork venv
   moshi 0.2.13, HF login, moshika download, rsync `prepared_dialogue` +
   `retrieval_nomarkers` wavs/transcripts/manifests.

### Tonight — recording lands → overnight

Two-pass pipeline per the existing plan (Tasks 2–5), with ordering that matters:

channel-isolation check **before anything expensive** (corr ≈ 0; > 0.5 → stop, re-export) →
pair → rsync to both boxes → annotate ch0 then ch1 (ultra GPU1) → **Track A launches on
wb-gpu-training as soon as ch0 transcripts exist** (needs only Pass A) → segment / filter /
cut locally → `turn_mix` sanity check (persona ≫ 30, declines 15–20; persona ≈ 0 → guard
failed, stop) → encoder :8001 on GPU1 → precompute → **verify gate: marker count = tensor
count, non-negotiable** → wire configs (0.35 / 0.15 / 0.25 / 0.25, tests enforce Σ = 1.0) →
launch 4a (GPU0) → kill encoder → launch 4b (GPU1).

| Box / GPU | Run | Launch line | Duration |
|---|---|---|---|
| wb-gpu-training | Track A `moshika_voice_max` | no env vars | 5–7 h |
| ultra GPU0 | 4a | `RAG_TOKEN_WEIGHT=25 MASK_UNLABELLED_RAG=1 RAG_DELAY=1 RAG_REF_DROPOUT=0.2` | ~3 h |
| ultra GPU1 | 4b (after precompute) | same four vars | ~3 h |

All in jupyter keep-alive; watch the first 20 steps (rank 128 × batch 16 is untested memory —
budget 50–60 GB, halve batch on OOM); checkpoints rsync to the Mac as they land. If the
recording slips past ~1 a.m.: push 4b to morning; never skip the verify gate.

### Friday — audition day

Coarse pass first (Track A 1200/1600/2000; 4a 400/600/800; 4b 200/300/400), refine around the
winner. Generate each winner's known-good question list. **Decision gate ~6 pm:**
Track B satisfactory → done training; otherwise Friday-night contingency chosen from evidence:

- voice under-imprinted → 4b-relaxed (2e-6 × 800; spec §2 warns 4b totals ~¼ of Stage 3
  adaptation)
- triggering still weak → 4c data-ratio (spec §2 preference order)
- noise-distraction dominant → F3 augmentation run

### Saturday — lock and rehearse

Audition any overnight run by noon → **lock both checkpoints**. Afternoon: serving
optimization (F1 round-trip instrumentation, stt-wait step-down, scaling 2.0 confirmed,
kill-server-between-checkpoints); demo script from known-good lists; **record backup videos of
both demos**; full dress rehearsal on demo hardware (headphones, directional mic, quiet room,
private browser window — the "maximum capacity" page has three times been a stale service
worker). Evening: optional last run **only if the audition says it's needed** (F5 means it is
never mandatory). Boxes stay up overnight.

### Sunday — demo

Servers up early: Track B = encoder GPU1 + RAG server GPU0 on the ultra box; Track A server on
wb-gpu-training. One harness pass as smoke test. Demo from the rehearsed script; videos on
standby.

## Audition harness (expands spec §4.4)

- **Fixed script:** the 8 questions of spec §6 + persona questions matching Part 1 topics
  (name, fighting, plants, tattoos, work), fixed phrasings, 3 sessions per checkpoint.
- **Per-checkpoint scores** from serve.log + session notes: trigger rate,
  answer-matches-reference, longest stall, lead before first fact, `batched step` timing
  (GPU contention detector), **retrieval round-trip time** (new, F1), voice/persona 1–5.
- **Verdict weighting matches decision 1:** stalls + voice/persona first, trigger rate second.
- **Direct demo artifact:** per-checkpoint `known_good.md` — questions that fired and
  grounded in ≥ 2 of 3 sessions (demo script), persona questions answered in character,
  known-bad list the demo must avoid.
- Implementation: `scripts/audition_checkpoint.py` parses serve.log and emits the scorecard;
  a checklist doc drives the live sessions. The user runs sessions; the script turns each into
  numbers in minutes.

## Serving & demo engineering (consolidated)

Known: raw `lora.safetensors` (never `lora_padded`); `REFERENCE_ENCODER_URL` required (selects
the 676-slot serve model); serve on an idle GPU; local DSM STT — no `--gradium-stt` (hosted
drops sessions at 35–75 s — plausibly some of the observed "timeouts");
`MOSHI_LORA_SCALING=2.0` (1.5 traded voice for triggering and broke grounding — wrong trade);
kill the server between checkpoints (~43 GB held). New from the sweep: stt-wait step-down +
round-trip instrumentation (F1); headphones + directional mic + quiet room (F2/F3). Persona
template in `serving/` reviewed after tonight's data lands — with identity in the weights it
becomes backstop, not load-bearer.

## Risks and fallbacks

| Risk | Mitigation |
|---|---|
| GPU stockout | Boxes stay up (keep-alive) from tonight; checkpoints rsynced to Mac; backup videos Saturday |
| Recording slips or fails channel check | Track A still trains tonight on existing 135 min + nomarkers; Pass B waits |
| Zoom processing damages overlap audio | Headphones + original-sound + 60 s pre-session test take |
| 4a and 4b both disappoint | Friday-night contingency slot, evidence-picked; Saturday-night slot behind it |
| wb-gpu-training won't provision | All three runs sequence on the ultra box: Track A GPU0 overnight, 4a GPU1, 4b Friday morning |
| Absolute worst case | Track A voice demo + Stage-3 run-2 ckpt 400 curated-question demo (F5) + videos |

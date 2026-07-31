# Path A Stage 2/3 — Synthetic RAG replay data (restore retrieval)

_Design doc — 2026-07-31. Follows Path A Stage 0 (trainer unblocked on moshika-rag)
and Stage 1 (voice-only, VALIDATED). Read alongside `NEXT_STEPS.md` Part 4._

## Problem

Path A Stage 1 (voice-only training on moshika-rag) gave a clean Danielle voice with
no screech — but **killed retrieval**: the model no longer emits the ⟨ret⟩/rag_token
and answers from its own head (confirmed live — the World Cup question produced no
`[RAG] model emitted RAG token` and no `[Remote Encoder]` call). That's the expected
"retrieval floor" from training with no retrieval in the loss. This spec restores
retrieval by **mixing synthetic RAG replay data into training** so the model keeps the
"emit ⟨ret⟩ → speak from the reference" behavior, while keeping the voice.

## Goal & target

- **Target = general assistant.** At serve, retrieval = **Gemini** answering from its
  own knowledge, shaped by the reference prompt. Replay teaches a **domain-agnostic
  behavior** ("use the reference"), which generalizes — so we do **not** need
  exhaustive knowledge coverage; a diverse-enough batch teaches the behavior and Gemini
  supplies the facts at serve. Start with a small diverse batch to prove the mechanism,
  then scale diversity.
- **Deliverable:** a Stage-3 adapter (voice-dialogue + replay mix, on moshika-rag) that
  is **Danielle's voice, takes turns, no screech, AND retrieves+answers** (emits ⟨ret⟩,
  uses the Gemini/ARC reference).

## The two make-or-break unknowns → de-risking spikes (gate everything)

Both are cheap and must pass **before** generating a large (TTS-expensive) dataset.

1. **TTS spike — render her voice from given text.** Can we drive `checkpoint_700`
   (our conversational voice adapter) as a Moshi-style TTS by **forcing its
   inner-monologue text stream** to a target transcript and getting clean Danielle
   audio? (Base Moshi rendered its synthetic instruct data with a Moshi-family TTS in
   the target voice — same idea.) Investigate whether the fork exposes forced-text
   generation (e.g. `moshi/run_inference.py`).
   - **Pass:** clean, her-voice audio for arbitrary text → use it for assistant turns.
   - **Fail:** fall back to **Gradium voice cloning** for her turns (turnkey; downside
     = a 3rd "Danielle" rendition that may slightly blur the voice vs the real dialogue
     data).
   - Partner (user) turns use a simple/varied TTS (Gradium default voices) either way.

2. **Conditioning-injection spike — feed a precomputed reference tensor into a training
   step (THE highest risk).** For replay to teach "use the reference," the model must
   *receive* the `reference_with_time` conditioning during training, matching serve.
   Approach = **inject a precomputed tensor** (NEXT_STEPS Part 4; what serve already
   does), NOT compute it in the trainer (which would need the ARC encoder un-stripped =
   meta crash + Llama-3.2 per step).
   - Serve reference: the model is built with the ARC **module** dropped but the
     **fuser still applies a precomputed `reference_with_time` streaming-sum tensor**
     (the "applying prepend/streaming_sum condition" log lines). Mirror that in training.
   - The trainer already has a conditioning path
     (`condition_tensors = condition_provider.prepare(batch.condition_attributes)`;
     `model(codes=…, condition_tensors=…)`), and the interleaver carries a
     `text_conditions` field — but wiring a **raw precomputed tensor** through it is
     unproven.
   - **Spike:** build the model keeping `reference_with_time` **in the fuser but not as
     a module**, feed one precomputed tensor into a single forward/backward, confirm the
     model consumes it (no crash, loss responds).
   - **Pass:** the replay data format = audio + alignments + a stored `reference_with_time`
     tensor + ⟨ret⟩ position; proceed to data-gen.
   - **Fail:** rethink (options: minimal trainer surgery to the interleaver/fuser, or
     the heavier ARC-in-trainer path with the meta patch).

## Pipeline (Stage 2b data generation, once spikes pass)

Per replay example (reconstructs the serve-time forward):
1. **Generate grounded QA** with an LLM (Claude/Gemini): question → a concise reference
   (the `"Reference: …"` Gemini would emit) → a faithful answer.
2. **Conversationalize** into a 2-speaker dialogue with real turn-taking
   (interruptions/backchannels), her=assistant, partner=user.
3. **Render to speech, stereo:** assistant (ch0) in **her voice** (spike-1 route), user
   (ch1) in a varied TTS; time-aligned.
4. **Precompute conditioning:** run the reference text through the **real `:8001` ARC
   encoder** → the `reference_with_time` tensor; store with the example (train==serve).
5. **Place the ⟨ret⟩/rag_token** via forced alignment (where retrieval fires).
6. **Alignments:** transcribe/align the assistant channel → `SPEAKER_MAIN` json (as with
   dialogue data).
7. **Hard/negative cases:** irrelevant reference → decline/no-hallucinate; no-retrieval
   turns; pick-the-right-reference. **Faithfulness filter:** LLM-judge drops answers not
   supported by their reference.

## Stages

- **Stage 2a — spikes** (TTS + conditioning injection). Cheap; gate the rest.
- **Stage 2b — small diverse batch** (~few hundred QA dialogues). Generate + render +
  precompute + filter. Retrain a probe adapter on **dialogue + this batch** and confirm
  **retrieval is restored** (model emits ⟨ret⟩ and uses the reference again) with voice
  intact. This is the proof-of-mechanism + eval.
- **Stage 3 — scale + final retrain.** Expand replay diversity; retrain a **fresh**
  adapter on the voice-dialogue + replay **mix** (~50/50 start; ratio is the
  voice-vs-retrieval dial). Judge **both** voice and retrieval.

## Success criteria

A Stage-3 adapter served on moshika-rag that, live: (1) is recognizably **Danielle**,
(2) **takes turns**, (3) **no screech** under conditioning, and (4) **retrieves** — asks
a factual question → model emits ⟨ret⟩, the reference is fetched (Gemini + `:8001`), and
the answer reflects it (vs Stage-1's "answered from its own head").

## Out of scope

- A document/vector corpus (this fork's retrieval is LLM-generated reference, not corpus
  lookup).
- Persona (separate; can ride the same reference channel later).
- A production-scale replay set before the small batch proves the mechanism.
- Multi-GPU/serving hardening.

## Open unknowns (resolved by the spikes / on the box)

- Does the fork expose forced-text generation to use `checkpoint_700` as TTS? (Spike 1)
- Can the trainer consume a precomputed `reference_with_time` tensor per example, and
  what's the exact config (reference_with_time in fuser, not as a module) + interleaver
  change? (Spike 2 — highest risk)
- ⟨ret⟩ token id + how forced alignment places it.
- Gradium voice-clone quality for her turns (fallback).
- The replay:dialogue mix ratio (tune in Stage 3).

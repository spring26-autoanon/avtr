# moshi-rag Voice + Retrieval LoRA — Experiment Log

Consolidated record of every training trial toward the goal: a LoRA that makes Moshi speak with
**Danielle's voice + personality** on **moshika-rag** while keeping **RAG retrieval** working.

## Shared setup
- **Base (Path A):** `kyutai/moshika-rag-pytorch-bf16`. **Base (Path B):** plain `kyutai/moshika-pytorch-bf16`.
- **LoRA:** rank 64, scaling 2.0, lr 4e-6, wraps self_attn + MLP (voice + turn-taking live in **attention**).
- **Serve stack (fork, moshi 0.2.13):** main `moshi.server` (GPU0) + `:8001` ARC reference encoder (GPU1) +
  Gemini retrieval LLM (`gemini-3.5-flash-lite`) + Gradium STT. Adapter via `MOSHI_LORA_WEIGHT`.
- **Retrieval mechanism:** model emits `⟨ret⟩`/rag_token (id **4**) → Gemini writes a "Reference:" text →
  `:8001` encodes it → injected as a `reference_with_time` streaming-sum tensor → model answers from it.
- **Data:** 2-speaker dialogue (Danielle isolated ch0 = SPEAKER_MAIN, partner ch1). Replay = synthetic
  RAG QA rendered to stereo. All alignments via Whisper (`annotate.py`).

---

## Trials

### 0. Freeze-attention probe (`moshika_voice_mlp_only`)
- Froze attention LoRA; trained MLP/gating only, on plain moshika.
- **Result:** voice **not** cloned (MLP alone can't carry timbre); turn-taking retained.
- **Conclusion:** voice identity AND turn-taking live in the **attention** layers → full LoRA required.

### 1. Path B voice clone (`moshika_dialogue`) → **checkpoint_000700**
- **Base:** plain moshika (mainline). **Config:** default (no RAG conditioners). Full attention+MLP LoRA.
- **Data:** 2-speaker dialogue (~135 min). **Steps:** ckpt 700.
- **Result:** ✅ cloned voice + turn-taking — recognizably her. Best pure-voice adapter we have.
- **Serving on moshika-rag (Path B overlay):** ❌ **SCREECHES** under RAG conditioning (adapter trained on
  plain moshika `W_m`, applied over moshika-rag `W_r` — only approximate; the reference streaming-sum
  perturbs the mismatched attention). → drove the move to **Path A** (train directly on moshika-rag).

### 2. Path A Stage 1 — voice-only on moshika-rag (`moshika_rag_pathA`) → **checkpoint_000400**
- **Base:** moshika-rag. **Config:** `config.norefenc.json` (`reference_with_time` removed from **both**
  conditioners AND `fuser.streaming_sum` → avoids the ARC meta-crash). **Data:** dialogue only (no replay).
- **Steps:** ckpts 100–400.
- **Result:** ✅ clean Danielle voice, **no screech** (Path A adapter serves directly on moshika-rag — no
  overlay mismatch). ❌ **retrieval KILLED** — the "retrieval floor": with no reference in the loss the
  model stops emitting `⟨ret⟩` and answers from its own head.
- **Tradeoff seen:** earlier ckpt = weaker voice / better behavior; later = stronger voice / monologues.
- **Takeaway:** need **replay data** (retrieval in the loss) to keep `⟨ret⟩` alive → Stage 2.

### 3. Stage 2b — dialogue + replay, default loss (`moshika_rag_stage2b`, RAG_TOKEN_WEIGHT=1)
- **Base:** moshika-rag. **Config:** `config.spike.json` (`reference_with_time` in `fuser.streaming_sum`
  but NOT a module; `first_speaker` prepend off). **Data:** dialogue (real voice) **+ replay** (127 ex:
  grounded/decline/smalltalk, rendered with **ElevenLabs instant clone** for her + premade TTS for the
  user), **weighted 0.5/0.5** multi-source. rag_token inserted at answer start; reference positioned at
  the `⟨ret⟩` frame in the text row. **Steps:** 600 planned; VM stopped ~500 (ckpts 100–500 saved).
- **Enabling fixes:** reference-conditioning-in-training surgery (`lm.py` streaming_sum over full seq) +
  data plumbing (per-example reference tensor) + `get_lora_moshi` meta-tensor fix.
- **Results by checkpoint:**
  - **300** — retrieval **reliable**, voice **weaker**.
  - **400** — retrieval **borderline** (fired once on World Cup), mid voice; confabulates ("robot/RHA").
  - **500** — **best voice** (most like her), **NO retrieval**, more empty-buffer **stalls**; confabulates
    ("AMA lowered transplant requirements in Florida").
- **Problems identified:**
  - **Confabulation** — invents official-sounding facts instead of emitting `⟨ret⟩` (rag_token is 1
    token/example vs a full content sentence at 0.5 weight → model keeps content, drops the trigger).
  - **Empty-buffer stalls** — short **terminal 2-turn** replay (Q→A→END) teaches "answer then stop."
  - **Voice↔retrieval tradeoff** along the checkpoint axis.
  - **Voice ceiling** — replay ch0 is the **ElevenLabs clone** ("like her but not completely") → half the
    voice signal is off-rendition.

### 4. Stage 2b + rag-token upweight (`stage2b_ragw25`, RAG_TOKEN_WEIGHT=25)
- **Same as #3** but the loss upweights rag_token (id 4) **25×** (`finetune/loss.py` + `RAG_TOKEN_WEIGHT`
  env). **Steps:** 600 (completed). Loss trended down cleanly (3.69 → 2.35 → 2.01).
- **Results by checkpoint (fully auditioned):**
  - **400** — ✅ sounds like **her** + natural conversation; ❌ **declines** factual Qs ("I'm sorry, I don't
    have that info") — no reliable `⟨ret⟩` (declines gracefully, does **not** hallucinate → decline
    examples worked).
  - **600** — ✅ **retrieves specific factual Qs reliably** (2 correct in one convo: first World Cup 1930
    Uruguay; Uruguay won twice 1930/1950) + **stall-free** natural conversation (she asks the user
    follow-ups); ❌ **lost her personality** — formal/robotic. Vague/open Qs ("tell me about the World
    Cup") still confabulate (no retrieval).
- **Verdict:** rag-upweight **works** — reliable retrieval reachable, stalls largely gone. But the
  **persona↔retrieval tradeoff persists** along the checkpoint axis (early=persona/decline,
  late=retrieval/formal). **No single checkpoint is both.** `checkpoint_000600` = keeper of this run
  (retrieval); `000400` = keeper for personality.

---

## Root-cause synthesis (why no checkpoint does both)
Retrieval and personality are trained in **separate, stylistically opposite datasets** — natural dialogue
(her real personality) vs replay (formal assistant Q&A, in the clone voice). As training progresses (and
with rag-upweight), the model shifts toward the replay behavior and away from natural-Danielle, so it is
forced to **trade** one for the other. Related: retrieval is **context-dependent** — leading with a
factual question primes "retrieval mode" and `⟨ret⟩` fires; chit-chat first primes "persona mode" and it
confabulates (replay examples are isolated Q&A with no surrounding conversation).

## What each trial proved (levers)
| Lever | Status |
|---|---|
| Full attention LoRA needed for voice | ✅ (trial 0) |
| Path A (train on moshika-rag) avoids screech | ✅ (trial 2 vs 1) |
| Replay data restores retrieval mechanism | ✅ (trial 3 — fires when it fires) |
| rag-token upweight → reliable `⟨ret⟩` triggering | ✅ (trial 4 / ckpt 600) |
| Multi-turn / non-terminal replay → kills stalls | designed (Stage 3) |
| Real her-voice recordings → removes clone voice ceiling | designed (Stage 3) |
| **Merge retrieval + persona in ONE dataset** → both at once | Stage 3 (the fix) |

## Stage 3 plan (next)
Record **real 2-person conversations**: ~2 hrs **retrieval conversation** (partner asks, Danielle
retrieves+answers naturally + follow-ups, from `replay/recording_dialogues.txt` — multi-turn, varied
shapes) + ~2 hrs **natural dialogue**. Both her real voice on isolated channels. Train **50/50**, with
`RAG_TOKEN_WEIGHT ≈ 15`. Because retrieval now happens **inside her natural conversation**, the model
learns "**natural Danielle who also retrieves**" instead of trading persona for retrieval, and learns to
retrieve regardless of prior context — targeting the vague-question confabulation too.

## Backed-up checkpoints (Mac `runs_backup/`)
- `stage2b_000300/400/500.safetensors` (trial 3)
- `ragw25_000600.safetensors` (trial 4 keeper) [+ optionally 000400 for personality]

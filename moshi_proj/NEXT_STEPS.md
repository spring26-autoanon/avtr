# NEXT_STEPS — moshi-finetune voice clone

_Status as of 2026-07-20. Read alongside `CLAUDE.md` (plan of record) and the
per-run gotchas in memory (`a100-runtime-gotchas`)._

## TL;DR

**The hard part is done: we can clone the target voice into Moshi.** The first
training run produced a LoRA whose `checkpoint_000300` clearly sounds like the
target speaker. The remaining problem is **interactivity** — it monologues and
won't let you interrupt — and that is a **data-design problem, not a
hyperparameter one.** The fix is to retrain on **two-speaker dialogue** (see
Part 1). Adding it to the full moshi-rag stack is a separate, heavier step
(Part 3) and should come only after the voice converses well on plain
`moshi.server`.

**Beyond Path B — the exact-voice + exact-retrieval roadmap is now Part 4.** Path B
(overlay a plain-moshika adapter onto moshika-rag) works but is an *approximation*
(measured attention drift 0.0547 — see `scripts/compare_attention.py`). The exact
route is **Path A (train on moshika-rag directly) + synthetic replay data**, staged
0→3 in Part 4. Do Path B first (it works today); pursue Part 4 only if the overlay
audibly smears voice or retrieval.

---

## Part 1 — Data for the next (interactive) run

### Core requirement: two speakers, separated onto the two stereo channels
- **Left channel (ch 0) = her, always.** The voice being cloned; must be the
  same person in every clip. Moshi treats this as its own voice.
- **Right channel (ch 1) = her conversation partner.** Can be anyone, and a
  *different* person per clip — variety is good. This channel just needs real
  speech so the model learns to listen and yield instead of monologuing.

### How to record (make-or-break part)
- Each speaker on their **own isolated track**, no bleed:
  - Remote-call recorder that saves **per-participant tracks** (Riverside,
    Zencastr, Zoom "record a separate audio file for each participant"). Her
    track → left, partner's track → right.
  - In person: **one mic per person** (two lavalier/headset mics), each to its
    own channel; sit them apart to minimize crosstalk.
- **Natural conversation**, not scripted turns. You *want* real interruptions,
  overlaps, backchannels ("mm-hm", "right"), and pauses — that temporal
  push-and-pull is what teaches turn-taking. Don't force clean non-overlapping
  turns.
- **Audio specs:** capture at 48 kHz, clean/low-noise, consistent levels (the
  pipeline downsamples to 24 kHz). Each speaker mono, then combine to stereo
  (her=left, partner=right), **time-aligned** so their turns line up as they
  actually happened.

### How much
- Today's ~58 min of *monologue* cloned the voice but had too little variety to
  behave. For interactive dialogue aim for **a few hours** — ideally **2–5 hrs
  of her in conversation**, across **many different partners and topics**. Even
  **1–2 hrs** would be a real improvement.
- Keep **her voice/recording conditions consistent** across clips (same mic if
  possible) to reinforce the clone.
- If the end goal is an **assistant**, bias content toward her *answering
  questions / helping* — the model imitates the behavior it sees.

### Pipeline changes this requires
- `prepare_stereo.py` today does mono→left, **silence→right**. New version: take
  the **two separate tracks** and place **her→left, partner→right**
  (time-aligned). This is the key change.
- `annotate.py` stays the same — transcribes **channel 0 (her)** only, which is
  what training needs (`SPEAKER_MAIN`). The partner's audio is used as audio,
  not transcribed.
- **Fix the eval set:** hold out **2–3 files**, not one, so `eval_loss` stops
  logging NaN and you get a real overfitting curve.
- ⚠️ **If you only have mixed recordings** (both people on one track), you'd
  need a **speaker-separation/diarization** step first — doable but lossy.
  Separate tracks at record time is far cleaner; prioritize that.

---

## Part 2 — What we learned today

### Worked
- **The whole training pipeline runs end-to-end** on the A100, and **the voice
  clone genuinely works** — `checkpoint_000300` sounds like the target.
- **Path B is the right approach:** train the LoRA on **plain
  `kyutai/moshika-pytorch-bf16`** with the repo's native stack (mainline moshi
  `0.2.4a1`, torch 2.6), then serve with `moshi.server --lora-weight`. The
  adapter **loaded cleanly** on the server — no key mismatch.
- Clear **voice-vs-behavior tradeoff** across checkpoints: 100 = weak
  voice/better behavior, 300 = strong voice/monologues. Good evidence for a
  writeup.

### Didn't work / key findings
- **Can't fine-tune the *true* moshika-rag architecture with this trainer.** It
  needs the `moshi-rag` fork (moshi 0.2.13, torch 2.9.1), whose RAG ARC-encoder
  crashes under the trainer's meta-device init
  (`unsupported autocast device_type 'meta'`). That's why we pivoted to Path B.
- **The monologuing/no-interruption is caused by the DATA, not hyperparameters.**
  Option A (voice-left / **silent**-right) taught the model the user never
  speaks, so it under-attends to your audio and talks forever. More training
  made it worse. → Part 1 is the fix.
- Operational gotchas: needs `CUDA_VISIBLE_DEVICES=0`; manifests need
  **absolute** paths (`sphn.dataset_jsonl` resolves relative to the jsonl dir);
  `annotate.py` needs `-l/--local` (no Slurm on this VM); tmux does **not**
  survive an SSH drop on this VM (use `setsid`/`nohup` or
  `loginctl enable-linger`); `eval_loss` logs NaN from a too-small eval set.

### The working commands (reference)

Train (native stack, mainline moshi pinned to `061cc4c`):
```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train example/moshika_rag_voice.yaml
```

Serve + audition a checkpoint:
```bash
CKPT=~/moshi-finetune/runs/moshika_rag_voice/checkpoints/checkpoint_000300/consolidated
CUDA_VISIBLE_DEVICES=0 uv run python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  --lora-weight "$CKPT/lora.safetensors" \
  --config-path "$CKPT/config.json"
# then, from the mac:  ssh -L 8998:localhost:8998 wb-gpu-training  → http://localhost:8998
```

---

## Part 2b — Freeze-attention experiment (done 2026-07-20): voice lives in attention

Tested a `freeze_attention: true` LoRA (MLP/gating only; new flag in `finetune/args.py`
+ `wrapped_model.py`, config `example/moshika_voice_mlp_only.yaml`). Served the
result and listened. **Result:**
- **Turn-taking works / no monologuing** (attention untouched → base moshika's
  conversational behavior preserved), BUT
- **Voice is NOT cloned** (MLP/gating alone can't carry her timbre).

**Conclusion: voice identity AND turn-taking both live in the attention layers —
they're entangled in the same weights.** You can't separate them by freezing:
- freeze attention → keep turn-taking, lose voice;
- train attention on monologue → get voice + monologuing.
So the ONLY route to voice + turn-taking is **full LoRA (attention included) trained
on DIALOGUE data** (turn-taking must be present in the data). This confirms Part 1
(dialogue data) is the real fix — not a layer-selection trick.

Serving note discovered: an MLP-only adapter won't load directly (the server builds
attention LoRA slots our adapter doesn't fill → `Cannot copy out of meta tensor`).
Worked around by **padding** the adapter with zero attention tensors (see the
`lora_padded.safetensors` step). Also: the standalone `moshi.server` needs
**sphn==0.1.12** (newer sphn drops `OpusStreamReader.read_pcm`).

## Part 3 — Adding it to moshi-rag (the full RAG stack)

Heavier "Stage 2", separate from the voice work. A **3-service system** (from
moshi-rag's README):

1. **Reference-encoder service** —
   `python -m moshi.moshi.server_conditioner --config hf://kyutai/moshika-rag-pytorch-bf16/config.json --moshi-weight hf://…/model.safetensors --conditioner reference_with_time --port 8001`
2. **Retrieval LLM** — vLLM serving `google/gemma-3-27b-it` on port 8002.
   ⚠️ ~54 GB bf16 alongside Moshi (~16 GB) is tight on one 80 GB A100 — may need
   a smaller retrieval model or a second GPU.
3. **Main server** — `python -m moshi.moshi.server` with env vars pointing at
   the reference encoder, the LLM, and an **external STT API**
   (`STT_URL`/`STT_API_KEY`, Gradium) to transcribe the user. **The STT key is a
   hard external dependency.**

**Where the LoRA goes:** into the **main server** (`server.py`, the LM loader) —
*not* `server_conditioner.py`. moshi-rag's `loaders.py` already supports
`CheckpointInfo.lora_weights` / `get_moshi_lm(lora_weights=, fuse_lora=)`, so add
a `--lora-weight` flag to `server.py` and thread it through (CLAUDE.md's Option
A, but on the correct file).

**Caveats to verify:**
- The LoRA was written by moshi **0.2.4a1**; moshi-rag serves on **0.2.13**. It
  loaded fine on 0.2.4a1 — the 0.2.13 key-naming compatibility is **untested**
  and may need a small key remap.
- The adapter is a delta over **moshika**, applied to **moshika-rag** weights —
  should be fine (shared layers), but confirm by ear.

**Sequencing:** RAG doesn't fix (or hide) the monologuing. Get the interactive
voice working first via the dialogue-data retrain (Part 1), validated on plain
`moshi.server`. Wire into moshi-rag only once she *converses* well.
Voice-first, RAG-second.

### Path B → moshika-rag overlay: attention keys DO match (resolved 2026-07-20)

Earlier I thought the attention LoRA was incompatible with moshika-rag. **That was a
false alarm** — I compared adapter *module* names (`self_attn.in_projs.N`) against
moshika-rag's *stored* weight names (`self_attn.in_proj_weight`, fused). Those never
match for ANY adapter, because storage != runtime module layout.

Verified with a 2-step fork run (`example/fork_attn_check.yaml`) + key inspection:
- Plain moshika **stores** attention fused (`in_proj_weight`), exactly like moshika-rag.
- Yet the loader **unfuses it into per-step `in_projs.N` Linear modules at load time**,
  and LoRA binds to those. So a fork-trained adapter has `in_projs.N` attention LoRA
  keys, and the fork-built moshika-rag model has the same `in_projs.N` modules →
  **attention (and thus the voice) transfers to moshika-rag.**

**So: train the retrain under the FORK (moshi 0.2.13) on plain moshika, full LoRA.**
The resulting adapter is moshika-rag-compatible (all layers, attention included).
The only real remaining blocker for serving on moshika-rag is the `get_lora_moshi`
meta-tensor bug (below) — a code patch, not an architecture problem.

**Integration tasks for Stage-2 serving:**
1. ~~Smoke-test that the fork's trainer runs on plain moshika (no conditioners).~~ **DONE 2026-07-20 —
   green, loss 2.761. pyproject.toml is now on the fork (moshi 0.2.13 / torch 2.9.1). Retrain this way.**
   Still to confirm on the first real fork checkpoint: that its LoRA keys match moshika-rag's fused
   attention (run scripts equivalent to `smoke_keys.py` on the new `lora.safetensors`). If the fork
   wraps attention differently, the adapter may cover fewer layers — verify early.
2. Patch the fork's `get_lora_moshi` (moshi/models/loaders.py ~line 560): it does
   `model.to(device)` on a meta-initialized model → `Cannot copy out of meta tensor`.
   Needs `to_empty()` + weight load. Serving moshika-rag with `--lora-weight` hits this.

Also: the moshika-rag ARC encoder needs gated `meta-llama/Llama-3.2-3B-Instruct`
(request HF access) for real serving.

## Part 4 — Roadmap: exact voice + exact retrieval (Path A + replay data)

**When to reach for this:** only if Path B's overlay audibly degrades voice or
retrieval. Path B works today with zero new training-code; Part 4 is the heavier,
*exact* alternative. See `scripts/compare_attention.py` — moshika-rag's attention
drifted only **0.0547 median** from plain moshika, so Path B will *probably* sound
fine. Part 4 is the fallback + the "both truly exact" ceiling.

### Why Path B is only an approximation
Path B serves `W_r + ΔW` where `ΔW` was trained against plain moshika `W_m`, not
`W_r`. A LoRA is calibrated to the base it trained on, so applying it to a different
base is only approximate. Voice AND retrieval-*integration* both live in the shared
attention (`in_projs.N`), so the overlaid `ΔW` can mildly perturb retrieval. The
exact fix = train `ΔW` against `W_r` directly (Path A).

### The load-bearing data principle (why replay is required)
- **Conditioner WEIGHTS never get lost.** Weights only change when trained. The 5
  RAG-only tensors (`reference_with_time`, `fuser`, `rag_token_id` = the retrieval
  *ingestion* machinery) are never LoRA-wrapped — `wrapped_model.py:174-184` only
  trains params named `"lora"` on `self_attn`/MLP; everything else stays
  `requires_grad=False`. Preserved bit-for-bit in BOTH paths.
- **But the model can learn to IGNORE conditioning** if training data contains no
  retrieval — the exact same failure as the silent-user-channel monologuing bug
  (omitted behavior → forgotten). Voice-only Path A *actively* trains "conditioning
  is uninformative" (conditioners present but fed null every step), so it can degrade
  retrieval **more** than Path B, which never trains that lesson.
- **Replay data is the only thing that keeps retrieval in the loss** — analogous to
  how 2-speaker dialogue data is the fix for turn-taking (Part 1). Not optional.

### Staged plan
**Stage 0 — Unblock Path A (prerequisite; pure engineering, not a config change).**
- Patch the meta-init crash: the ARC-encoder builds on the `meta` device →
  `unsupported autocast device_type 'meta'`; and `get_lora_moshi`
  (`moshi/models/loaders.py` ~560) does `model.to(device)` on a meta model →
  `Cannot copy out of meta tensor`. Fix = `to_empty()` + weight load, materialize
  conditioners on a real device, drop autocast-on-meta. **Pinpoint the exact lines
  against the fork checkout ON THE A100** (moshi isn't installed on the mac).
- Get gated `meta-llama/Llama-3.2-3B-Instruct` access (ARC-encoder dependency).
- Wire **null/neutral conditioning** through the training forward pass; freeze the
  conditioners (LoRA only on transformer attention+MLP).
- Config: `hf_repo_id` = moshika-rag, `config_path` = the **full** moshika-rag
  `config.json` (conditioners live — do NOT strip).
- Milestone: builds + steps cleanly on the RAG architecture.

**Stage 1 — Path A voice-only run.**
- Purpose: prove the RAG-arch trainer runs + get **exact** voice.
- ⚠️ **JUDGE VOICE ONLY.** Retrieval is worst-case here by design (trained to ignore
  conditioning) — expected, not a failure. This run establishes the *retrieval floor*
  (good writeup baseline: how bad retrieval gets with zero replay).

**Stage 2 — Synthesize replay data (can start NOW, in parallel — use the existing
`checkpoint_000300` clone as the TTS voice; not gated on Stage 1).**
Per example (reconstructs moshika-rag's serve-time forward):
- User question (ch1) that *requires* external knowledge.
- The retrieved passage = ground-truth reference.
- Assistant answer (ch0, **her voice**, faithful to the passage) + `SPEAKER_MAIN`
  alignments.
- Conditioning rebuilt by running the passage through **moshika-rag's OWN reference
  encoder** → `reference_with_time` + `rag_token` (so train == serve, no skew).

Pipeline: pick corpus = **the deployment retrieval corpus** (coverage = retrieval
ceiling) → LLM generates grounded QA → conversationalize into multi-turn dialogue
with **real interruptions/overlap** (turn-taking, per Part 1) → render to speech
(assistant = her-voice TTS anchored with a few real recordings; user = varied voices;
time-align stereo her=L/partner=R) → rebuild conditioning via the real reference
encoder → add **hard/negative cases** (irrelevant passage → decline/no-hallucinate,
multi-passage pick-the-right-one, no-retrieval turns) → **faithfulness filter**
(LLM-judge drops any answer not supported by its passage). Mix with pure her-voice
dialogue; ratio (~50/50 start) is the voice-vs-retrieval dial.

**Stage 3 — Path A retrain on the mix (voice dialogue + replay).**
- The real deliverable. Train a **fresh** LoRA on the mixed set (don't continue
  Stage 1 — you want retrieval in the loss from step 0).
- Judge **BOTH** voice and retrieval (retrieval probe: ask something only the corpus
  supplies).

### Ceiling honesty
"Exact + exact" is a **Pareto** optimum, not literal double-100%: capped by (a) the
shared-capacity compromise (voice and retrieval share attention), (b) replay
**coverage** → match the deployment corpus to collapse this, (c) LoRA **rank** →
raise it if both don't fit. Near-perfect-both is achievable; literal-perfect-both is
a category error.

### Dependency chain
Exact voice first → that cloned model is the TTS voice for the replay set → retrain.
(The existing Path B clone can bootstrap Stage 2 now, so data-gen isn't gated on
Stage 0/1.) Companion diagram of the Path B mechanism (fused/unfused attention, why
the fork): artifact `pathb-fork-lora` (published 2026-07-20).

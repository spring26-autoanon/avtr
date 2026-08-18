# Phase 2 — Voice + persona + retrieval QA on moshika-rag (the papers corpus)

_Design doc — 2026-07-30. Sub-project 2 of the moshi-rag stage. Follows Phase 1
(`…-phase1-serve-adapter-design.md`, DONE: adapter overlays on fork moshika-rag).
Read alongside `NEXT_STEPS.md` Parts 3–4 and the moshi-rag paper._

## Goal

Stand up the full moshi-rag serving stack so Danielle's voice adapter answers
**factual questions grounded in a retrieval corpus**, with a **consistent persona**.
First corpus = the two papers already in the repo (`base-moshi-paper.pdf`,
`moshi-rag-paper.pdf`) → she becomes an expert on her own architecture. Small,
self-contained, ideal for proving the pipeline before pointing it at anything bigger.

## What Phase 1 established (inputs)

- `checkpoint_000700` overlays cleanly on fork moshika-rag via `lora_padded.safetensors`
  (674 voice keys match exactly; 372 conditioner slots zero-padded) with
  `get_moshi(fuse_lora=True, lm_kwargs_overrides={"lora":True,"lora_rank":64,"lora_scaling":2.0})`.
- Fork env recipe known (moshi 0.2.13 / torch 2.9.1+cu128, runs on A100 12.4 driver).
- The fork's `server.py` is the RAG main server, hard-coupled to the retrieval/STT
  services — so a live conversation *requires* those services (this doc stands them up).

## The retrieval mechanism (from the moshi-rag paper — how it actually works)

Moshi predicts a **⟨ret⟩ token** mid-utterance when it needs external knowledge. On
⟨ret⟩: the **conversation transcript** (Moshi's inner-monologue text + the user's text
from **STT**) is sent to a backend → relevant material is **retrieved from the corpus**
→ a **reference LLM (gemma-27b)** produces reference text → the **reference/ARC encoder**
turns that text into conditioning (`reference_with_time`) → Moshi continues speaking,
grounded in it. So the corpus is a **text index consulted live**; the model already has
the general "use retrieved reference" skill (no corpus-specific training — see NEXT_STEPS
Part 4 / prior discussion).

## Architecture — the 3-service stack on `wb-gpu-a1ultra2g` (2×A100-80GB)

1. **Retrieval LLM** — gemma-3-27b via vLLM. **GPU 0** (~54 GB, fits with headroom).
2. **Reference-encoder service** — `server_conditioner.py`, the `reference_with_time`
   conditioner (ARC encoder, needs gated `meta-llama/Llama-3.2-3B`). **GPU 1**.
3. **Main server** — `server.py` + the voice adapter (`lora_padded`), env vars pointing
   at services 1 & 2 and the **Gradium STT**. **GPU 1** (Moshi ~16 GB + encoder).

Services talk over HTTP (`get_reference_encoder_url`, retrieval env, `STT_URL`).

## Key decisions

1. **Corpus = the two papers**, extracted to text and indexed in whatever format
   moshi-rag's retrieval expects (determined in Step 2). Swappable later.
2. **Persona = Route A (inference-time grounding)** — inject a Danielle persona/bio
   through moshi-rag's conditioning/retrieval context; **no persona training** (Route B
   synthetic-data is a later fallback only if identity still wobbles).
3. **Path B overlay** (Phase-1 adapter), not Path A exact-training. Path A + replay is
   out of scope (NEXT_STEPS Part 4), pursued only if retrieval measurably smears.
4. **Documented stack as-is** (2 GPUs available) — no smaller model / hosted-API detour.

## Approach & steps (each ends at a verification checkpoint; branch where noted)

**Step 1 — Provision the 2-GPU box.** Build the fork env on `wb-gpu-a1ultra2g` (mirror
`~/fork-venv` from Phase 1); rsync `checkpoint_000700/consolidated/` (incl.
`lora_padded.safetensors`) from the mac; `huggingface-cli login`; confirm 2×A100 +
moshika-rag + Llama-3.2 access. **Checkpoint:** `serve_check.py` loads moshika-rag +
padded adapter fused on this box.

**Step 2 — Learn moshi-rag's retrieval pipeline + index the papers (the crux, INSPECT
ON BOX).** Read the fork's `reference/llm_reference_generator.py`,
`inference_utils/retrieval_profiles.py` (`load_retrieval_env`, `default_profile_id`),
and how `server_conditioner.py` / the reference encoder consume corpus text. Determine:
(a) how a **retrieval profile / corpus** is configured and pointed at documents; (b)
whether retrieval is vector-index-based or LLM-over-documents; (c) the required corpus
format. Then **extract the 2 PDFs → text → chunk → build the index/profile** in that
format. **Checkpoint:** a retrieval profile over the papers that returns a relevant
passage for a test query (e.g. "what is Inner Monologue?"). **Decision branch:** if
moshi-rag hard-codes its own corpora / has no clean custom-corpus hook, we add a minimal
retrieval profile or a thin retrieval shim over the papers.

**Step 3 — Bring up the reference-encoder + retrieval-LLM services.** Launch
`server_conditioner.py` (reference encoder, GPU 1) and gemma-27b under vLLM (GPU 0),
detached (`setsid nohup`, linger). **Checkpoint:** both services healthy and reachable
at their URLs.

**Step 4 — Source + wire the Gradium STT.** Obtain the Gradium key (external, user
to-do); set `STT_URL` / key env. **Checkpoint:** STT transcribes a test utterance.

**Step 5 — Boot the main server with the adapter.** Add `--lora-weight` to `server.py`
threaded into the LM loader with the Phase-1 overrides + `lora_padded.safetensors`;
point env at services 3/4. Boot on GPU 1, detached. **Checkpoint:** server accepts a
WebSocket connection with the adapter fused, services wired.

**Step 6 — Persona (Route A).** Author a Danielle persona/bio; inject it through the
conditioning/retrieval-context channel identified in Step 2. **Checkpoint:** across
several fresh conversations she states a *consistent* identity.

**Step 7 — Live end-to-end test.** Tunnel + converse. **Success = all:** (a) her voice,
(b) turn-taking, (c) **answers a papers-only question correctly** (retrieval probe: ask
something answerable only from the corpus, e.g. a specific number/method from a paper),
(d) consistent persona. **Checkpoint:** recorded judgment; note voice-vs-retrieval
quality (Path A trigger if retrieval is clearly smeared).

## Success criteria (Phase 2 done)

The 3-service stack runs on the 2-GPU box; a live conversation is in **Danielle's
voice**, **takes turns**, **answers questions grounded in the two papers**, and holds a
**consistent persona** — with no persona/Path-A training required.

## Out of scope (later / other)

- Path A exact-training + replay data (NEXT_STEPS Part 4) — only if retrieval smears.
- Route B synthetic persona data — only if Route A identity wobbles.
- A larger/production corpus — swap in after the papers pipeline works.
- Multi-user / production hardening, autoscaling, cost work.

## Open unknowns (resolve on the box; can't see the fork from the mac)

- How moshi-rag configures a **custom retrieval corpus/profile**, and the corpus format
  (Step 2 — the biggest unknown).
- Whether the reference encoder / retrieval expects a specific chunking/embedding.
- gemma-27b vLLM memory/latency on GPU 0 alongside everything (retrieval latency >1.5 s
  hurts, per the paper).
- Exact `server.py` env wiring for the reference encoder / retrieval LLM / STT.
- Gradium key acquisition + endpoint details.

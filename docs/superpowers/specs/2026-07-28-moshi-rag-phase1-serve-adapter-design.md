# Phase 1 — Serve checkpoint_000700 on moshika-rag under the fork

_Design doc — 2026-07-28. Sub-project 1 of the moshi-rag stage. Read alongside
`NEXT_STEPS.md` (Parts 2b, 3, 4) and `CLAUDE.md` (Packaging Option A)._

## Where this sits

The dialogue retrain is done: `checkpoint_000700` clones Danielle's voice **and**
takes turns, served today on plain `moshi.server` (native stack, moshi 0.2.4a1).
The original goal is to run it on **moshika-rag**. That splits into two sequenced
sub-projects:

- **Phase 1 (this doc):** get `checkpoint_000700` running on the moshika-rag base
  **under the fork**, via the real `server.py`, and confirm it preserves voice +
  turn-taking. No retrieval.
- **Phase 2 (separate, later):** voice + persona + factual retrieval QA — the full
  3-service RAG product. Scoped on its own.

## The question Phase 1 answers

**Does the mainline-trained (moshi 0.2.4a1) `checkpoint_000700` adapter load onto the
fork-built moshika-rag and preserve her voice + turn-taking?**

This is genuinely unresolved — the docs contradict each other:
- `pyproject.toml` (lines 15–19): a **mainline**-trained adapter has per-step
  `in_projs.N` attention naming → "HALF-incompatible with moshika-rag"; only a
  **fork**-trained adapter overlays cleanly.
- `NEXT_STEPS.md` Part 2b "CORRECTION": that was a false alarm — plain moshika (and
  moshika-rag) both **store** attention fused (`in_proj_weight`) but the loader
  **unfuses** to per-step `in_projs.N` modules that LoRA binds to at runtime, so the
  keys match.

The correction was verified for a **fork-trained** adapter. **Ours is mainline-trained**,
so whether mainline and fork unfuse identically is untested. Phase 1 settles it
empirically. If they don't match, the fallback is a key-remap or a fork retrain.

## Key facts / constraints

- **Adapter under test:** `runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/`
  (`lora.safetensors` + `config.json`), LoRA rank 64, trained on plain
  `kyutai/moshika-pytorch-bf16`, **mainline moshi 0.2.4a1 / torch 2.6**. Backed up to
  the mac at `adapters/checkpoint_000700/`.
- **Fork stack:** moshi 0.2.13 / torch 2.9.1 / sphn 0.2.x — declared in
  `pyproject.toml`, and the fork **training** smoke test previously ran green on plain
  moshika. The box's *active* venv is currently mainline (0.2.4a1); a fork env may
  still exist from the smoke test, else build fresh.
- **External access:** `meta-llama/Llama-3.2-3B-Instruct` HF access is **approved**
  (the moshika-rag ARC/reference encoder depends on it).
- **Known serving blocker — the `get_lora_moshi` meta-tensor bug:** on the fork,
  `moshi/models/loaders.py` (~L560) calls `model.to(device)` on a meta-device model →
  `Cannot copy out of meta tensor`. Fix pattern (from NEXT_STEPS Part 4 Stage 0):
  `to_empty()` + explicit weight load; materialize conditioners on a real device; drop
  autocast-on-meta. **Exact lines must be pinpointed against the fork checkout on the
  A100** (the fork is not on the mac).
- **LoRA integration point:** moshi-rag's `loaders.py` already supports
  `CheckpointInfo.lora_weights` and `get_moshi_lm(lora_weights=, fuse_lora=)`. The
  work is adding a `--lora-weight` flag to the **main `server.py`** and threading it
  through (CLAUDE.md Option A / NEXT_STEPS Part 3). This plumbing is **reused in
  Phase 2** — not throwaway.
- **Path B, under the fork:** we overlay a plain-moshika adapter onto moshika-rag at
  serve time. Measured attention drift plain→rag was 0.0547 median
  (`scripts/compare_attention.py`), so voice/behavior should survive the overlay.
  Exact-training (Path A) is Part 4, out of scope here.

## Approach

Use the **real moshi-rag `server.py`**, integrated with a `--lora-weight` flag, booted
in the **lightest config that runs** — neutral/null conditioning, **no gemma, no STT**.
This tests exactly what Phase 1 needs (adapter overlays + she converses in her voice on
the real moshika-rag server) while leaving retrieval infrastructure for Phase 2, and it
produces the real serving plumbing we keep.

**The load-bearing unknown:** does the fork's `server.py` **boot standalone** with
conditioning fed neutral, or does it **hard-require** the reference-encoder / retrieval
LLM / STT services to start? Unknown from the mac. The plan branches on it (below).

## Steps (each ends at a verification checkpoint)

1. **Locate or build the fork env.** Search the box for the existing smoke-test env
   (e.g. `ls -d ~/*venv* ~/.venv*`; check which moshi the fork smoke configs used). If
   present, use it; else build fresh from `pyproject.toml`'s fork spec.
   **Checkpoint:** `python -c "import moshi, torch; print(moshi.__version__, torch.__version__)"`
   → `0.2.13`, `2.9.1`.

2. **Stage moshika-rag assets + confirm the gated dep pulls.** Fetch
   `kyutai/moshika-rag-pytorch-bf16` `config.json` + `model.safetensors`; confirm
   `meta-llama/Llama-3.2-3B-Instruct` actually downloads with the box's HF login.
   **Checkpoint:** both resolve locally; no 401/gated errors.

3. **Load moshika-rag + overlay 700 (find & patch the meta bug).** In a short throwaway
   *loader probe* (not the server yet), build `CheckpointInfo` for moshika-rag with
   `lora_weights = checkpoint_000700/consolidated/lora.safetensors` and call
   `get_moshi_lm(..., lora_weights=, fuse_lora=True)`. Expect the `get_lora_moshi`
   meta-tensor crash (and possibly the ARC meta-init crash). **Pinpoint the exact lines
   on the box** and apply the `to_empty()` + weight-load / real-device-materialize
   patch. **Checkpoint:** the model constructs and the adapter loads with no
   meta-tensor / autocast-on-meta error.

4. **Key-compat verdict (the crux).** Compare the adapter's parameter names against the
   fork model's LoRA-bound attention modules — confirm the `in_projs.N` names line up
   (no unfilled/meta LoRA slots, no leftover adapter keys). **Decision branch:**
   - **Match** → proceed to Step 5.
   - **Mismatch** → do NOT force it. Options, in order: (a) a mechanical key-remap
     (mainline↔fork attention naming) if the difference is purely cosmetic; (b) if the
     layout genuinely differs, **retrain the adapter under the fork** (the fork
     *training* path is already smoke-tested green) and re-enter at Step 3. Record which.

5. **Integrate `--lora-weight` into the real `server.py` and boot minimal.** Add the
   flag to moshi-rag's main `server.py` and thread it into the LM loader
   (`CheckpointInfo.lora_weights` / `get_moshi_lm(lora_weights=, fuse_lora=True)`). Boot
   with moshika-rag's **full** config (conditioners live) but **neutral/null
   conditioning** and **no gemma/STT**. **Decision branch on the boot dependency:**
   - **Boots standalone** → this is the Phase-1 test rig; continue.
   - **Hard-requires the retrieval/STT services** → either stub those endpoints (empty
     reference / no-op STT) enough to boot, or, if that's not quick, fall back to a
     load-and-converse harness (reuse the Step-3 probe + Mimi) for the compat verdict
     and **defer full-server boot to Phase 2**. Record which path was taken.
   **Checkpoint:** something serves and accepts a WebSocket connection.

6. **Converse & judge.** Tunnel (`ssh -L 8998:localhost:8998 wb-gpu-training`) and talk
   to her. **Success = both:** (a) voice is recognizably Danielle on the moshika-rag
   base, and (b) she still takes turns / lets you speak (turn-taking survived the
   overlay). Also note whether neutral conditioning leaves conversation coherent (input
   for Phase 2). **Checkpoint:** live judgment recorded.

## Success criteria (Phase 1 done)

1. A fork env (moshi 0.2.13) loads moshika-rag with `checkpoint_000700` overlaid, no
   key-mismatch or meta-tensor errors (meta bug patched).
2. The compat verdict is recorded: adapter keys matched (Path B overlay works) **or**
   we retrained under the fork / remapped — with the resulting working adapter.
3. `--lora-weight` is wired into the real `server.py` (reusable for Phase 2).
4. Live conversation on the moshika-rag base preserves **voice + turn-taking**.

## Out of scope (→ Phase 2)

- Actual retrieval QA: the reference-encoder service, gemma-3-27b via vLLM, the STT key,
  and wiring the 3-service stack together.
- Persona consistency (conditioning grounding and/or synthetic persona data).
- Path A exact-training + replay data (NEXT_STEPS Part 4).

## Open unknowns to resolve on the box (not from the mac)

- Exact lines of the `get_lora_moshi` meta bug in the fork's `loaders.py`.
- Whether the mainline-trained adapter's keys match the fork model (Step 4 verdict).
- Whether `server.py` boots without the retrieval/STT services (Step 5 branch).
- Whether moshika-rag under **neutral** conditioning converses coherently.

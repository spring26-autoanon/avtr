# MoshiRAG Eval Pipeline — Requirements

## Context

Building an eval pipeline for MoshiRAG (arxiv: 2604.12928, https://arxiv.org/abs/2604.12928), a full-duplex speech language model with asynchronous RAG retrieval built on top of Moshi. The pipeline must support clean checkpoint swapping, be extensible with new evals, and serve as the measurement framework across base and fine-tuned model variants.

The repo serves two purposes that share all infrastructure:

1. **Automated evals** — batch runs over benchmark datasets, scored and written to JSON
2. **Live demo** — a browser-based full-duplex voice interface for ad-hoc testing, edge-case probing, and stakeholder demos, served from the same VM as the evals

There is no native macOS execution. The VM runs everything. macOS (and any other client) connects to the demo via browser over an SSH tunnel — no Python, no model weights, no torch on the client side.

Claude Code should read the MoshiRAG paper before implementing anything — the MoshiRAGAdapter, LLM judge prompts, keyword extraction prompt, and E2EKD measurement methodology all have specific details in the paper (particularly Tables 16 and 17 and Section 3) that must not be approximated.

---

## Architecture: Target moshi-rag's Production Server

Earlier development cycles tried to decouple from the moshi-rag production
server architecture, building a custom in-process `MoshiRAGAdapter` +
FastAPI/WebSocket demo. That direction is abandoned — too many debugging
cycles chased bugs that turned out to be self-inflicted by the custom
implementation rather than genuine upstream defects (confirmed via direct
A/B testing against the unmodified stack — see CLAUDE.md's Phase 0
findings). Current direction: both automated evals and the live demo target
kyutai-labs/moshi-rag's own, unmodified `moshi.server` +
`moshi.server_conditioner` as the underlying framework, sharing as much
pathway as possible between the two for consistency. Reversibility is via
git history, not a maintained dual-path.

**Conditioning is separate-process, for both demo and evals** — this
reverses an earlier requirement that it run in-process. The ARC-Encoder
(reference-text conditioner) always runs as a real, separately launched
`server_conditioner` process; nothing here loads it in-process. This
started as a scoped exception for `respond()`/evals only (a confirmed bug:
in-process conditioning broke every `respond()` call after the first one
whenever retrieval was enabled) but was promoted to the sole architecture
for both paths once the broader pivot made clear that in-process
conditioning was never how moshi-rag's own production server operates,
for any path.

**Do not:**
- Run the ARC-Encoder / conditioner in-process
- Reimplement or fork moshi-rag's `InferenceJob`/`ServerState`/`Channel`
  step-loop logic beyond narrow, documented, evidence-backed patches (see
  CLAUDE.md's `_patch_output_loop` notes for the one that remains)
- Use `sounddevice` anywhere
- Port or adapt the Rust backend
- Reference the Gradio tunnel pattern

**Do:**
- Launch the demo via `scripts/run_demo.sh` (real `moshi.server` +
  `moshi.server_conditioner`, unmodified)
- Point `respond()`/evals at a real `server_conditioner` process via
  `REFERENCE_ENCODER_URL` — mandatory, not optional; `MoshiRAGAdapter`
  raises at construction if unset
- Keep `core/` (`model_interface.py`, `retrieval_backend.py`,
  `checkpoint.py`, `config.py`) as the shared abstraction layer evals
  build on — `MoshiRAGAdapter` drives real `ServerState`/`InferenceJob`
  classes directly, not a WebSocket client of `moshi.server`
- Apply the same narrow, documented, evidence-backed `RAGManager` patch
  technique already used for `respond()`'s retrieval routing
  (`model_interface.py`'s `_patch_rag_manager`) to the demo's
  `moshi.server` process too, via `scripts/instrumented_server.py`. Most of
  `instrumented_server.py`'s patches are additive logging only (see
  "Session instrumentation" below) — but its `RAGManager.get_reference_text`
  patch is a **deliberate, bounded exception to "logging only"**: it
  redirects the demo's actual retrieval call through the same
  `core/retrieval_backend.py` `RetrievalBackend` (same registry-selected
  backend, same prompt template) evals use, bypassing moshi's native
  `LLMReferenceGenerator` for real retrieval entirely — the only way to get
  genuine prompt/backend parity between the demo and evals, since
  `LLMReferenceGenerator` only ever offers a choice between two bundled
  canned templates, not arbitrary prompt text (confirmed against real
  `kyutai-labs/moshi-rag` source — see "Session instrumentation" below for
  the full rationale and what's given up). It still does not touch
  `trigger()`/step-loop/`Channel` control flow — only what happens one call
  deeper, inside `RAGManager._background_task`'s existing call to
  `get_reference_text` — so it fits the same "narrow, documented,
  evidence-backed patch" allowance as the bullet above, just not the
  "additive logging only" framing that used to describe every
  `instrumented_server.py` patch uniformly
- Align the demo's per-turn instrumentation with the same three latency
  definitions specified for evals/registry/latency (`ttfat`, `e2ekd`,
  `retrieval_breakdown`), rather than inventing new ones — see "Session
  instrumentation" below for how each maps onto a live session. As of this
  writing none of the three eval files exist yet (only `knowledge/` evals
  are implemented) — `ttfat_s` and the `retrieval_breakdown_s` stages have
  no external methodology dependency and are built directly in the demo;
  `e2ekd_s`/`keyword_delay_s` require the MoshiRAG paper's Table 17
  keyword-extraction prompt and are deliberately deferred (see "Session
  instrumentation") until `evals/registry/latency/e2ekd.py` implements and
  proves that methodology first — the demo then mirrors it, not the reverse

---

## Repository Structure

```
moshirag-evals/
  core/
    model_interface.py
    retrieval_backend.py
    tts.py
    checkpoint.py
    config.py
  evals/
    runner.py
    registry/
      knowledge/
        open_audio_bench.py
        halu_eval_audio.py
        gsm8k.py
      duplex/
        full_duplex_bench.py
      latency/
        ttfat.py
        e2ekd.py
        retrieval_breakdown.py
      voice/
        wer.py
        speaker_similarity.py
    results/
  scripts/
    run_demo.sh
    print_demo_env.py
    instrumented_server.py
    summarize_demo_session.py
  demo/
    client/          # forked, rebranded moshi-rag client, built via npm/vite
    sessions/         # per-session turns.jsonl + raw_events.jsonl + raw conditioner/server logs
  configs/
    baseline_no_retrieval.yaml
    baseline_with_retrieval.yaml
    checkpoints.yaml
    retrieval_backends.yaml
    tts_backends.yaml
    prompts/
      retrieval_reference_simple.txt
      retrieval_reference_moshi_style.txt
  pyproject.toml
  uv.lock
  Makefile
  .env.example
  README.md
```

---

## Environment and Secrets

Secrets and non-sensitive config are split across two files:

**`.env` (gitignored, secrets only)**
```
GEMINI_API_KEY=...
GCS_BUCKET=your-project-moshirag
GCP_PROJECT=your-gcp-project
```

**`.env.example` (committed, no real values)**
```
GEMINI_API_KEY=
GCS_BUCKET=
GCP_PROJECT=
```

**`configs/checkpoints.yaml` (committed, non-sensitive)**
```yaml
aliases:
  base: gs://${GCS_BUCKET}/checkpoints/base/moshirag-base-bf16
```

`core/config.py` loads `.env` via `python-dotenv` and interpolates environment variables into config values at load time.

---

## Core Abstractions

### `core/model_interface.py`

- Abstract base class `ModelInterface` with two methods:
  - `transcribe(audio: bytes) -> str`
  - `respond(audio_in: bytes) -> tuple[bytes, str, dict]` returning `(audio_out, text_out, metadata)`
- Concrete class `MoshiRAGAdapter(ModelInterface)` that:
  - Accepts a checkpoint path or GCS URI resolved via `core/checkpoint.py`
  - Accepts a retrieval backend instance or `None`
  - Accepts a `generation: dict` of the model/inference-behavior parameters
    listed under "Generation parameters" below — any field omitted falls
    back to the same default that was previously hardcoded, so configs
    written before this existed (or a bare `StubModelAdapter`/`tiny` run)
    keep working unchanged
  - Requires a real, separately launched `server_conditioner` process — conditioning is never in-process. Raises at construction if `REFERENCE_ENCODER_URL` is unset
  - Exposes timing metadata in the returned `metadata` dict including first audio token timestamp
  - **Important:** MoshiRAG runs two parallel token streams simultaneously — an inner monologue text channel and an audio output channel. The adapter must capture both. Read the MoshiRAG repo `run_inference.py` before implementing this class — that script is the ground-truth inference path the paper used, and the adapter must match it exactly before adding instrumentation on top
  - `respond(audio_in: bytes) -> tuple[bytes, str, dict]` is batch mode only (complete audio bytes in, complete audio bytes out) — used by evals. There is no streaming variant: the demo is served entirely by moshi.server directly, not through this adapter
- No eval logic lives here — this is purely model I/O

#### Generation parameters

Every parameter below directly affects MoshiRAG's generation behavior and
was previously hardcoded inside `MoshiRAGAdapter._load_models()`'s
`argparse.Namespace` construction — invisible from any config file. They
now live in a config's `model.generation` block (see "Configs" below) and
are passed through to `MoshiRAGAdapter` as a `dict`. Field names match
`moshi.server`'s own CLI flag names (underscores vs. hyphens) exactly —
confirmed directly against `kyutai-labs/moshi-rag`'s `server.py` argparse
block — so `scripts/print_demo_env.py` (see "Demo" below) can translate
this same block into the demo's CLI flags with a pure name transform, no
per-field mapping table:

| Config field | Maps to (eval) | Maps to (demo CLI flag) | Default |
|---|---|---|---|
| `cfg_coef` | `InferenceJob` args | `--cfg-coef` | `1.0` |
| `stt_wait_time` | `InferenceJob` args | `--stt-wait-time` | `0.5` |
| `rag_timeout` | `InferenceJob` args | `--rag-timeout` | `8.0` |
| `max_reference_tokens` | `InferenceJob` args | `--max-reference-tokens` | `64` |
| `vad_window_size` | `InferenceJob` args | `--vad-window-size` | `4` |
| `vad_threshold` | `InferenceJob` args | `--vad-threshold` | `0.5` |
| `power_threshold` | `InferenceJob` args | `--power-threshold` | `-65` |
| `tail_silence_steps` | `_TAIL_SILENCE_STEPS` (eval-only) | n/a | `25` |

`batch_size` and `init_active_speaker` are deliberately **not** part of
this shared block — they're structural/path-specific, not
performance-affecting knobs comparable across paths: evals always run
`batch_size=1` (one question at a time, offline) while the demo needs
headroom for concurrent live sessions (moshi's own default is `16`), and
evals start `init_active_speaker="user"` (immediately feeding a question)
while the demo starts `init_active_speaker="model"` (model greets first).
Both stay hardcoded/CLI-only per path, same as today.

`tail_silence_steps` has no demo equivalent — it's `respond()`'s own
silence-based turn-termination mechanism (see CLAUDE.md's `_TAIL_SILENCE_STEPS`
notes), not something `moshi.server`'s live, continuous session model needs.

### `core/retrieval_backend.py`

- Abstract base class `RetrievalBackend` with method:
  `retrieve(context: str, history: list[tuple[int, str]] | None = None) -> tuple[str, float]`
  returning `(reference_text, latency_seconds)`. `history` is optional and
  defaults to `None` — see "Context formatting" below for what it's for and
  why the eval path never passes one
- `BACKEND_REGISTRY: dict[str, type[RetrievalBackend]]` and
  `build_retrieval_backend(name: str, backend_def: dict, latency_gate_ms: int | None) -> RetrievalBackend`
  — a factory keyed by the `type` field of a resolved entry from
  `configs/retrieval_backends.yaml` (see "Configs" below). This is what
  makes a config's `model.retrieval.backend` selection actually take
  effect — previously `evals/runner.py`'s `build_model()` always
  constructed `GeminiAPIBackend` directly whenever retrieval was enabled,
  regardless of what `backend:` said, a dead field. Raises on an unknown
  `type`. Called from **both** `evals/runner.py`'s `build_model()` and
  `scripts/instrumented_server.py`'s `main()` (see "Session
  instrumentation" below) — the same factory, the same
  `configs/retrieval_backends.yaml`, builds the backend instance for
  whichever path is running, which is what makes eval and demo retrieval
  genuinely the same code rather than independently-configured
  approximations of each other. This registry is also the intended
  extension point for a future closed/local backend (e.g. an
  OpenAI-compatible HTTP endpoint for a self-hosted LLM, or a RAG server
  over raw markdown) — not implemented now, since no such backend exists
  yet to build or test against, but any future backend registers here and
  becomes selectable purely via config, with no changes needed to either
  caller. It's also the natural place to eventually add live,
  user-switchable backend selection for production (mirroring moshi-rag's
  own native multi-provider mechanism and its already-built, currently
  dormant client UI — see "Not emitted, deliberately:
  `MOSHI_RETRIEVAL_LLMS_JSON`" under `scripts/run_demo.sh` above) — also
  not implemented now
- `describe_backend(name: str, backend_def: dict) -> dict` — a small,
  pure helper returning a display-safe summary of a resolved backend
  definition (`{"name": ..., "type": ..., "model": ...}`, plus
  `"base_url"` only if the backend definition has one — `gemini_api`
  entries don't). Never includes a secret value (`api_key_env` only names
  an env var, never its value, so it's already safe, but isn't included
  either since it's not identifying information). Exists specifically so
  `scripts/instrumented_server.py` has **one** place to compute "what
  backend is this session using" for display, instead of two independent
  copies — see "Session instrumentation" below for the bug this fixes
- Concrete class `GeminiAPIBackend(RetrievalBackend)` that:
  - Calls Gemini API with conversation context
  - Records and returns latency of every call
  - Raises `LatencyGateError` if latency exceeds configurable `latency_gate_ms`
  - Logs every call duration regardless of whether gate is breached
  - Retries failed calls up to 3 times with exponential backoff before raising
  - Reads its prompt template from the file named by its backend
    definition's `prompt_template` field (defaults to
    `configs/prompts/retrieval_reference_simple.txt`, the current
    hardcoded prompt text extracted verbatim) rather than a string literal
    in the class — same file-based-template convention moshi-rag's own
    `LLMReferenceGenerator` uses for its bundled prompts
  - Before formatting the prompt, passes `context`/`history` through
    `format_context()` (see "Context formatting" below) using its own
    backend definition's `context_formatting` settings, then substitutes
    the result into the template's `{context}` placeholder. If the
    resulting `num_turns` is `0` (nothing left after the configured
    drops — e.g. a lone greeting), skips the Gemini call entirely and
    returns `("", 0.0)` — the same short-circuit moshi's own
    `generate_reference_text()` does on an empty context, not new
    behavior invented here
  - Shares its Gemini call-tuning (`thinking_config.thinking_level="low"`,
    `max_output_tokens=1024` — both workarounds for a real Gemini 3.x quirk,
    not experiment variables) with `core/llm_judge.py`'s `call_gemini()`
    via one shared helper, rather than the two independent copies that
    exist today
- `NullBackend(RetrievalBackend)` that returns empty string immediately
  regardless of `history` (ignored), used for Config A (no retrieval)

#### Context formatting

Three of the behaviors moshi-rag's own `LLMReferenceGenerator.process_reference_text()`
performs before ever calling its retrieval LLM (see the "On prompt parity"
note below for the full background) are available, independently
config-gated, to any `RetrievalBackend` via a shared, pure, module-level
function:

```python
def format_context(
    context: str,
    history: list[tuple[int, str]] | None,
    *,
    drop_leading_incomplete_turn: bool = False,
    drop_trailing_incomplete_turn: bool = False,
    thread_reference_history: bool = False,
    history_max_entries: int | None = None,
    role_labels: dict[str, str] = {"user": "Human", "model": "moshi"},
) -> tuple[str, int]:
    """Returns (formatted_context, num_turns) — num_turns is the post-drop
    turn count, mirroring process_reference_text()'s identical return
    shape, needed by any caller doing reference-history bookkeeping."""
```

Not backend-specific — any current or future `RetrievalBackend` subclass
can call it. Its keyword arguments are exactly the fields of a backend's
`context_formatting` block in `configs/retrieval_backends.yaml` (see
"Configs" below), spread in directly:

| Field | Type | Default | Effect |
|---|---|---|---|
| `drop_leading_incomplete_turn` | bool | `false` | Drops a leading `moshi:`-role turn (a model-initiated greeting) before formatting |
| `drop_trailing_incomplete_turn` | bool | `false` | Drops a trailing `moshi:`-role turn — moshi's real fix for `<ret>` firing mid-utterance, since the model is still speaking when retrieval triggers and the raw context would otherwise end mid-sentence |
| `thread_reference_history` | bool | `false` | Re-inserts prior turns' retrieved reference text back into the formatted context, at the point in the transcript where each was originally retrieved |
| `history_max_entries` | int \| null | `null` | Caps how many prior history entries `thread_reference_history` re-inserts (oldest dropped first). `null` matches moshi's own real behavior (unbounded growth) — a known, disclosed latent risk on long-running demo sessions that moshi itself doesn't guard against either; set a number to cap it |
| `role_labels` | dict | `{user: "Human", model: "moshi"}` | Labels used when reformatting turns into a transcript — moshi's own choice by default; a new `prompt_template` expecting different labels (e.g. `"User:"/"Assistant:"`) can override these without touching Python |

**History threading is structurally a no-op on the eval path, not a
symmetric toggle.** `thread_reference_history` re-inserts *this session's*
prior retrieved reference text — but eval's `respond()` calls are
deliberately isolated per question (see the "RESOLVED (Phase 2)" section
above: making `respond()` session-like was explicitly rejected, since a
shared session would leak one eval question's retrieval context into
another's and corrupt scores). `core/model_interface.py`'s
`_patch_rag_manager` (eval path) never has a `history` list to pass —
`RetrievalBackend.retrieve()`'s `history` parameter defaults to `None`
there and stays `None`. Setting `thread_reference_history: true` on a
backend used only for evals has no observable effect; it only does
something real on the demo path, where one `RAGManager` instance's
`self._history` genuinely accumulates across a session's turns, same as
moshi's own unpatched behavior — see "Session instrumentation" below for
how the demo path wires this up.

**On prompt parity with the demo path (resolved at two levels, not just
documented):** an earlier direction here considered making the eval-path
prompt independently tunable via `prompt_template` while leaving the demo
on moshi-rag's own native `LLMReferenceGenerator` (bundled
`original`/`simplified` templates only, selected via `prompt_style`), and
explicitly accepting that the two paths would ask the retrieval LLM for
differently-shaped reference text. That direction was dropped: there is no
point tuning a retrieval prompt in eval configs if the result can never be
observed in the demo. Two things now make that a non-issue instead of a
disclosed gap:

1. **Code-path parity.** `scripts/instrumented_server.py` patches
   `RAGManager.get_reference_text` (see "Session instrumentation" below)
   so the demo's actual retrieval call goes through this same
   `GeminiAPIBackend`/`RetrievalBackend`, built from this same registry —
   not an equivalent mechanism, the literal same code path. moshi's own
   `LLMReferenceGenerator` is only ever invoked once more, for its
   unavoidable startup `warmup()` call (see `scripts/run_demo.sh`'s
   "Config-derived environment" note on why `LLM_BASE_URL`/`LLM_MODEL_NAME`/
   `LLM_API_KEY` are still required despite being otherwise unused) — real
   retrieval never reaches it again.
2. **Formatting parity, opt-in.** "Context formatting" above makes
   `process_reference_text()`'s own behaviors (turn-trimming,
   reference-history threading) available to `GeminiAPIBackend` directly,
   config-gated per backend definition. Two starting `prompt_template`
   files are shipped, meant to be paired with matching
   `context_formatting` settings rather than mixed arbitrarily —
   `configs/prompts/retrieval_reference_simple.txt` (today's flat
   `"Context:\n{context}"` prompt, pairs with every `context_formatting`
   field at its default/off) and `configs/prompts/retrieval_reference_moshi_style.txt`
   (a completion-style prompt ending in a bare trailing `Reference:` cue
   with moshi's own length/formatting guidelines, modeled on its bundled
   template, pairs with the three boolean toggles on) — see "Configs"
   below for both. New prompt files, differently named as experimentation
   proceeds, are expected; each should note in a comment which
   `context_formatting` settings it assumes, since the two layers only
   reproduce a coherent result when authored together.

### `core/tts.py`

Added to support `knowledge.gsm8k` (see "Eval Implementations" below), and
deliberately built as a reusable scaffold rather than a one-off — GSM8K is
the first eval whose source dataset is text-only with no accompanying
audio, and it won't be the last (a future custom eval was raised as the
motivating second use case when this was designed). Mirrors
`core/retrieval_backend.py`'s registry shape closely, since the same
config-driven-swap need applies to both:

- Abstract base class `TTSBackend` with method `synthesize(text: str) ->
  bytes` returning a complete mono WAV file's bytes at MoshiRAG's own
  sample rate (`core/model_interface.py`'s `_SAMPLE_RATE = 24000` — no
  resampling needed, since Gemini's TTS output is natively 24kHz/mono/16-bit
  PCM too, confirmed against Google's current API docs)
- `TTS_REGISTRY: dict[str, type[TTSBackend]]` and `build_tts_backend(name:
  str, backend_def: dict) -> TTSBackend` — same factory-keyed-by-`type`
  pattern as `retrieval_backend.py`'s `BACKEND_REGISTRY`/
  `build_retrieval_backend`. Not implemented now, but the intended
  extension point for a local engine (e.g. Kokoro) if one is ever needed —
  registers here, becomes selectable purely via
  `configs/tts_backends.yaml`, no changes needed to any caller. Raises on
  an unknown `type`, same as the retrieval registry
- Concrete class `GeminiTTSBackend(TTSBackend)`:
  - Calls Gemini's TTS-capable `generate_content` path
    (`response_modalities=["AUDIO"]`, a `speech_config` naming a prebuilt
    voice) — the same `client.models.generate_content()` call shape
    `core/llm_judge.py`/`core/retrieval_backend.py` already use, not
    Google's newer `client.interactions.create()` API (confirmed both
    exist as of this writing; Google's docs mark `generate_content`-based
    TTS "legacy" and recommend the Interactions API for new work, but this
    project deliberately stays on `generate_content` for one reason: it's
    the one Gemini call shape already used everywhere else in this
    codebase, and learning a second SDK surface for a single feature isn't
    worth it pre-emptively. Revisit only if the legacy path is actually
    sunset, not before)
  - Requires no new dependency — `google-genai` is already a base
    dependency (used by `core/llm_judge.py`/`core/retrieval_backend.py`
    already), so adding this needs no `uv add`
  - Extracts raw PCM16 bytes from `response.candidates[0].content.parts[0].inline_data.data`
    and wraps them as a WAV via `core/model_interface.py`'s existing
    `_pcm16_to_wav()` (imported, not duplicated — same helper the browser
    client's raw-PCM path already uses)
  - Retries failed calls up to 3 times with exponential backoff — same
    convention as `GeminiAPIBackend`/`call_gemini()`
  - Config fields: `model` (a Gemini TTS-capable model name — see
    `configs/tts_backends.yaml` below for the shipped default), `voice` (a
    prebuilt voice name, e.g. `"Kore"` — pinned per backend definition so
    synthesized audio is reproducible across runs, not re-rolled per call),
    `api_key_env`
- `synthesize_cached(text: str, backend: TTSBackend, backend_name: str,
  backend_def: dict, cache_dir: Path) -> bytes` — the reusable scaffold
  itself. Computes a cache key from a hash of `backend_def` (`type`,
  `model`, `voice` — everything that affects the resulting audio) plus the
  text being synthesized, so switching backend/model/voice in config
  naturally starts populating a new cache namespace rather than silently
  reusing stale audio; nothing needs manual invalidation. Looks for
  `cache_dir / f"{key}.wav"`; if present, returns its bytes directly with
  no API call. Otherwise calls `backend.synthesize(text)`, writes the
  result atomically (temp file + rename, so a killed process mid-write
  can't corrupt a cache entry future runs would otherwise trust), and
  returns it. `knowledge.gsm8k` calls this directly; a future custom eval
  needing synthesized audio does too, with no per-eval caching logic to
  write from scratch
  - Cache lives at `./tts_cache/` (repo root, sibling to
    `checkpoint_cache/`) — generated locally, not committed. Gitignored and
    excluded from `make sync`'s rsync (see Makefile section below), same
    treatment as `checkpoint_cache/`
- `describe_tts_backend(name: str, backend_def: dict) -> dict` — same
  small, pure display-summary helper as `retrieval_backend.py`'s
  `describe_backend()` (`{"name", "type", "model", "voice"}` — `voice`
  included since, unlike a secret, it's real, useful identifying
  information for reproducing a given run's audio; never a secret value),
  for the same reason: one place to compute "what TTS backend produced this
  audio" instead of a second independent copy wherever it's reported (see
  `knowledge.gsm8k`'s `tts_backend` result metadata field below)

### `core/checkpoint.py`

- `resolve_checkpoint(uri: str) -> str` that:
  - Handles `gs://` URIs by downloading to `./checkpoint_cache/` and returning local path
  - Passes through local paths unchanged
  - Resolves named aliases from `configs/checkpoints.yaml`

### `core/config.py`

- Loads YAML configs
- Loads `.env` via `python-dotenv`
- Interpolates environment variables into config values
- Resolves checkpoint aliases
- Resolves `model.retrieval.backend` against `configs/retrieval_backends.yaml`
  (same alias-file pattern as checkpoints — loaded once, looked up by
  name), attaching the resolved backend definition at
  `config["model"]["retrieval"]["_resolved_backend"]` so callers
  (`evals/runner.py`'s `build_model()`, `scripts/instrumented_server.py`'s
  `main()` — not `scripts/print_demo_env.py`, which only translates
  `model.generation` into demo CLI flags, see "Demo" below) don't each
  re-read the file or re-implement the lookup. Raises if
  `model.retrieval.backend` names an entry that doesn't exist in
  `retrieval_backends.yaml`
- Resolves `tts.backend` against `configs/tts_backends.yaml` the same way,
  attaching the resolved definition at `config["tts"]["_resolved_backend"]`.
  Unlike retrieval, this isn't gated behind an `enabled` flag — `tts:` is
  simply absent from a config that runs no TTS-dependent eval. Only
  resolved if `config.get("tts", {}).get("backend")` is actually set; a
  config that includes `knowledge.gsm8k` in its `evals:` list without a
  `tts:` block is a config error the eval itself raises clearly at run
  time, not something `core/config.py` cross-checks against the `evals:`
  list
- Validates required fields including `latency_gate_ms`

---

## Eval Runner

### `evals/runner.py`

**Running evals:**
```
python runner.py --config configs/baseline_with_retrieval.yaml [--mode tiny|smoke|sample|full]
```

**Comparing runs:**
```
python runner.py --compare results/run-a.json results/run-b.json
```

**Mode behavior:**

| Mode | Sample size | Target runtime | Purpose |
|---|---|---|---|
| `--mode tiny` | 1 question per subset | under a minute per eval | Rapid dev-iteration sanity ping while building/debugging a single eval against a real checkpoint — not a substitute for `smoke`, sample size too small to trust any score from |
| `--mode smoke` | 5 questions per subset, 3-5 duplex scenarios | ~10 minutes | Sanity check pipeline works end to end |
| `--mode sample` | 100 per knowledge subset, 200 for latency | ~1-2 hours | Regression detection between checkpoints |
| `--full` | Complete datasets | Overnight (~8-16 hours) | Final baseline and major checkpoint evals |

Default mode if flag is omitted: `sample`

`tiny` added during eval implementation once real per-question latency on actual
hardware (checkpoint loading, real STT/generation, retrieval-trigger stalls) made
even `smoke`'s 5-per-subset sample noticeably slow (~10 minutes) for the tight
build-test-fix loop of standing up one eval at a time — `smoke`'s own definition
and purpose are unchanged.

**Runner behavior:**
- Discovers and instantiates evals listed in config from the registry by name
- Instantiates `ModelInterface` from config
- Wraps the instantiated model's `respond()` with an in-memory,
  process-lifetime cache keyed by a hash of `audio_in` bytes
  (`_cache_model_respond()`), applied once, here, before any eval runs —
  not inside individual eval files. `latency.ttfat` and
  `latency.retrieval_breakdown` deliberately sample from
  `knowledge.open_audio_bench`'s own question pool (see their own sections
  above), so the exact same audio can legitimately get fed to `respond()`
  more than once across different evals in one run — confirmed on a real
  VM run: a `tiny`-mode run fed the same first TriviaQA question through
  `respond()` three separate times (once per eval touching that pool),
  each a real GPU inference pass plus a real Gemini retrieval call, for no
  additional signal a repeated call on byte-identical input can't add.
  Scoped to one process's lifetime only (a plain dict, never persisted) —
  a restarted/resumed run starts with an empty cache; each eval's own
  resumability via `last_completed_index` is unaffected and still governs
  which items get (re-)processed across restarts
- Runs each eval, collects `EvalResult` objects
- On Gemini API failure: retries up to 3 times with exponential backoff, then writes partial results to JSON and exits gracefully with a clear error message indicating where it stopped
- On restart: detects existing partial result file for the same checkpoint and config, resumes from last completed question rather than starting over
- Writes one JSON output file per run to `evals/results/` tagged with checkpoint, timestamp, git hash, config, and mode used
- Comparison CLI computes deltas and derived metrics including `rag_lift_pp` on the fly — no pre-computation in individual run files

**Direction of improvement is hardcoded per metric inside each eval class** (lower is better for TOR and latency, higher for accuracy and GPT scores) and used by the comparison CLI to render `✓` and `↓` indicators.

**Judge model configuration:** the LLM judge model name comes from a
config's top-level `judge.model` field (see "Configs" below), read by each
knowledge eval from the `config: dict` already passed into `run()` —
previously a `_JUDGE_MODEL = "gemini-3.5-flash"` module-level constant
duplicated identically in both `open_audio_bench.py` and
`halu_eval_audio.py`. Gemini call-tuning (`thinking_level`,
`max_output_tokens`) is not a `judge` config field — see
`core/retrieval_backend.py`'s note on why those stay a shared code default
rather than a per-run knob.

### `EvalResult` dataclass

```python
@dataclass
class EvalResult:
    eval_name: str
    scores: dict
    metadata: dict
    errors: list
    completed: bool
    last_completed_index: int  # for resumability
```

### Eval base class

- Abstract class `BaseEval` with method `run(model: ModelInterface, config: dict, mode: str) -> EvalResult`
- Each eval is a single file in the registry implementing this class
- Runner discovers evals by name — adding a new eval means dropping a file, no other changes

---

## Eval Implementations

### Knowledge

**`evals/registry/knowledge/open_audio_bench.py`**
- Downloads from HuggingFace `baichuan-inc/OpenAudioBench` if not cached. (Original
  intent, per an earlier draft of this spec, was `AudioLLMs/OpenAudioBench` — the
  `AudioLLMs` org publishes many AudioBench-framework eval sets on HF and was assumed
  to host this one too, following that pattern. Verified via the HF API that no such
  repo exists — `AudioLLMs` hosts 40 other datasets, none of them OpenAudioBench or its
  TriviaQA/WebQ/LlamaQ subsets. `baichuan-inc/OpenAudioBench` is the actual, only public
  HF dataset matching this spec's description: LlamaQ/WebQ/TriviaQA/AlpacaEval/
  ReasoningQA subsets, audio + CSV metadata per subset under `eval_datas/<subset>/`,
  same underlying benchmark, just published under its originating org instead.)
- Runs TriviaQA, WebQ, and LlamaQ subsets
- For each question: feeds audio file into `model.respond()`, collects transcribed text output
- Scores each response via Gemini API LLM judge (model from config's `judge.model`, see "Judge model configuration" above) using prompt from MoshiRAG paper appendix (Table 16)
- Returns per-subset accuracy scores
- Respects mode sample size limits and resumability via `last_completed_index`

**`evals/registry/knowledge/halu_eval_audio.py`**
- Downloads from HuggingFace `kyutai/HaluEvalAudio_1000` if not cached
- Same pattern as OpenAudioBench — audio in, transcribed text out, LLM judge scoring
- Scores retrieved reference separately (ref. score) and final response (resp. score), matching paper Table 1 format
- Per-run JSON records raw `ref_acc` and `resp_acc` only — `rag_lift_pp` is computed by comparison CLI when diffing Config A vs Config B runs
- Respects mode sample size limits and resumability

**`evals/registry/knowledge/gsm8k.py`**

Not part of the MoshiRAG paper's own eval suite (the paper's Table 1 covers
TriviaQA/WebQ/LlamaQ/HaluEvalAudio only) — added independently to track a
standard, widely-cited benchmark number, so this file's scoring methodology
is this project's own design, not a paper-verbatim reproduction the way the
Table 16 judge prompt is elsewhere in this section.

- Downloads from HuggingFace `openai/gsm8k`, `main` config (not
  `socratic`, which adds sub-question annotations this eval doesn't use),
  `test` split — 1,319 examples, columns `question` and `answer` (`answer`
  is a full worked solution ending in a literal `#### <number>` line, GSM8K's
  own convention for marking the final answer)
- **GSM8K has no existing audio version and isn't part of the MoshiRAG
  paper's own audio benchmark suite** (confirmed: no audio GSM8K dataset is
  publicly available, and `baichuan-inc/OpenAudioBench`'s own `reasoning_qa`
  subset — checked directly against its real CSV rows — is a different,
  Chinese-language, 202-example, self-constructed grab-bag of comparison/
  riddle/geometry questions with free-text answers, not a GSM8K derivative
  and not comparable to any published GSM8K score). Question audio is
  synthesized via `core/tts.py`'s `synthesize_cached()`, using the config's
  resolved `tts` backend (see "Configs" below), and disk-cached
  indefinitely — each of the 1,319 questions is synthesized at most once,
  ever, across every future run
- For each question: synthesizes audio for `question`, feeds it into
  `model.respond()`, collects transcribed text output — same pattern as
  every other knowledge eval
- **Scoring is exact-match on the final number, not the Correct/Incorrect
  Gemini judge the other two knowledge evals use** — deliberate, since a
  benchmark whose entire point is one precise numeric answer is better
  served by extract-then-compare than a free-form verdict call:
  1. Ground truth: parsed directly from `answer`'s trailing `#### <number>`
     via regex, stripping commas — no LLM call needed, this is deterministic
     data-file parsing. Raises if a row's `answer` doesn't match the
     expected format (a parsing bug in this code, not a legitimate scoring
     outcome, since every real GSM8K row has this exact convention)
  2. Model's answer: extracted from the transcribed response via a small
     Gemini call, reusing `core/llm_judge.py`'s `call_gemini()` (same
     retry/backoff infra as every other Gemini call in this codebase) with
     a purpose-built extraction prompt this project owns (not a paper
     prompt):
     ```
     Extract the final numeric answer from the response below. Respond
     with ONLY the number — digits only, no words, no units, no currency
     symbols, no punctuation other than a decimal point or minus sign. If
     the response contains multiple numbers, extract the one that
     represents the final answer to the question. If no numeric answer can
     be determined, respond with exactly: NONE

     Question:
     {question}

     Response:
     {response}
     ```
     A dedicated extraction call (rather than a regex over the transcribed
     text directly) handles MoshiRAG spelling a number out as words rather
     than digits, which a plain regex can't reliably catch — same rationale
     `latency/e2ekd.py`'s keyword extraction already established for a
     different field. Reads the extraction model from `config["judge"]["model"]`,
     same as every other knowledge eval's judge model — no new config field
  3. Both values normalized (strip `$`/`,`/`%`/whitespace, parse as float)
     and compared for equality — GSM8K's ground truth is always an exact
     integer or simple decimal, so no fuzzy tolerance is needed. A `NONE`
     extraction result (or anything that fails to parse as a number) scores
     as incorrect, not an error — this is a legitimate, expected outcome for
     a genuinely non-numeric or degenerate response, same status as any
     other wrong answer
  - As with `latency/e2ekd.py`'s keyword extraction, this extraction step's
    own accuracy isn't independently validated by this spec — review a
    handful of real transcript entries after the first real-checkpoint run
    before trusting `gsm8k_acc` at scale, same caution as any new
    LLM-mediated scoring step, though this is a much simpler extraction task
    than e2ekd's and doesn't warrant a dedicated `--spot-check` flag the way
    that one does
- Same `degenerate_silence` flagging, resumability (`last_completed_index`/
  `_progress`), and mode sample sizes as `open_audio_bench.py` (`tiny`: 1,
  `smoke`: 5, `sample`: 100; `full`: all 1,319)
- Returns `gsm8k_acc` (`METRIC_DIRECTIONS = {"gsm8k_acc": "higher"}`)
- Retrieval is orthogonal here, same as for every other knowledge eval —
  GSM8K questions are self-contained math word problems needing no external
  knowledge, but whatever the config's `model.retrieval` says still applies
  normally (this eval doesn't force retrieval on or off)

### Duplex

**`evals/registry/duplex/full_duplex_bench.py`**
- Integrates with Full-Duplex-Bench v1 and v1.5 — read their repo interface before implementing the wrapper
- Wraps `ModelInterface` to conform to Full-Duplex-Bench's expected model calling convention
- Runs pause track, backchannel track, turn taking track, and user interruption track
- Returns TOR, latency, GPT score, and JSD metrics matching paper Table 2 format
- Smoke mode runs 3-5 scenarios only

### Latency

**`evals/registry/latency/ttfat.py`**
- Captures timestamp at end of user utterance and at first audio token emitted
- Delta is TTFAT in seconds
- Default: 200 questions sampled from the knowledge eval question set; `--full` runs all ~1500
- Reports mean, P50, P95

**`evals/registry/latency/e2ekd.py`** — **deliberately deferred, not implemented**
(2026-07-23). Original target design, kept below for reference if this is
picked up later:
- Two-step pipeline:
  1. Gemini API extracts keyword from transcribed response using prompt from paper appendix (Table 17)
  2. `nvidia/parakeet-tdt-0.6b-v2` gives onset timestamp of that keyword in audio output
- E2EKD = TTFAT + keyword delay
- Default: 200 questions; `--full` runs complete set
- Reports mean TTFAT, mean keyword delay, mean E2EKD, P95 E2EKD
- `--spot-check` flag outputs 20 examples as numbered list of `(transcribed response, extracted keyword, verdict: correct/incorrect)` for human review before trusting metric at scale
- `spot_check_completed: false` flag in result metadata surfaces as reminder in console output until manually flipped to `true`

**Why deferred:** the keyword-onset step requires `nemo_toolkit["asr"]`
(NVIDIA NeMo, to run `parakeet-tdt-0.6b-v2`) — a large, opinionated
framework (its own ASR/NLP/TTS collections, `pytorch-lightning`/
`hydra-core`/`torchmetrics`/etc. transitive deps) that pins its own
`torch`/`torchaudio` compatible ranges. This project already has `torch`
pinned carefully for `moshi` compatibility, and CLAUDE.md documents a real
precedent (`moshi`'s own `uv pip install` step silently downgrading
`transformers`, unnoticed until a plain `uv run` — no `--all-extras` —
left it that way) for exactly this kind of second-large-framework version
conflict. Weighed against that concrete, hard-to-test-in-advance risk (untestable
in this environment either way — no GPU, aarch64 musl, same reason `torch`
itself can't install here), the user judged E2EKD's incremental diagnostic
value too narrow to justify it right now: `latency.ttfat` (TTFAT) and
`latency.retrieval_breakdown` (`asr_wait_s`/`api_call_s`/
`context_injection_s`) already cover the mechanism most likely to regress
from fine-tuning a RAG-augmented model — whether `<ret>` fires appropriately
and how the retrieval round-trip behaves. The one thing E2EKD's keyword-delay
component would catch that those don't is a change in how long the model
rambles *after* receiving retrieved context before actually stating the
answer (a fine-tune could plausibly learn to hedge/pad more without
`<ret>` behavior or retrieval latency moving at all) — a real but narrower,
more speculative regression than "did retrieval break," and one
`duplex.full_duplex_bench`'s turn-taking/interruption-latency tracks
partially proxy for anyway from a different angle. Revisit if a lighter
word-timestamp option surfaces, or if `<ret>`-behavior/retrieval-timing
metrics stay stable across a fine-tune but end-to-end factuality or
qualitative review still suggests a real latency regression slipping
through.

**`evals/registry/latency/retrieval_breakdown.py`** — implemented
(2026-07-23), reusing `latency.ttfat`'s `_load_question_pool()` and
`knowledge.open_audio_bench`'s `_load_audio_bytes()` rather than a third
independent copy of the same pool-building logic.
- Reports the same three stages as the demo's `retrieval_breakdown_s`
  (`asr_wait_s`, `api_call_s`, `context_injection_s`) for consistency, but
  **not** via the demo's mechanism — `RAGManager._background_task` is
  never patched on the eval path (only `get_reference_text` is;
  `core/model_interface.py`'s `_TimedInferenceJob._patch_rag_manager`
  doesn't touch it), so there's no existing hook there to reuse.
  `context_injection_s` is straightforward regardless: same technique as
  the demo, a second, redundant, timed call to `core/retrieval_backend.py`'s
  `format_context()` (see "Context formatting" above) around whatever eval
  code calls `MoshiRAGAdapter.respond()`. `api_call_s` is `retrieval_latency_s`
  directly, no subtraction needed — confirmed `GeminiAPIBackend.retrieve()`'s
  own timer already starts *after* its `format_context()` call, so there's
  no double-counting to correct for on this path (unlike the demo's own,
  separate, now-superseded timing mechanism)
- **`asr_wait_s`'s open design question (previously unresolved by this
  spec) is now resolved**: `RAGManager._background_task`'s `wait_steps`-based
  delay between a `<ret>` trigger and grabbing the final context is real,
  unpatched code shared by `InferenceJob` and `Channel` alike — it
  genuinely happens on the one-shot `respond()` path too, it just wasn't
  being *timed* there yet. Fixed via a narrow addition to the two
  already-existing allowed patches in `_TimedInferenceJob` (not a new
  patch): `_patch_output_loop()`'s `<ret>` branch now stamps
  `self.retrieval_trigger_ts = time.perf_counter()`, and
  `_patch_rag_manager()`'s `_patched_get_reference_text` computes
  `self.asr_wait_s = t0 - retrieval_trigger_ts` at entry. Both surface into
  `respond()`'s metadata (`asr_wait_s`, 0.0 default when no `<ret>` fired).
  Same last-trigger-wins fidelity limitation `rag_trigger_step` already has
  for multiple `<ret>`s in one turn — not a new gap introduced here
- Runs fine over `NullBackend` (retrieval disabled) — `<ret>` still fires
  and gets timed regardless of backend, so `configs/baseline_no_retrieval.yaml`
  legitimately keeps this eval in its `evals:` list; `api_call_s` is simply
  near-zero in that case (`NullBackend.retrieve()`'s instant empty return),
  not an error condition
- Reports mean and P95 per stage
- Flags any run where P95 total exceeds `latency_gate_ms` — this surfaces in console output, JSON errors list, and is treated as a correctness concern not just a performance one. No `evals/runner.py` changes were needed for this — the gate-check console/compare-CLI handling and every metric key this eval reports were already generic/present from the original spec

### Voice (Step 2 only — registered but excluded from baseline configs)

**`evals/registry/voice/wer.py`**
- Transcribes MoshiRAG audio output using Whisper
- Compares against MoshiRAG inner monologue text token stream as ground truth
- Reports WER
- Only meaningful after fine-tuning — excluded from baseline configs via `enabled: false`

**`evals/registry/voice/speaker_similarity.py`**
- Uses SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) downloaded automatically from HuggingFace
- **Critical:** resamples MoshiRAG audio from 24kHz to 16kHz before passing to ECAPA-TDNN — skipping this produces invalid scores
- Computes cosine similarity against reference audio samples taken from base MoshiRAG checkpoint output
- Reference samples path configurable in config
- `--validate-sensitivity` flag runs against a known-degraded audio sample to confirm metric moves in expected direction before trusting it in production
- Only meaningful after fine-tuning — excluded from baseline configs via `enabled: false`

---

## Demo

The demo provides a browser-based full-duplex voice interface for ad-hoc
testing, edge-case probing, and stakeholder demos. It runs on the VM as
kyutai-labs/moshi-rag's `moshi.server` + `moshi.server_conditioner` — the
main server process is wrapped by a thin, narrow instrumentation layer
(`scripts/instrumented_server.py`, see below), not reimplemented or
forked; `server_conditioner` remains fully unmodified. Most of that layer
is logging-only, with one deliberate, bounded exception (retrieval
routing — see "Session instrumentation" below); step-loop/turn-taking/VAD
mechanics stay moshi.server's own, unmodified logic throughout.
macOS (or any machine with a browser) connects via SSH tunnel — no Python,
no model weights on the client side.

Every session persists a structured log and both processes' raw output to
`demo/sessions/<session_id>/`, and drives a maintained fork of the web
client that surfaces retrieval/latency instrumentation live — see
"Session instrumentation" and "Web client" below. Neither of these existed
prior to this instrumentation work: earlier iterations of this demo had no
persisted logs at all, so a session's console output was lost once its
tmux session ended or scrolled past the pane's buffer.

Access pattern:
```
ssh -L 8998:localhost:8998 user@vertex-vm
# then open http://localhost:8998
```

`localhost` is treated as a secure context by browsers, so microphone
access works without TLS certificates.

### `scripts/run_demo.sh`

- Resolves a checkpoint alias via `core/checkpoint.py`, generates a
  session id (same `%Y-%m-%dT%H-%M-%SZ` timestamp format eval run ids use)
  and creates `demo/sessions/<session_id>/`
- Builds `demo/client/` (npm/vite build — see "Web client") and launches
  `moshi.server_conditioner` (port 8001, unmodified) and
  `scripts/instrumented_server.py` (port 8998, wraps `moshi.server`) in a
  detached tmux session, pointed at the built client via `--static`
- Tees both processes' stdout/stderr into
  `demo/sessions/<session_id>/{conditioner,server}.log` so raw logs
  survive independent of tmux's lifecycle/scrollback, and prints the
  session dir path in its final instructions (mirroring the eval console's
  "results written to: ..." line)
- `--checkpoint`, `--stt`, `--rag-timeout`, `--conditioner-only`, and
  `--config <path>` flags (default `configs/baseline_with_retrieval.yaml`
  for `--config`) — see the script's own header comment
- Turn-taking/VAD/step-loop *mechanics* remain moshi.server's own,
  unmodified logic — only *retrieval* (which backend and prompt answer a
  `<ret>` trigger) is redirected, via the `RAGManager.get_reference_text`
  patch described under "Session instrumentation" below; instrumentation
  proper (logging/session files/live client push) stays additive on top of
  that

**Config-derived environment (`scripts/print_demo_env.py`):** rather than
hardcoding `LLM_MODEL_NAME=gemini-3.5-flash` and the generation-affecting
CLI flags directly in the shell script (as it did before this pass),
`run_demo.sh` shells out to a new small Python helper — same pattern it
already uses for `resolve_checkpoint` — that loads the `--config` YAML and
prints a shell-sourceable block covering only:
- `LLM_BASE_URL` / `LLM_MODEL_NAME` / `LLM_API_KEY`, resolved from the
  config's `model.retrieval.backend` entry — needed **only** because
  `ServerState.__init__` unconditionally constructs its own
  `LLMReferenceGenerator` and `ServerState.warmup()` makes one real,
  synchronous API call against it before the server ever accepts a
  connection (same requirement `core/model_interface.py`'s
  `_DEFAULT_LLM_BASE_URL`/`_DEFAULT_LLM_MODEL_NAME` notes already document
  for the eval path). These values are otherwise **vestigial** once the
  `get_reference_text` patch is active — `LLMReferenceGenerator` is never
  called again after that one startup warmup call, so this is "keep
  `ServerState.warmup()` from raising," not "configure retrieval." Worth a
  clear comment in the generated env block so this isn't mistaken for the
  actual retrieval configuration later
- The generation CLI flags (`--cfg-coef`, `--stt-wait-time`, `--rag-timeout`,
  `--max-reference-tokens`, `--vad-window-size`, `--vad-threshold`,
  `--power-threshold`) derived from the config's `model.generation` block —
  see `core/model_interface.py`'s "Generation parameters" table for the
  field-name mapping

**Not emitted, deliberately: `MOSHI_RETRIEVAL_LLMS_JSON`.** This is
moshi-rag's own native multi-provider mechanism (server-side
`RetrievalProfile` config + a real, already-built live-switching UI in our
`demo/client/` fork — `useRetrievalBackendChoice.ts`,
`SearchPanel.tsx`'s `RetrievalBackendTabs`, dormant since it's never been
populated). It's a real, working feature, confirmed against actual source,
but it only affects `LLMReferenceGenerator.generate_reference_text()` —
which the `get_reference_text` patch below bypasses entirely. Setting it
anyway (e.g. to reach `prompt_style`, an earlier direction considered and
dropped) would render the client's tabs visible and clickable but
functionally inert, since the patched `get_reference_text` never consults
`RAGManager._active_profile_id`. **Not implemented in this pass**, but
worth recording as a real extension point: live open/closed backend
switching in production would mean re-pointing that same already-built
client UI (same kind-byte-4 metadata message shape) at a switch inside our
own `RetrievalBackend` layer instead of moshi's profiles — see
`core/retrieval_backend.py`'s registry note.

`run_demo.sh`'s own `--rag-timeout`/`--stt` flags remain as explicit
overrides layered on top of whatever the config produced, not the sole
source — so `bash scripts/run_demo.sh --config configs/baseline_with_retrieval.yaml --rag-timeout 12`
still works as an ad-hoc override. `--batch-size` and `--init-active-speaker`
are not derived from the shared config (see `core/model_interface.py`'s
note on why those two stay demo-specific) — `run_demo.sh` keeps setting
`--init-active-speaker model` itself, as it does today.

`scripts/instrumented_server.py` additionally needs a `DEMO_CONFIG` env var
(alongside the existing `DEMO_SESSION_DIR`/`DEMO_CHECKPOINT`), set by
`run_demo.sh` to the same `--config` path — used directly in Python (via
`core/config.py` + `core/retrieval_backend.py`'s `build_retrieval_backend`)
to construct the one `RetrievalBackend` instance the `get_reference_text`
patch below routes through. This doesn't need shell-env translation the
way the CLI flags above do, since `instrumented_server.py` is Python and
can load the YAML directly.

### Session instrumentation

**`scripts/instrumented_server.py`** forwards all CLI args through to
moshi-rag's real `moshi.server` entrypoint unchanged, then, before
invoking its serve function:

- **Patches `RAGManager.get_reference_text` to actually route retrieval
  through `core/retrieval_backend.py`, not just to log it** — same
  technique and signature as `core/model_interface.py`'s
  `_patch_rag_manager` (`context: str -> (context, ref_text, elapsed,
  backend_label)`), ported near-verbatim, since `RAGManager` is a plain
  Python object (not the PyO3-native kind instance-patching can't touch —
  confirmed the hard way: an early attempt to wrap `Channel.opus_writer.append_pcm`
  directly failed, since `sphn.OpusStreamWriter` is a compiled/native
  extension type) and `Channel`
  constructs its own fresh `RAGManager` per connection exactly like
  `InferenceJob` does, confirmed directly against real
  `inference_utils/channel.py` and `inference_utils/rag_manager.py`
  source. Installed inside the existing `_patch_channel`'s `patched_init`,
  right where `self.rag_manager` becomes available. The `RetrievalBackend`
  instance is built once at server startup (`main()`, via
  `core/config.py` + `core/retrieval_backend.py`'s `build_retrieval_backend`
  factory, from the same `--config` YAML the eval side uses — see
  `core/retrieval_backend.py`'s "On prompt parity with the demo path" note
  above) and shared across every connection/turn in the session, same
  lifetime pattern as `MoshiRAGAdapter.retrieval_backend` across multiple
  `respond()` calls.
  *(Correction to a stale claim in an earlier draft of this spec: this
  section previously described a "for logging" version of this patch,
  capturing the retrieved reference text moshi's own stdout logging omits.
  That was never actually implemented in shipped code — `apply_patches()`
  as committed only wires up `_patch_rag_manager_trigger` and
  `_patch_rag_manager_background_task`, no `get_reference_text` patch at
  all. The logging need described there is real and still applies — moshi
  never logs the retrieved reference text — but it's now satisfied as a
  side effect of the routing patch below rather than a separate logging-only
  one, since our own `RetrievalBackend.retrieve()` call sites already log
  their own request/response.)*
  **Reference-history bookkeeping**, mirroring moshi's own unpatched
  `RAGManager.get_reference_text`: after `backend.retrieve(context,
  history=self._history)` returns, the patch calls `core/retrieval_backend.py`'s
  `format_context()` itself — a second, redundant call with the same
  inputs `retrieve()` already used internally, computing the identical
  result at identical cost since the function is pure and does no I/O —
  purely to recover `num_turns`, then does
  `self._history.append((num_turns, reference_text))` if both are
  non-empty, exactly like moshi's real source does. This is also where a
  genuine `context_injection_s` timing figure comes from — see the
  `retrieval_breakdown_s` bullet below.
- **`session_start`'s `retrieval_backend` field, and the live
  `"instrumentation_session"` push to the browser, are sourced from
  `describe_backend(name, backend_def)` — the same resolved backend
  `main()` built the real `RetrievalBackend` instance from — not from
  `LLM_MODEL_NAME`/`LLM_BASE_URL`.** Fixing a real bug found while
  designing the `get_reference_text` patch above, not a pre-existing
  correct behavior: as shipped, `_patch_server_state` (writes
  `session_start` to `turns.jsonl`, also feeds the console header's
  `retrieval: ...` line) and `_patch_channel` (pushes
  `"kind": "instrumentation_session"` to the browser — the live "which
  backend is answering" indicator a user actually sees mid-session) each
  independently build `{"model": os.environ.get("LLM_MODEL_NAME"),
  "base_url": os.environ.get("LLM_BASE_URL")}`. Those env vars are exactly
  the ones `scripts/run_demo.sh`'s "Config-derived environment" section
  above documents as vestigial — needed only to keep `ServerState.warmup()`
  from raising, no longer connected to what actually answers retrieval
  once `get_reference_text` is patched. Left alone, the session log and
  the live browser UI would both report whatever `warmup()` happened to be
  pointed at rather than the real backend — silently correct only by
  coincidence in the single-backend case (`print_demo_env.py` currently
  derives both from the same resolved entry), and wrong the moment that
  coincidence doesn't hold (a `fallback:` backend, or any future backend
  type without a `model`/`base_url` shaped like Gemini's). `main()` now
  computes `describe_backend(name, backend_def)` exactly **once** and
  threads that single dict through `apply_patches()` to both patch
  functions, rather than each independently reading environment state —
  the duplication (two independently-maintained copies of the same
  lookup) was the root cause, not just the wrong source, so the fix
  removes the duplication too. See "Demo session log" below for the
  corrected `session_start` JSON shape (`{"name", "type", "model"}`, no
  `base_url` for a `gemini_api` backend — that field in the old, buggy
  version was always moshi's own internal OpenAI-shim endpoint, never
  something from `configs/retrieval_backends.yaml` at all).
  `scripts/summarize_demo_session.py`'s console header must tolerate a
  missing `model` key (e.g. a `null_backend` session has none) rather than
  assuming one is always present.
- Reuses the same `<ret>`-token detection already validated for
  `respond()`, applied to the live `Channel`/`RAGManager` objects — but
  `rag_triggered`/`rag_trigger_count`/`retrieval_context`/
  `retrieved_reference_text`/`retrieval_breakdown_s` are **not** computed
  live, in-process, at all. Confirmed against real moshi-rag source: the
  model frequently predicts the RAG token *before* our own VAD-based turn
  boundary has caught up to the user having already finished the next
  question (two real `await` points sit between detecting the token and
  the trigger actually firing, during which a concurrent task can advance
  the boundary first) — so at the moment a trigger or its retrieval result
  would be logged, the live process cannot yet know which turn it truly
  belongs to. Every `<ret>` trigger, retrieval completion (full context/
  reference text/breakdown, not a lump `retrieval_latency_s`), and turn
  boundary is instead logged to `raw_events.jsonl` with whatever turn is
  live at that instant — a deliberately best-effort tag, not a final
  answer — and `scripts/summarize_demo_session.py` computes the correct
  attribution afterward, once the full session (every trigger and every
  boundary) is known. `turns.jsonl`'s own `turn` records therefore carry
  no retrieval fields at all; see "Demo session log" below for the actual
  schema and the reattribution rule
- `ttfat_s` here means the same thing it does in
  `evals/registry/latency/ttfat.py` — time from end-of-user-utterance
  (VAD) to first audio token — since a live full-duplex session has no
  fixed "generation complete" boundary, only a first-token one. This one
  *is* computed live and stored directly on the turn record — it isn't
  subject to the same cross-turn attribution ambiguity as retrieval,
  since it's always about whichever turn is currently pending
- Breaks retrieval timing into the same three stages as
  `evals/registry/latency/retrieval_breakdown.py` — ASR transcription
  wait, Gemini API call, context injection. **`context_injection_s` is
  genuinely meaningful again**, not a stale/dead measurement: it was
  previously captured by patching `LLMReferenceGenerator.process_reference_text`
  (moshi's `Human:`/`moshi:`/`Reference:` transcript-reformatting step),
  which is unreachable now that `get_reference_text` bypasses
  `LLMReferenceGenerator` entirely — `_patch_process_reference_text` is
  dropped from `apply_patches()` rather than kept as a dead no-op — but
  "Context formatting" (`core/retrieval_backend.py`) restores a real
  equivalent: the `get_reference_text` patch's own redundant
  `format_context()` call (see above, done for `num_turns` bookkeeping) is
  timed and reported as `context_injection_s` directly. With every backend's
  `context_formatting` field at its default (`false`/`null` — no drops, no
  history threading), that measured time is honestly close to `0.0`, since
  no real formatting work happens; turning any of them on makes it a real,
  non-fabricated number reflecting actual work done, not a hardcoded
  constant either way. `asr_wait_s` (from `RAGManager._background_task`'s
  own `_wait_event`) and `api_call_s` (`retrieve()`'s own measured Gemini
  latency, minus `context_injection_s`, same computation as before) are
  unaffected and still real. **Decided: `context_injection_s` is always
  displayed as a real number** (console `[aggregate]` table, `raw_events.jsonl`,
  `--json` output alike) — never `—`. It is not the same situation as
  `e2ekd_s` below, which shows `—` because it is genuinely unmeasured
  pending a methodology dependency; `context_injection_s` is always
  actually measured, and a session with every `context_formatting` field
  at its default legitimately has one that's ~`0.0` — that's a true
  measurement of "no formatting work happened," not a placeholder for
  missing data, so it renders the same way any other real number does
- **`e2ekd_s`/`keyword_delay_s` are deferred, not implemented in this
  pass** — every turn record carries them as `null`. `e2ekd.py` doesn't
  exist yet anywhere in this repo (only `knowledge/` evals are
  implemented), and its methodology (Gemini keyword extraction using the
  MoshiRAG paper's Table 17 prompt, then `nvidia/parakeet-tdt-0.6b-v2` for
  the keyword's onset timestamp) must not be approximated — see this
  spec's Context section. The demo mirrors that implementation once it
  exists and is proven in the eval path, rather than prototyping it here
  first. The schema already reserves a `turn_e2ekd_update` record type
  (keyed by `turn_index`) for when this is wired up: computed
  asynchronously as a background task per turn even then, since a
  synchronous extra LLM call plus a full parakeet forward pass on the
  critical path of every live turn is a cost the batch eval path can
  absorb but a live conversation cannot
- Writes two JSONL files per session, both **appended incrementally** (not
  atomic-at-end like eval's `run-*.json` — a demo session has no defined
  completion point and can be killed at any time via
  `tmux kill-session`), to `demo/sessions/<session_id>/`:
  `turns.jsonl` (one record per completed turn — question, response,
  `ttfat_s` only) and `raw_events.jsonl` (one record per raw user/model
  text chunk and RAG event, un-turn-scoped) — see "Demo session log" under
  Output Format for both schemas
- Additionally pushes a live, best-effort version of the same fields over
  the existing client WebSocket message that already carries
  `isRetrieving`/reference text to the browser (additive fields on an
  existing message, not a new message-passing scheme) — see "Web client"
  below. Since the browser only ever needs "most recently known state,"
  not a final, correctly-attributed answer, it doesn't need the same
  post-hoc reattribution `scripts/summarize_demo_session.py` does
- Does not suppress moshi.server's normal stdout logging — the JSONL files
  and the WebSocket fields are all additive, not a replacement

**`scripts/summarize_demo_session.py <session_dir>`** reads both
`turns.jsonl` and `raw_events.jsonl` and prints a console report for
reviewing a session after the fact, mirroring `evals/runner.py`'s existing
conventions (`_print_header`-style
header, `SEP` separator, `⚠` warning prefix): session header, a per-turn
table, and aggregate stats (mean/p95 per retrieval-breakdown stage,
mean/p95 `ttfat_s`, `<ret>` trigger rate) computed on the fly — `e2ekd_s`
is not yet implemented (see "Session instrumentation"), so its column
reads "—" until `evals/registry/latency/e2ekd.py` exists. No precomputed
summary file, matching the existing "no pre-computation in individual run
files" philosophy already used by the eval comparison CLI. A `--json` flag
switches output to one pretty-printed (`indent=2`) JSON document instead of
the console report — see "Demo session log" below for why `turns.jsonl`
itself is deliberately not pretty-printed and needs this flag to get an
eval-style browsable view. This is a CLI tool only — there is no web-based
viewer for historical sessions (see Explicitly Out of Scope).

**On the two, unrelated, `--config` JSON files in this stack** (confirmed
directly against `kyutai-labs/moshi-rag`'s real source, not assumed): both
`moshi.server_conditioner` and `moshi.server` accept a `--config <path>`
flag pointing at the checkpoint's `config.json`. `server_conditioner`
actually reads it — it's the model/conditioner architecture config (`dim`,
`conditioners`, `fuser`), describing how the ARC-Encoder's output fuses
into the LM. On `moshi.server` itself, `--config` is a **dead argument** —
declared via `argparse`, never referenced anywhere else in `server.py`.
Neither of these is where retrieval-backend fallback configuration lives —
that's `MOSHI_RETRIEVAL_LLMS_JSON` (an environment variable, not a file),
covered under `scripts/run_demo.sh` above.

### Web client

Served by `moshi.server` (via `scripts/instrumented_server.py`) from a
static build at `demo/client/` — a fork of moshi-rag's own `client/`
(TypeScript/React/Vite, Apache-2.0/MIT licensed) vendored into this repo
and built via `moshi.server --static <dir>`. This repo owns and maintains
this fork rather than depending on the default downloaded artifact, and
extends it beyond the stock retrieval-visibility UI (`SearchPanel`,
`isRetrieving`, reference text display) to also show, live during a
session: whether `<ret>` fired and how many times, the retrieval backend,
the retrieval breakdown (asr wait / API call / context injection), and
`ttfat_s` — reading the added WebSocket fields described in "Session
instrumentation" above. `e2ekd_s` is deferred (not yet implemented — see
"Session instrumentation") and is not shown in the UI in this pass; once
`evals/registry/latency/e2ekd.py` exists, it would appear a moment after
the rest of the turn's fields, computed asynchronously so it never blocks
the live conversation. The UI shell (title, header,
footer, metadata/favicon, any about/info panels) is rebranded to
unambiguously identify this as an internal eval tool — e.g. "MoshiRAG Eval
Demo" — with moshi-rag's own product branding and disclaimers removed.
Building it requires Node/npm on the VM (see Dependencies).

### Makefile demo target

```makefile
demo:
	$(GCLOUD_SSH) -- "$(REMOTE_INIT) && cd $(REMOTE_DIR) && bash scripts/run_demo.sh"
```

---

## Output Format

### Console output (per run)

```
MoshiRAG Eval Run
checkpoint : base
config     : configs/baseline_with_retrieval.yaml
mode       : sample
git hash   : a3f9c12
timestamp  : 2025-07-13T09:32:11Z
─────────────────────────────────────────────────────────

[knowledge.open_audio_bench]
  TriviaQA     acc: 73.2%   (n=100)
  WebQ         acc: 74.7%   (n=100)
  LlamaQ       acc: 80.3%   (n=100)
  judge model  : gemini-3.5-flash

[knowledge.halu_eval_audio]
  ref acc      : 42.0%   (n=100)
  resp acc     : 36.3%   (n=100)
  judge model  : gemini-3.5-flash

[knowledge.gsm8k]
  acc              : 61.4%   (n=100)
  extractor model  : gemini-3.5-flash
  tts backend      : gemini_tts (gemini-3.1-flash-tts-preview, voice: Kore)

[duplex.full_duplex_bench]
  pause TOR (synthetic)    : 0.32  ↓ lower is better
  pause TOR (candor)       : 0.56  ↓
  backchannel freq (/s)    : 0.010 ↑ higher is better
  backchannel JSD          : 0.94  ↓
  turn taking TOR          : 0.83  ↑
  turn taking latency (s)  : 0.18  ↓
  interruption GPT score   : 3.75  ↑
  interruption latency (s) : 1.02  ↓

[latency.ttfat]
  mean : 0.04s   p50 : 0.03s   p95 : 0.07s
  (n=200)

[latency.e2ekd]
  mean TTFAT         : 0.04s
  mean keyword delay : 3.07s
  mean E2EKD         : 3.11s
  p95  E2EKD         : 4.20s
  (n=200)
  ⚠ spot_check_completed is false — run with --spot-check before trusting this metric

[latency.retrieval_breakdown]
  asr transcription wait  mean: 0.51s   p95: 0.63s
  gemini api call         mean: 0.87s   p95: 1.24s
  context injection       mean: 0.03s   p95: 0.05s
  total retrieval delay   mean: 1.41s   p95: 1.82s
  ⚠ p95 total (1.82s) exceeds latency gate (1500ms) — scores may be quietly degraded

─────────────────────────────────────────────────────────
results written to: evals/results/run-2025-07-13T09-32-11Z.json
```

### Per-run JSON schema

```json
{
  "run_id": "2025-07-13T09-32-11Z",
  "checkpoint": "base",
  "checkpoint_uri": "gs://your-project/checkpoints/base/moshirag-base-bf16",
  "config": "configs/baseline_with_retrieval.yaml",
  "mode": "sample",
  "git_hash": "a3f9c12",
  "timestamp": "2025-07-13T09:32:11Z",
  "completed": true,
  "evals": {
    "knowledge.open_audio_bench": {
      "scores": {
        "triviaqa_acc": 0.732,
        "webq_acc": 0.747,
        "llamaq_acc": 0.803
      },
      "metadata": {
        "n_triviaqa": 100,
        "n_webq": 100,
        "n_llamaq": 100,
        "judge_model": "gemini-3.5-flash"
      },
      "completed": true,
      "last_completed_index": 300,
      "errors": []
    },
    "knowledge.halu_eval_audio": {
      "scores": {
        "ref_acc": 0.420,
        "resp_acc": 0.363
      },
      "metadata": {
        "n": 100,
        "judge_model": "gemini-3.5-flash"
      },
      "completed": true,
      "last_completed_index": 100,
      "errors": []
    },
    "knowledge.gsm8k": {
      "scores": {
        "gsm8k_acc": 0.614
      },
      "metadata": {
        "n": 100,
        "judge_model": "gemini-3.5-flash",
        "tts_backend": {"name": "gemini_tts", "type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore"}
      },
      "completed": true,
      "last_completed_index": 100,
      "errors": []
    },
    "duplex.full_duplex_bench": {
      "scores": {
        "pause_tor_synthetic": 0.32,
        "pause_tor_candor": 0.56,
        "backchannel_freq_per_sec": 0.010,
        "backchannel_jsd": 0.94,
        "turn_taking_tor": 0.83,
        "turn_taking_latency_s": 0.18,
        "interruption_gpt_score": 3.75,
        "interruption_latency_s": 1.02
      },
      "metadata": {
        "version": "v1",
        "n_scenarios": 30
      },
      "completed": true,
      "last_completed_index": 30,
      "errors": []
    },
    "latency.ttfat": {
      "scores": {
        "mean_s": 0.04,
        "p50_s": 0.03,
        "p95_s": 0.07
      },
      "metadata": { "n": 200 },
      "completed": true,
      "last_completed_index": 200,
      "errors": []
    },
    "latency.e2ekd": {
      "scores": {
        "mean_ttfat_s": 0.04,
        "mean_keyword_delay_s": 3.07,
        "mean_e2ekd_s": 3.11,
        "p95_e2ekd_s": 4.20
      },
      "metadata": {
        "n": 200,
        "keyword_extractor": "gemini-3.5-flash",
        "timestamp_model": "nvidia/parakeet-tdt-0.6b-v2",
        "spot_check_completed": false
      },
      "completed": true,
      "last_completed_index": 200,
      "errors": []
    },
    "latency.retrieval_breakdown": {
      "scores": {
        "asr_wait_mean_s": 0.51,
        "asr_wait_p95_s": 0.63,
        "api_call_mean_s": 0.87,
        "api_call_p95_s": 1.24,
        "context_injection_mean_s": 0.03,
        "context_injection_p95_s": 0.05,
        "total_mean_s": 1.41,
        "total_p95_s": 1.82
      },
      "metadata": {
        "latency_gate_ms": 1500,
        "gate_breached_p95": true
      },
      "completed": true,
      "last_completed_index": 200,
      "errors": [
        "p95 total retrieval latency (1.82s) exceeds gate (1500ms) — scores may be quietly degraded"
      ]
    }
  }
}
```

### Comparison CLI output

```
python runner.py --compare \
  evals/results/run-2025-07-13T09-32-11Z.json \
  evals/results/run-2025-07-20T14-11-03Z.json

comparing:
  A  base      (2025-07-13)  configs/baseline_with_retrieval.yaml  [sample]
  B  lora-v1   (2025-07-20)  configs/baseline_with_retrieval.yaml  [sample]
─────────────────────────────────────────────────────────

knowledge.open_audio_bench
  triviaqa_acc      A: 73.2%   B: 74.8%   Δ +1.6pp  ✓
  webq_acc          A: 74.7%   B: 73.1%   Δ -1.6pp  ↓
  llamaq_acc        A: 80.3%   B: 81.0%   Δ +0.7pp  ✓

knowledge.halu_eval_audio
  ref_acc           A: 42.0%   B: 43.1%   Δ +1.1pp  ✓
  resp_acc          A: 36.3%   B: 37.9%   Δ +1.6pp  ✓
  rag_lift_pp       (computed) A→B: +1.6pp over no-retrieval baseline

knowledge.gsm8k
  gsm8k_acc         A: 61.4%   B: 63.0%   Δ +1.6pp  ✓

duplex.full_duplex_bench
  pause_tor_synth   A: 0.32    B: 0.35    Δ +0.03   ↓
  turn_taking_tor   A: 0.83    B: 0.81    Δ -0.02   ↓
  interruption_gpt  A: 3.75    B: 3.70    Δ -0.05   ↓

latency.ttfat
  mean_s            A: 0.04    B: 0.04    Δ  0.00   ✓

latency.e2ekd
  mean_e2ekd_s      A: 3.11    B: 3.08    Δ -0.03   ✓

latency.retrieval_breakdown
  total_mean_s      A: 1.41    B: 1.39    Δ -0.02   ✓
  total_p95_s       A: 1.82    B: 1.77    Δ -0.05   ⚠ still above gate in both runs

─────────────────────────────────────────────────────────
summary
  knowledge  : B marginally better overall
  duplex     : small regressions, within noise — monitor
  latency    : stable
  gate       : retrieval p95 exceeds 1500ms in both runs — review infrastructure
```

### Demo session log

Written incrementally to two files under `demo/sessions/<session_id>/` by
`scripts/instrumented_server.py` — both appended as the session progresses,
not written atomically at the end, since a live session has no defined
completion point.

**`turns.jsonl`** — one record per completed turn. First line is a
`session_start` record; one `turn` record follows per turn. Deliberately
carries **no retrieval fields at all** — see the note below for why.

```json
{"type": "session_start", "session_id": "2026-07-13T09-32-11Z", "checkpoint": "base", "git_hash": "a3f9c12", "timestamp": "2026-07-13T09:32:11Z", "retrieval_backend": {"name": "gemini_api", "type": "gemini_api", "model": "gemini-3.5-flash"}, "rag_timeout_s": 8, "stt_mode": "local"}
{"type": "turn", "turn_index": 1, "timestamp": "2026-07-13T09:32:45Z", "user_question_text": "What's the capital of France?", "model_response_text": "Paris.", "ttfat_s": 0.05, "e2ekd_s": null, "keyword_delay_s": null}
{"type": "turn", "turn_index": 2, "timestamp": "2026-07-13T09:33:20Z", "user_question_text": "What did the Q3 earnings report say about revenue growth?", "model_response_text": "Revenue grew twelve percent year over year.", "ttfat_s": 1.42, "e2ekd_s": null, "keyword_delay_s": null}
```

`retrieval_backend`'s shape is `describe_backend()`'s output (see
`core/retrieval_backend.py` and "Session instrumentation" above) —
`{"name", "type", "model"?, "base_url"?}`, the latter two only present if
the resolved backend definition has them. A `null_backend` session (no
retrieval) carries `{"name": "null", "type": "null_backend"}`, no `model`
key — consumers must not assume one is always present.

**`raw_events.jsonl`** — one record per raw user/model text chunk and RAG
event, un-turn-scoped. This is the *only* source of retrieval data. Each
event's `turn_index` is a live, best-effort tag (whatever turn is current
in `instrumented_server.py` at write time) — confirmed against real
moshi-rag source, the model frequently predicts the `<ret>` token *before*
the demo's own VAD-based turn boundary has caught up to the user having
already finished the next question, so this tag cannot be trusted as a
final answer at write time.

```json
{"type": "raw_event", "event": "ret_triggered", "timestamp": "2026-07-13T09:33:18Z", "t_rel_s": 67.104, "turn_index": 1}
{"type": "raw_event", "event": "utterance_end", "timestamp": "2026-07-13T09:33:19Z", "t_rel_s": 68.330, "turn_index": 2}
{"type": "raw_event", "event": "retrieval_complete", "timestamp": "2026-07-13T09:33:20Z", "t_rel_s": 69.790, "turn_index": 1, "retrieval_context": "User asked about Q3 earnings report revenue growth...", "retrieved_reference_text": "According to the Q3 report, revenue grew 12% year over year...", "retrieval_breakdown_s": {"asr_wait_s": 0.12, "api_call_s": 0.71, "context_injection_s": 0.02, "total_s": 0.85}}
```

**Why retrieval data lives only in `raw_events.jsonl`, reconstructed by
`scripts/summarize_demo_session.py`'s `compute_retrievals_from_raw_events()`,
rather than being written directly into each `turn` record:** an
already-appended JSONL line can't be retroactively edited, and the live
process genuinely cannot know at write time whether a `<ret>` trigger it
just tagged with turn N actually belongs to turn N or N+1 — only after the
full session is over, with every trigger and turn boundary known, can that
be decided correctly. The rule: for each turn_index a `<ret>` trigger was
tagged with, all but the *last* trigger sharing that tag stay exactly
where they're tagged (same-turn re-triggers — e.g. the model correcting
itself mid-monologue, no new user input); the last one moves forward to
turn_index + 1 if that next turn's boundary actually happened in the
session. A turn can end up with more than one retrieval this way (0, 1, or
more), not a single slot — see "Demo session summary (console)" below for
how that's rendered.

`e2ekd_s`/`keyword_delay_s` are always `null` in this pass — deferred
until `evals/registry/latency/e2ekd.py` exists (see "Session
instrumentation"). The `turn_e2ekd_update` record type (keyed by
`turn_index`, carrying `keyword`/`keyword_delay_s`/`e2ekd_s`, written to
`turns.jsonl` alongside `turn` records) is reserved schema for that future
wiring, not emitted today — readers should join on `turn_index`, not
assume a single record per turn.

**This file is intentionally compact JSONL, not pretty-printed, unlike
`run-*.json`.** The eval path's JSON is pretty-printed
(`json.dump(..., indent=2)`) because it's one document written once,
atomically, at the end. `turns.jsonl` has to stay one compact JSON value
per line because that's what makes it safely appendable mid-session — a
session killed mid-turn (`tmux kill-session`, disconnect, crash) shouldn't
corrupt or require rewriting the whole file. Practical consequence:
browsers auto-pretty-print a `.json` file because it's a single JSON
document, but won't do that for `.jsonl` (multiple documents) — opened
directly, it reads as a wall of minified lines. `scripts/
summarize_demo_session.py --json <session_dir>` closes that gap on
demand: it reads `turns.jsonl`, joins each turn with its
`turn_e2ekd_update` by `turn_index`, computes its retrievals from
`raw_events.jsonl` (see above), and prints one `indent=2` JSON document —
same convention as `run-*.json`, for whoever wants a browsable artifact
instead of (or alongside) the console summary. Not precomputed/stored,
matching the "no pre-computation in individual run files" philosophy
already used elsewhere in this doc.

### Demo session summary (console)

Printed by `scripts/summarize_demo_session.py demo/sessions/<session_id>/`,
mirroring the eval console conventions above. `[turns]`'s `<ret>` column
shows `yes ×N` and `ret.total` lists every value comma-separated (not
summed/averaged) when a turn ends up with more than one retrieval (see
"Demo session log" above); `[aggregate]`'s retrieval breakdown counts
every individual retrieval, not one per turn. `[full transcript]` numbers
each retrieval block `retrieval i/N (...)` when a turn has more than one.
`[raw stream]` shows the same events un-reattributed — its `turnN` tags
are the raw, live-tagged values, deliberately not corrected, since its
whole purpose is showing what literally happened:

```
MoshiRAG Demo Session
session id : 2026-07-13T09-32-11Z
checkpoint : base
git hash   : a3f9c12
retrieval  : gemini-3.5-flash (rag_timeout: 8s)
─────────────────────────────────────────────────────────

[turns]
  #   time      question                                    <ret>  ret.total  ttfat   e2ekd
  1   09:32:45  What's the capital of France?               no     —          0.05s   —
  2   09:33:20  What did the Q3 earnings report say abo...  yes    0.85s      1.42s   —

─────────────────────────────────────────────────────────
[aggregate]
  turns              : 2
  <ret> trigger rate : 50.0%   (1/2)
  retrieval breakdown  asr_wait  mean 0.12s   p95 0.12s   (n=1)
                       api_call  mean 0.71s   p95 0.71s   (n=1)
                       ctx_inj   mean 0.02s   p95 0.02s   (n=1)
                       total     mean 0.85s   p95 0.85s   (n=1)
  ttfat                          mean 0.73s   p95 1.35s   (n=2)
  ⚠ e2ekd not yet implemented — evals/registry/latency/e2ekd.py doesn't
    exist yet; this column stays "—" until that eval lands and the demo
    mirrors it

─────────────────────────────────────────────────────────

[full transcript]
  #1 user : What's the capital of France?
  #1 moshi: Paris.

  #2 user : What did the Q3 earnings report say about revenue growth?
  #2 moshi: Revenue grew twelve percent year over year.

        retrieval (what the model asked with):
          user: What did the Q3 earnings report say about revenue growth?
        retrieval (what came back):
          According to the Q3 report, revenue grew 12% year over year.

─────────────────────────────────────────────────────────

[raw stream]
  Chronological, un-turn-scoped view of exactly what streamed in from the
  user and out from the model — use this, not the turn table above, to see
  <ret> firing too early, too late, or unprompted: turn boundaries here are
  just one more timestamped event, not an assumed structure on the data.

  +  33.201s  turn0   USER   What's the capital of France?
  +  34.010s  turn1   · · · turn boundary (utterance end / VAD) · · ·
  +  34.060s  turn1   first audio (ttfat 0.05s)
  +  34.100s  turn1   MOSHI  Paris.
  +  67.001s  turn1   USER   What did the Q3 earnings report say about revenue growth?
  +  67.104s  turn1   <ret> triggered
  +  68.330s  turn2   · · · turn boundary (utterance end / VAD) · · ·
  +  68.380s  turn2   first audio (ttfat 1.42s)
  +  69.790s  turn1   retrieval complete (0.85s)
  +  69.850s  turn2   MOSHI  Revenue grew twelve percent year over year.

raw logs: demo/sessions/2026-07-13T09-32-11Z/{server,conditioner}.log
```

Note the raw stream tags the `<ret>` trigger and its retrieval completion
`turn1` throughout (the live, best-effort tag at write time), while
`[turns]`/`[full transcript]` correctly show it grounding turn 2 instead —
this is `compute_retrievals_from_raw_events()`'s reattribution at work,
not an inconsistency between the two sections.

---

## Configs

### `configs/baseline_no_retrieval.yaml`
```yaml
model:
  checkpoint: base
  generation:
    cfg_coef: 1.0
    stt_wait_time: 0.5
    rag_timeout: 8.0
    max_reference_tokens: 64
    vad_window_size: 4
    vad_threshold: 0.5
    power_threshold: -65
    tail_silence_steps: 25
  retrieval:
    enabled: false

judge:
  model: gemini-3.5-flash

tts:
  backend: gemini_tts          # key into configs/tts_backends.yaml — used by knowledge.gsm8k

evals:
  - knowledge.open_audio_bench
  - knowledge.halu_eval_audio
  - knowledge.gsm8k
  - duplex.full_duplex_bench
  - latency.ttfat
  - latency.retrieval_breakdown

output_dir: ./evals/results/
```

### `configs/baseline_with_retrieval.yaml`
```yaml
model:
  checkpoint: base
  generation:
    cfg_coef: 1.0
    stt_wait_time: 0.5
    rag_timeout: 8.0
    max_reference_tokens: 64
    vad_window_size: 4
    vad_threshold: 0.5
    power_threshold: -65
    tail_silence_steps: 25
  retrieval:
    enabled: true
    backend: gemini_api          # key into configs/retrieval_backends.yaml
    latency_gate_ms: 1500
    fallback: null                # scaffold slot — a second backend name to
                                   # race/fall back to; unimplemented beyond
                                   # this config slot existing, see
                                   # core/retrieval_backend.py's registry note

judge:
  model: gemini-3.5-flash

tts:
  backend: gemini_tts          # key into configs/tts_backends.yaml — used by knowledge.gsm8k

evals:
  - knowledge.open_audio_bench
  - knowledge.halu_eval_audio
  - knowledge.gsm8k
  - duplex.full_duplex_bench
  - latency.ttfat
  - latency.retrieval_breakdown

output_dir: ./evals/results/
```

`model.retrieval.model` (the Gemini model name) has moved out of the
per-run config and into `configs/retrieval_backends.yaml`'s `gemini_api`
entry below — it's a property of the backend, not something that varies
per eval run the way `latency_gate_ms` does.

### `configs/retrieval_backends.yaml`

New file (committed, non-sensitive — `${VAR}` interpolation available same
as `checkpoints.yaml`). Named backend definitions, looked up by
`model.retrieval.backend`/`.fallback` in the eval configs above and
resolved by `core/config.py`. `type` is the `BACKEND_REGISTRY` key
`core/retrieval_backend.py`'s factory dispatches on:

```yaml
backends:
  gemini_api:
    type: gemini_api
    model: gemini-3.5-flash
    api_key_env: GEMINI_API_KEY
    prompt_template: configs/prompts/retrieval_reference_simple.txt
    context_formatting:
      drop_leading_incomplete_turn: false
      drop_trailing_incomplete_turn: false
      thread_reference_history: false
      history_max_entries: null
      role_labels: {user: "Human", model: "moshi"}

  # Alternate starting point, not the shipped default — demonstrates the
  # full moshi-defaults pairing described in core/retrieval_backend.py's
  # "On prompt parity" note. Select it explicitly
  # (model.retrieval.backend: gemini_api_moshi_style) rather than editing
  # the entry above in place, so the existing baseline configs' behavior
  # doesn't change silently underneath anyone already relying on it.
  # gemini_api_moshi_style:
  #   type: gemini_api
  #   model: gemini-3.5-flash
  #   api_key_env: GEMINI_API_KEY
  #   prompt_template: configs/prompts/retrieval_reference_moshi_style.txt
  #   context_formatting:
  #     drop_leading_incomplete_turn: true
  #     drop_trailing_incomplete_turn: true
  #     thread_reference_history: true
  #     history_max_entries: null   # true parity with moshi's own real
  #                                 # (unbounded) behavior — set a number
  #                                 # instead if long-session growth becomes
  #                                 # a real problem; moshi itself doesn't
  #                                 # cap this either
  #     role_labels: {user: "Human", model: "moshi"}

  null:
    type: null_backend

  # Extension point, not implemented yet — no closed/local retrieval
  # backend exists to build or test against today. A future entry would
  # look something like this and become selectable purely via config, with
  # a new RetrievalBackend subclass registered in BACKEND_REGISTRY:
  #
  # local_llm:
  #   type: openai_compatible
  #   base_url: http://localhost:8080/v1
  #   model: llama-3-8b-instruct
  #   api_key_env: LOCAL_LLM_API_KEY
  #   prompt_template: configs/prompts/retrieval_reference_simple.txt
```

`context_formatting` is optional per backend entry — a backend definition
that omits it entirely gets `format_context()`'s own all-off defaults
(identical to the `gemini_api` entry above written out explicitly). See
`core/retrieval_backend.py`'s "Context formatting" section for what each
field does.

### `configs/tts_backends.yaml`

New file (committed, non-sensitive — same `${VAR}` interpolation and
alias-file pattern as `retrieval_backends.yaml`). Named backend
definitions, looked up by `tts.backend` in an eval config and resolved by
`core/config.py`. `type` is the `TTS_REGISTRY` key `core/tts.py`'s factory
dispatches on:

```yaml
backends:
  gemini_tts:
    type: gemini_tts
    model: gemini-3.1-flash-tts-preview
    voice: Kore
    api_key_env: GEMINI_API_KEY

  # Alternate starting point — Gemini's 2.5-generation TTS-capable models,
  # if the 3.1 preview model above is ever unavailable or unstable. Select
  # explicitly (tts.backend: gemini_tts_25) rather than editing the entry
  # above in place.
  # gemini_tts_25:
  #   type: gemini_tts
  #   model: gemini-2.5-flash-preview-tts
  #   voice: Kore
  #   api_key_env: GEMINI_API_KEY

  # Extension point, not implemented yet — no local TTS engine has been
  # evaluated for this project. A future entry would look something like
  # this and become selectable purely via config, with a new TTSBackend
  # subclass registered in TTS_REGISTRY:
  #
  # kokoro:
  #   type: kokoro
  #   voice: af_heart
```

### `configs/prompts/retrieval_reference_simple.txt`

The `GeminiAPIBackend` prompt template, extracted verbatim from what was
previously `core/retrieval_backend.py`'s `_RETRIEVE_PROMPT` string literal.
Plain text with a `{context}` placeholder — not moshi-rag's own
`Human:`/`moshi:`/`Reference:` transcript-reformatting format. Pairs with
every `context_formatting` field at its default (`false`/`null`) — the
shipped default for both `baseline_*.yaml` configs.

### `configs/prompts/retrieval_reference_moshi_style.txt`

A completion-style prompt modeled on moshi-rag's own bundled
`reference_prompt_template.txt`: ends in a bare trailing `Reference:` with
nothing after it (so the LLM continues the pattern rather than answering
an instruction), includes moshi's own guidelines (concise, factual, no
markdown, roughly 50 words), and expects `{context}` to already be a
`Human:`/`moshi:`/`Reference:`-labeled transcript — i.e. it only produces
sensible output when paired with `drop_leading_incomplete_turn`,
`drop_trailing_incomplete_turn`, and `thread_reference_history` all
enabled (see the commented `gemini_api_moshi_style` entry above). Not
selected by default; a concrete, working starting point for whoever
experiments with this next, and the naming convention
(`retrieval_reference_<style>.txt`) new prompt files as experimentation
proceeds are expected to follow.

Both files now serve **both** the demo and evals for whichever backend
references them (see `core/retrieval_backend.py`'s "On prompt parity with
the demo path" note for how) — editing one changes retrieval behavior
everywhere that backend is used at once; there is no separate demo-side
prompt to keep in sync.

### `configs/checkpoints.yaml`
```yaml
aliases:
  base: gs://${GCS_BUCKET}/checkpoints/base/moshirag-base-bf16
```

---

## Dependencies

Managed via `pyproject.toml` and `uv.lock`. A single lockfile is
sufficient — all execution targets are Linux/CUDA (VM). There is no macOS
execution target. This list is illustrative, not authoritative — see
`pyproject.toml` for actual current dependencies.

`sounddevice` is not a dependency, and neither is any web-server framework
(`fastapi`/`uvicorn`/`websockets`) — the demo is now real `moshi.server`,
which brings its own web stack; nothing here needs to.

Building `demo/client/` (the forked, rebranded web client — see "Demo")
requires Node/npm on the VM. This is the one non-Python, non-`uv` build
tool this project depends on; it is a build-time dependency only,
`moshi.server` serves the built static output at runtime and needs nothing
further from Node.

`core/tts.py`'s `GeminiTTSBackend` (see "Core Abstractions") needs no new
dependency — `google-genai` is already a base dependency for the LLM judge
and retrieval backend. `./tts_cache/` (generated, gitignored, excluded from
`make sync` — see Makefile) holds synthesized question audio, same
treatment as `./checkpoint_cache/`.

Local dev (Ubuntu VM): `uv venv && uv sync`
Vertex AI VM: `make sync && make install` (install only needed when dependencies change)

---

## Infrastructure

- GCP Vertex AI, single `a2-ultragpu-1g` VM (1 x A100 80GB) — corrected
  from an earlier `a2-highgpu-1g` in this section, which names the
  40GB-per-GPU tier; the VM has always been the 80GB variant, confirmed
  via GCP's own machine-type table once combined front-end+conditioner
  memory usage (~66-71GB, see "GPU Sizing and Multi-GPU Deployment" below)
  was actually measured and wouldn't have fit on a 40GB card at all
- MoshiRAG 7B + 1B streaming ASR run in-process on the A100; the ARC-Encoder always runs as a separate `server_conditioner` process — for both the demo and evals, not just one or the other
- All LLM backend and judge calls go to Gemini API — no local LLM
- Checkpoints stored in GCS, referenced by URI or named alias
- Local dev uses uv directly on Ubuntu VM: `uv venv && uv sync`
- Production VM environment managed via `make sync` and `make install` — no Docker
- `.venv` excluded from rsync, built on VM to match Linux and CUDA environment
- Demo accessed from macOS (or any browser) via SSH tunnel to VM port 8998 — no software installation required on the client machine

---

## GPU Sizing and Multi-GPU Deployment

**Correction (2026-07-26): the "GPU contention" root cause below, and the
MIG/dual-GPU sizing work that followed from it, is superseded.** Testing on
real dual-GPU hardware (`wb-gpu-a1ultra2g`, since stopped) showed the
conditioning latency (~1.5-1.9s) persists essentially unchanged even with
the front-end and conditioner on genuinely separate physical GPUs — proving
GPU compute sharing was never the actual mechanism. The real cause is
upstream moshi-rag's own `ServerState._step_loop` running each generation
step as a synchronous, unyielding call inside an `async def` function,
starving its own concurrent conditioning HTTP call of event-loop time —
see CLAUDE.md's "SUPERSEDED: GPU contention conclusion was wrong" section
for the full evidence chain (GPU utilization traces, matched timer values,
real moshi-rag source). **GPU count is not the relevant variable for this
latency** — `wb-gpu-a1ultra` (single A100) is sufficient for all further
work. The sections below are kept largely as originally written (not
deleted or silently rewritten, per this repo's convention for correcting a
wrong conclusion) since the reasoning was sound given single-GPU-only
evidence at the time — only disentangled once a genuinely separate GPU
existed to test against. Where a subsection's conclusion no longer holds,
that's called out inline rather than removed.

Original root cause claim (superseded, see above): running the front-end
model and `server_conditioner`'s ARC-Encoder on the same physical GPU means
the front-end's own 66-76% utilization during generation starves the
conditioner's `/embed` calls — measured `context_injection_s` mean 1.58s /
p95 1.68s under real contention vs. ~42ms solo, a ~37x slowdown from the
same process and code. This exceeds the MoshiRAG paper's own stated budget
(§3.1: "the entire retrieval process completes within two seconds," with "a
sharp decline in accuracy when retrieval latency exceeds 1.5 seconds"
observed empirically) and has produced a confirmed hallucination-feedback
loop in a live session, not just a theoretical risk. This was believed
ruled out via solo-vs-contended `nvidia-smi` polling on the single-GPU box
at the time — but that comparison couldn't actually distinguish "conditioner
compute contending for the GPU" from "front-end's own step loop starving
its own I/O," since both look identical when there's only one GPU in the
picture. The dual-GPU test is what actually disentangled them.

### MIG: ruled out (moot now, not just infeasible)

The largest MIG sub-partition available on an 80GB A100 short of the
whole card (`7g.80gb`, which isn't a partition at all) is 40GB (`4g.40gb`
+ `3g.40gb`, the standard two-way split). The front-end's own measured
peak memory footprint is **47.8GB** — from a 25-question *smoke* run,
itself a lower bound since nothing in this codebase calls
`torch.cuda.empty_cache()` between questions and a real sample/full run
would plausibly push it higher still. That already exceeds the largest
partition MIG can offer without allocating the entire GPU, which defeats
the point of partitioning. MIG does not fit this workload on capacity
grounds alone — and separately, now moot regardless: the correction above
means no GPU-partitioning scheme was ever going to fix this latency, since
the bottleneck isn't GPU compute sharing at all.

### Dual-GPU (`a2-ultragpu-2g`): tried, confirmed not the fix

Provisioned as `wb-gpu-a1ultra2g` (2x A100 80GB, same project/zone as
`wb-gpu-a1ultra` — see CLAUDE.md's "Remote instances"), specifically to test
whether genuine physical GPU separation resolved the conditioning latency.
It did not — see the correction at the top of this section and CLAUDE.md's
"SUPERSEDED: GPU contention conclusion was wrong" for the full evidence.
`nvidia-smi` confirmed the device-pinning mechanism below worked exactly as
designed (front-end on GPU 0, conditioner alone on GPU 1, correct memory
footprints on each) — the mechanism isn't broken, it just doesn't address
the actual bottleneck. Stopped (not deleted) once this was confirmed;
resuming single-GPU-only work on `wb-gpu-a1ultra` for the underlying
step-loop fix, which needs no GPU-count-specific infrastructure at all.

### Device assignment: auto-detected, same codebase on both instances

`make sync`'s rsync, and every file it carries, stay identical regardless
of which VM they land on — no hostname or machine-type branching
anywhere. Device assignment is a pure function of GPU count, resolved
fresh at each process launch:

- **Front-end**: always `cuda:0`
- **Conditioner**: `cuda:1` if `torch.cuda.device_count() >= 2`, else
  `cuda:0` (shared with the front-end — see the warning gate below)

This fixed, role-based rule (not negotiation between processes) is
required because the eval path's two processes are launched
independently, from two separate terminals (see "Required setup:
`server_conditioner` process" in CLAUDE.md) with no shared parent to
coordinate a split at launch time — each has to independently arrive at
the same answer.

- **Single source of truth**: `core/gpu.py`'s `resolve_devices()`
  (Python, via `torch.cuda.device_count()`) is the one place this logic
  lives. `scripts/run_demo.sh` shells out to it (same pattern as its
  existing `resolve_checkpoint`/`print_demo_env.py` calls) rather than
  reimplementing device-count detection separately in bash — avoids the
  two ever disagreeing.
- **Explicit override always wins**: auto-detection only sets
  `CUDA_VISIBLE_DEVICES` when the launching process's environment doesn't
  already have it set. This preserves `scripts/gpu_diag_solo.sh`/
  `scripts/gpu_diag_contended.sh`'s existing pattern of deliberately
  forcing both processes onto the same GPU for a controlled comparison,
  including on the dual-GPU box if that comparison is ever rerun there.
- **Replaces two existing hardcodes**: `core/model_interface.py`'s
  `device="cuda:0"` in `MoshiRAGAdapter._load_models()`, and
  `scripts/run_demo.sh`'s `--cuda-device 0` (conditioner) / `--device
  cuda` (server) flags.

### Warning gate: contended runs are tagged, not silently trusted

`resolve_devices()` returns `contended: bool` — true whenever the
conditioner ends up sharing device 0 with the front-end (the single-GPU
case). Whenever `contended` is true:

- Logged at startup on both the demo and eval paths (`⚠ single-GPU mode:
  known conditioner contention, do not trust retrieval-latency or
  grounding-dependent scores from this run`)
- Stamped into the eval result JSON's metadata as
  `conditioner_contended: true`, alongside the existing
  checkpoint/timestamp/git-hash/config/mode tags — so `--compare` between
  two runs surfaces a contended run immediately, without the operator
  needing to remember which VM produced which file. Same spirit as the
  existing `spot_check_completed` gate.

### Current guidance: `wb-gpu-a1ultra` (single GPU) is sufficient for all work

Superseding the "Fallback scope" framing this section originally had (which
assumed retrieval-enabled `sample`/`full` runs and the demo required
dual-GPU): since GPU count was never the relevant variable, single-GPU is
not a restricted fallback anymore — it's simply the instance, for
`tiny`/`smoke`/no-retrieval work and retrieval-enabled `sample`/`full`/demo
work alike, until the real step-loop fix (see CLAUDE.md's "Not yet decided"
list under the superseded-conclusion section) lands. The
`conditioner_contended` JSON tag and warning gate (below) still fire
correctly on single-GPU runs — they're accurate about *contention*, just no
longer the whole explanation for the latency those runs will still show.
Leave that tag and warning in place (harmless, still meaningful for
diagnosing genuine GPU-sharing scenarios in the future) without treating an
absence of contention as proof a run's conditioning latency is fine — that
now depends on whether the step-loop fix has landed, not on GPU topology.

---

## Makefile

```makefile
REMOTE_USER ?= user
REMOTE_HOST ?= vertex-vm-ip
REMOTE_DIR  ?= /app/moshirag-evals

sync:
	rsync -avz \
	  --exclude '.venv' \
	  --exclude '__pycache__' \
	  --exclude 'checkpoint_cache' \
	  --exclude 'tts_cache' \
	  --exclude 'evals/results' \
	  --exclude '*.pyc' \
	  . $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/

install:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) \
	  "cd $(REMOTE_DIR) && uv sync --frozen"

ssh:
	ssh $(REMOTE_USER)@$(REMOTE_HOST)

demo:
	$(GCLOUD_SSH) -- "$(REMOTE_INIT) && cd $(REMOTE_DIR) && bash scripts/run_demo.sh"
```

`.venv` is excluded from rsync — compiled dependencies (torch, SpeechBrain) are platform and CUDA-specific and must be built on the VM via `make install`. Run `make install` only after changes to `pyproject.toml` or `uv.lock`. For code-only changes `make sync` is sufficient.

---

## Explicitly Out of Scope

- Vanilla Moshi comparison
- Faithfulness eval
- Abstention rate eval
- Human MOS evaluation
- Any training or fine-tuning code — lives in `moshirag-finetune` (separate repo)
- Production serving for real end users — this repo's use of `moshi.server`/`moshi.server_conditioner` is for evals and an internal demo tool, not a production deployment; real production deployment of validated fine-tuned weights is a separate concern, still out of scope here
- Multi-user demo access — the demo server is single-session by design
- TLS / HTTPS for the demo — SSH tunnel provides the secure context; certificates are not needed
- Web-based viewer for *historical* demo sessions — review is CLI-only via
  `scripts/summarize_demo_session.py`; the client fork (`demo/client/`)
  covers *live* in-session visibility only, not browsing past sessions
- Instrumenting `server_conditioner` beyond raw log persistence — none of
  the six tracked fields (question, `<ret>`, retrieval context/reference
  text, backend, retrieval/generation latency) originate there; it stays
  fully unmodified

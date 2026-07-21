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

---

## Repository Structure

```
moshirag-evals/
  core/
    model_interface.py
    retrieval_backend.py
    checkpoint.py
    config.py
  evals/
    runner.py
    registry/
      knowledge/
        open_audio_bench.py
        halu_eval_audio.py
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
  configs/
    baseline_no_retrieval.yaml
    baseline_with_retrieval.yaml
    checkpoints.yaml
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
  - Requires a real, separately launched `server_conditioner` process — conditioning is never in-process. Raises at construction if `REFERENCE_ENCODER_URL` is unset
  - Exposes timing metadata in the returned `metadata` dict including first audio token timestamp
  - **Important:** MoshiRAG runs two parallel token streams simultaneously — an inner monologue text channel and an audio output channel. The adapter must capture both. Read the MoshiRAG repo `run_inference.py` before implementing this class — that script is the ground-truth inference path the paper used, and the adapter must match it exactly before adding instrumentation on top
  - `respond(audio_in: bytes) -> tuple[bytes, str, dict]` is batch mode only (complete audio bytes in, complete audio bytes out) — used by evals. There is no streaming variant: the demo is served entirely by moshi.server directly, not through this adapter
- No eval logic lives here — this is purely model I/O

### `core/retrieval_backend.py`

- Abstract base class `RetrievalBackend` with method:
  `retrieve(context: str) -> tuple[str, float]` returning `(reference_text, latency_seconds)`
- Concrete class `GeminiAPIBackend(RetrievalBackend)` that:
  - Calls Gemini API with conversation context
  - Records and returns latency of every call
  - Raises `LatencyGateError` if latency exceeds configurable `latency_gate_ms`
  - Logs every call duration regardless of whether gate is breached
  - Retries failed calls up to 3 times with exponential backoff before raising
- `NullBackend(RetrievalBackend)` that returns empty string immediately, used for Config A (no retrieval)

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
- Runs each eval, collects `EvalResult` objects
- On Gemini API failure: retries up to 3 times with exponential backoff, then writes partial results to JSON and exits gracefully with a clear error message indicating where it stopped
- On restart: detects existing partial result file for the same checkpoint and config, resumes from last completed question rather than starting over
- Writes one JSON output file per run to `evals/results/` tagged with checkpoint, timestamp, git hash, config, and mode used
- Comparison CLI computes deltas and derived metrics including `rag_lift_pp` on the fly — no pre-computation in individual run files

**Direction of improvement is hardcoded per metric inside each eval class** (lower is better for TOR and latency, higher for accuracy and GPT scores) and used by the comparison CLI to render `✓` and `↓` indicators.

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
- Scores each response via Gemini API LLM judge using prompt from MoshiRAG paper appendix (Table 16)
- Returns per-subset accuracy scores
- Respects mode sample size limits and resumability via `last_completed_index`

**`evals/registry/knowledge/halu_eval_audio.py`**
- Downloads from HuggingFace `kyutai/HaluEvalAudio_1000` if not cached
- Same pattern as OpenAudioBench — audio in, transcribed text out, LLM judge scoring
- Scores retrieved reference separately (ref. score) and final response (resp. score), matching paper Table 1 format
- Per-run JSON records raw `ref_acc` and `resp_acc` only — `rag_lift_pp` is computed by comparison CLI when diffing Config A vs Config B runs
- Respects mode sample size limits and resumability

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

**`evals/registry/latency/e2ekd.py`**
- Two-step pipeline:
  1. Gemini API extracts keyword from transcribed response using prompt from paper appendix (Table 17)
  2. `nvidia/parakeet-tdt-0.6b-v2` gives onset timestamp of that keyword in audio output
- E2EKD = TTFAT + keyword delay
- Default: 200 questions; `--full` runs complete set
- Reports mean TTFAT, mean keyword delay, mean E2EKD, P95 E2EKD
- `--spot-check` flag outputs 20 examples as numbered list of `(transcribed response, extracted keyword, verdict: correct/incorrect)` for human review before trusting metric at scale
- `spot_check_completed: false` flag in result metadata surfaces as reminder in console output until manually flipped to `true`

**`evals/registry/latency/retrieval_breakdown.py`**
- Timing hooks inside `GeminiAPIBackend.retrieve()` recording duration of:
  - ASR transcription wait
  - Gemini API call
  - Context injection
- Reports mean and P95 per stage
- Flags any run where P95 total exceeds `latency_gate_ms` — this surfaces in console output, JSON errors list, and is treated as a correctness concern not just a performance one

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
testing, edge-case probing, and stakeholder demos. It runs entirely on the
VM as kyutai-labs/moshi-rag's own, unmodified `moshi.server` +
`moshi.server_conditioner` — not custom code. macOS (or any machine with a
browser) connects via SSH tunnel — no Python, no model weights on the
client side.

Access pattern:
```
ssh -L 8998:localhost:8998 user@vertex-vm
# then open http://localhost:8998
```

`localhost` is treated as a secure context by browsers, so microphone
access works without TLS certificates.

### `scripts/run_demo.sh`

- Resolves a checkpoint alias via `core/checkpoint.py`, then launches
  `moshi.server_conditioner` (port 8001) and `moshi.server` (port 8998) in
  a detached tmux session — both real, unmodified moshi-rag processes
- `--checkpoint`, `--stt`, `--rag-timeout`, `--conditioner-only` flags —
  see the script's own header comment
- No Python code of ours runs in this path at all; retrieval on/off and
  everything else about session behavior is moshi.server's own, unmodified
  behavior

### Web client

Served by `moshi.server` itself from a static build — either the default
downloaded artifact or a custom fork of moshi-rag's own `client/`
(TypeScript/React/Vite, editable, Apache-2.0/MIT licensed) pointed at via
`moshi.server --static <dir>`. Already implements retrieval-visibility UI
(`SearchPanel`, `isRetrieving`, reference text display) — no custom client
code needed by default.

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

---

## Configs

### `configs/baseline_no_retrieval.yaml`
```yaml
model:
  checkpoint: base
  retrieval:
    enabled: false

evals:
  - knowledge.open_audio_bench
  - knowledge.halu_eval_audio
  - duplex.full_duplex_bench
  - latency.ttfat
  - latency.e2ekd
  - latency.retrieval_breakdown

output_dir: ./evals/results/
```

### `configs/baseline_with_retrieval.yaml`
```yaml
model:
  checkpoint: base
  retrieval:
    enabled: true
    backend: gemini_api
    model: gemini-3.5-flash
    latency_gate_ms: 1500

evals:
  - knowledge.open_audio_bench
  - knowledge.halu_eval_audio
  - duplex.full_duplex_bench
  - latency.ttfat
  - latency.e2ekd
  - latency.retrieval_breakdown

output_dir: ./evals/results/
```

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

Local dev (Ubuntu VM): `uv venv && uv sync`
Vertex AI VM: `make sync && make install` (install only needed when dependencies change)

---

## Infrastructure

- GCP Vertex AI, single `a2-highgpu-1g` VM (1 x A100 80GB)
- MoshiRAG 7B + 1B streaming ASR run in-process on the A100; the ARC-Encoder always runs as a separate `server_conditioner` process — for both the demo and evals, not just one or the other
- All LLM backend and judge calls go to Gemini API — no local LLM
- Checkpoints stored in GCS, referenced by URI or named alias
- Local dev uses uv directly on Ubuntu VM: `uv venv && uv sync`
- Production VM environment managed via `make sync` and `make install` — no Docker
- `.venv` excluded from rsync, built on VM to match Linux and CUDA environment
- Demo accessed from macOS (or any browser) via SSH tunnel to VM port 8998 — no software installation required on the client machine

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

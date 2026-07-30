# MoshiRAG Eval Pipeline

Evaluation harness for [kyutai-labs/moshi-rag](https://github.com/kyutai-labs/moshi-rag) — measures answer quality and latency of the retrieval-augmented voice model against benchmark question sets (LlamaQ, WebQ, and others), plus a live demo stack for manual testing.

Full requirements/spec: [`specs/moshirag-evals-requirements.md`](specs/moshirag-evals-requirements.md) — the authoritative source of truth for this project; `CLAUDE.md` is engineering/session notes, not spec.

## Layout

| Path | What it is |
|---|---|
| `core/` | Model adapter (`MoshiRAGAdapter`), GPU device resolution, checkpoint download |
| `evals/` | Eval runner, scoring, latency/retrieval-breakdown instrumentation |
| `configs/` | Eval configs (checkpoint, retrieval backend, dataset subset) |
| `demo/` | Instrumented launch script + forked web client for live manual testing |
| `scripts/` | VM setup, demo launch, diagnostic tooling |
| `tests/` | Unit tests (no GPU required) |
| `specs/` | Authoritative requirements doc |
| `docs/` | Standalone investigation reports and work plans, referenced from `CLAUDE.md` |

## Status

Actively developed. See `CLAUDE.md`'s dated sections for the current investigation thread and what's confirmed vs. still open on the VM.

**Current thread:** the demo's long turn-onset delay ("pad stall") was re-analysed from scratch on 2026-07-30 and traced to a regression this repo introduced, not model behaviour — see [`docs/demo-turn-onset-regression.md`](docs/demo-turn-onset-regression.md) for the evidence and [`docs/demo-turn-onset-fix-plan.md`](docs/demo-turn-onset-fix-plan.md) for the sequenced fix. One decisive VM measurement is still outstanding before any code changes. `.env`, model checkpoints, eval results, and demo/session recordings are gitignored — this repo is code and config only.

## Requirements & setup

GPU work (model loading, evals, demo) runs on a remote GCP VM, not locally — see `CLAUDE.md`'s "Environment" and "Remote instances" sections for connection details, first-time setup, and dependency installation (`uv sync --all-extras`).

## Running an eval

```bash
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke
```

See `CLAUDE.md`'s "Running evals" section for mode options (`tiny`/`smoke`/`sample`/`full`) and required setup (a `server_conditioner` sidecar process).

## Running the demo

```bash
make demo
```

Launches the unmodified `moshi.server` + `moshi.server_conditioner` stack; reachable via SSH tunnel at `http://localhost:8998`. See `CLAUDE.md`'s "Running the demo" section.

## Tests

```bash
uv run pytest
```

# Quick start

Command reference for running this repo. For first-time VM setup (SSH/IAP alias, `uv`
install, `.env`, checkpoint download), see `CLAUDE.md`'s "First-time setup on the VM"
section — this page assumes that's already done.

## Requirements & setup

GPU work (model loading, evals, demo) runs on a remote GCP VM, not locally — there is no
native macOS/local execution path. See `CLAUDE.md`'s "Environment" and "Remote instances"
sections for connection details and dependency installation:

```bash
uv sync --all-extras
```

## Running an eval

```bash
# Tiny — 1 question/subset, under a minute — dev-iteration sanity ping
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode tiny

# Smoke — 5 questions/subset, ~10 min — sanity check
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke

# Sample — 100-200 questions, ~1-2 hours — regression detection (default)
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml

# Full — complete datasets, 8-16 hours — final baseline
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode full
```

Requires a running `server_conditioner` process — see `CLAUDE.md`'s "Required setup:
`server_conditioner` process" section. Full mode/config details, `--compare`, and
`--spot-check` are in `CLAUDE.md`'s "Running evals" section.

## Running the demo

```bash
make demo
```

Launches `moshi.server_conditioner` + the instrumented `moshi.server` in a detached tmux
session. Reach it via SSH tunnel from a machine with the VM's SSH alias configured:

```bash
ssh -L 8998:localhost:8998 wb-gpu-a1ultra
```

Then open **http://localhost:8998** (`localhost` is a secure context, so microphone access
works without TLS). See `CLAUDE.md`'s "Running the demo" section for `--stt`/`--rag-timeout`
flags, session log locations, and `scripts/summarize_demo_session.py`.

## Tests

```bash
uv run pytest
```

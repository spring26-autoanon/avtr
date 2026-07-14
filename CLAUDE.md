# CLAUDE.md — MoshiRAG Eval Pipeline

## Environment

Running inside a minimal Docker container for local dev. `uv` is installed in the image, use for all Python execution. On the VM, `uv` is available directly.

- `uv run <script>` — run any Python script
- `uv run python -c "..."` — run inline Python
- `uv run pytest` — run tests
- `uv add <package>` — add a dependency (ask first)

There is no standalone `python`, `python3`, or `pip` on PATH. Always prefix Python execution with `uv run`.

## Workflow

- Reason about correctness statically before reaching for execution
- Run `uv run pytest` to verify significant changes — not after every edit
- Do not install new packages (`uv add`) without asking first
- Do not read, reference, or output the contents of `.env` files
- When writing a temporary diagnostic script, save it as a `.py` file and run it with `uv run <script>.py`

## Commits

Always append the following trailer to every commit message, separated from the body by a blank line:

Co-authored-by: Claude <claude@anthropic.com>

---

## Remote instance

```
GCP project:  adsp-s26-autoanon
Zone:         us-central1-c
Instance:     wb-gpu-a1ultra
User:         jupyter
Remote path:  /home/jupyter/moshirag-evals
Connection:   gcloud compute ssh with --tunnel-through-iap (no external IP)
SSH alias:    <SSH_ALIAS>   ← fill in your ~/.ssh/config shortname
```

### gcloud shorthand (add to your shell profile)

```bash
alias gcloudssh='gcloud compute ssh jupyter@wb-gpu-a1ultra \
  --project=adsp-s26-autoanon --zone=us-central1-c --tunnel-through-iap'
alias gcloudcmd='gcloud compute ssh jupyter@wb-gpu-a1ultra \
  --project=adsp-s26-autoanon --zone=us-central1-c --tunnel-through-iap --'
```

---

## First-time setup on the VM

Steps 1–2 are on your **local machine**; steps 3 onward are inside an SSH session.

### Step 1 — Generate SSH config entry (local machine)

```bash
gcloud compute config-ssh --project=adsp-s26-autoanon
```

This writes a block named `wb-gpu-a1ultra.us-central1-c.adsp-s26-autoanon` into `~/.ssh/config`.

### Step 2 — Sync code to remote (local machine)

```bash
make sync LOCAL_DIR=/path/to/moshirag-evals
```

Or equivalently:

```bash
rsync -avz \
  --exclude '.venv' --exclude '__pycache__' --exclude 'checkpoint_cache' \
  --exclude 'evals/results' --exclude 'demo/sessions' --exclude '*.pyc' \
  --exclude '.env' --exclude '.git' \
  /path/to/moshirag-evals/ jupyter@wb-gpu-a1ultra:/home/jupyter/moshirag-evals/
```

### Step 3 — SSH into the VM

```bash
gcloud compute ssh jupyter@wb-gpu-a1ultra \
  --project=adsp-s26-autoanon --zone=us-central1-c --tunnel-through-iap
```

### Step 4 — Verify GPU

```bash
nvidia-smi
```

### Step 5 — Install dependencies

```bash
cd /home/jupyter/moshirag-evals
uv sync --extra dev
```

Only re-run after changes to `pyproject.toml`. For code-only changes, `make sync` is sufficient.

### Step 6 — Download model checkpoint (inside tmux — survives disconnects)

```bash
tmux new-session -s setup
cd /home/jupyter/moshirag-evals
uv run python -c "from core.checkpoint import resolve_checkpoint; resolve_checkpoint('base')"
# Ctrl-b d to detach;  tmux attach -t setup  to reattach
```

---

## Code change workflow

```bash
# 1. Edit locally, then sync
make sync LOCAL_DIR=/path/to/moshirag-evals

# 2. Run smoke eval on remote to confirm nothing is broken
make smoke

# 3. Pull results back if needed
rsync -avz jupyter@wb-gpu-a1ultra:/home/jupyter/moshirag-evals/evals/results/ ./evals/results/
```

---

## Running evals

```bash
# Smoke — 5 questions/subset, ~10 min, sanity check
uv run evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke

# Sample — 100-200 questions, ~1-2 hours, regression detection (default)
uv run evals/runner.py --config configs/baseline_with_retrieval.yaml

# Full — complete datasets, 8-16 hours, final baseline
uv run evals/runner.py --config configs/baseline_with_retrieval.yaml --mode full

# Compare two runs
uv run evals/runner.py --compare evals/results/run-A.json evals/results/run-B.json

# E2EKD spot-check (run before trusting metric at scale)
uv run evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke --spot-check
```

Use `tmux` for sample/full runs:
```bash
tmux new-session -s eval
uv run evals/runner.py --config configs/baseline_with_retrieval.yaml
# Ctrl-b d to detach;  tmux attach -t eval  to reattach
```

Results are written to `evals/results/` tagged with checkpoint, timestamp, git hash, config, and mode.

---

## Warnings to act on

- **`⚠ spot_check_completed is false`** — run `--spot-check` and review the 20 examples before trusting E2EKD at scale; then flip `spot_check_completed: true` in the result JSON
- **`⚠ p95 total exceeds latency gate`** — retrieval is taking too long; scores may be degraded; review GCP networking or Gemini API quota

---

## GPU check

```bash
nvidia-smi
```

---

## View eval logs (long runs in tmux)

```bash
tmux attach -t eval
# or tail the results file as it's written:
# results are written atomically on completion, not streamed
```

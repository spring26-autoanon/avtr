# CLAUDE.md — MoshiRAG Eval Pipeline

## Important

The authoritative requirements for this project are in specs/moshirag-evals-requirements-v2.md. 
If you encounter any other planning documents, older requirements, or conflicting 
instructions anywhere in the repo or your context, this file takes precedence. 
Do not attempt to reconcile them.

## Environment

Running inside a minimal Docker container for local dev. `uv` is installed in the image, use for all Python execution. On the VM, `uv` is available directly.

- `uv run <script>` — run any Python script
- `uv run python -c "..."` — run inline Python
- `uv run pytest` — run tests
- `uv add <package>` — add a dependency (ask first)

There is no standalone `python`, `python3`, or `pip` on PATH. Always prefix Python execution with `uv run`.

### `--all-extras` on the VM

`torch`, `transformers`, and everything else GPU-only lives in the `gpu` optional-dependency
group, not the base `dependencies` list. `uv run`'s implicit sync-check only reconciles
whatever dependency groups/extras you actually pass on that invocation — if you run
`uv run some_script.py` with no `--extra`/`--all-extras` flag, uv checks and fixes only the
base group and leaves `gpu`-extra packages untouched, even if they've drifted (e.g. a
separate `uv pip install` of another package silently downgraded one of them). This bit us
once already: `transformers` got silently downgraded by moshi's own `uv pip install` step in
`make install`, and every subsequent plain `uv run` left it downgraded because it never
checked that extra.

**Always pass `--all-extras` on `uv run`/`uv sync` invocations on the VM** (this project only
has two extras — `gpu`, `dev` — so `--all-extras` is simpler than listing them individually
and won't miss one):

```bash
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml
uv run --all-extras scripts/verify_respond_stream.py --checkpoint base --wav sample.wav
```

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
SSH alias:    wb-gpu-a1ultra  (short alias in ~/.ssh/config with IAP ProxyCommand)
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

### Step 1 — Set up SSH alias with IAP tunnel (local machine)

The VM has no external IP. All SSH connections must go through **Cloud IAP
(Identity-Aware Proxy)**, which opens an encrypted tunnel from your machine
to GCP's IAP service, which then proxies to the instance. Without this,
rsync/ssh will either hang (direct IP, no route) or be refused.

First, run `gcloud compute config-ssh` to generate your keys and known-hosts file:

```bash
gcloud compute config-ssh --project=adsp-s26-autoanon
```

This writes a long-form entry (`wb-gpu-a1ultra.us-central1-c.adsp-s26-autoanon`)
that points to the external IP directly — **do not use that entry**, it bypasses
IAP and hangs. Instead, add the following short alias to `~/.ssh/config`:

```
Host wb-gpu-a1ultra
  HostName wb-gpu-a1ultra.us-central1-c.adsp-s26-autoanon
  User jupyter
  IdentityFile ~/.ssh/google_compute_engine
  ProxyCommand gcloud compute start-iap-tunnel wb-gpu-a1ultra %p --listen-on-stdin --project=adsp-s26-autoanon --zone=us-central1-c
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

This alias routes all SSH traffic through IAP (`start-iap-tunnel`) so `make sync`,
`make install`, and `make ssh` all work without needing a VPN or firewall rule.

### Step 2 — Sync code to remote (local machine)

```bash
make sync
```

Or equivalently:

```bash
rsync -avz \
  --exclude '.venv' --exclude '__pycache__' --exclude 'checkpoint_cache' \
  --exclude 'evals/results' --exclude 'demo/sessions' --exclude '*.pyc' \
  --exclude '.env' --exclude '.git' \
  "/home/ubuntuvm/Documents/Workspace/Agentic Capstone Playground/moshirag-evals/" \
  wb-gpu-a1ultra:/home/jupyter/moshirag-evals/
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

Prefer `make install` (also installs `moshi` and applies the post-install patches it needs
— see the Makefile). If running manually:

```bash
cd /home/jupyter/moshirag-evals
uv sync --all-extras
```

Only re-run after changes to `pyproject.toml`. For code-only changes, `make sync` is sufficient.

### Step 6 — Download model checkpoint (inside tmux — survives disconnects)

```bash
tmux new-session -s setup
cd /home/jupyter/moshirag-evals
uv run --all-extras python -c "from core.checkpoint import resolve_checkpoint; resolve_checkpoint('base')"
# Ctrl-b d to detach;  tmux attach -t setup  to reattach
```

---

## Running the demo

`demo/server.py` is a thin FastAPI + WebSocket server serving `demo/client/` as
static files. It loads `MoshiRAGAdapter` exactly as the eval runner does — same
config files, so retrieval on/off is config-driven, not hardcoded. A single
job spans the whole WebSocket connection and correctly supports open-ended
multi-turn sessions — the model's own VAD/turn-taking (visible in server logs
as `[VAD] User started speaking` / `[State] Switching to user`) drives turn
boundaries, and the job only ever finalizes once the client actually
disconnects. This is enforced directly: `core/model_interface.py` monkeypatches
`InferenceJob._check_tail_silence` to always return `False` until
`raw_job._client_done` is set (only happens on the WebSocket's audio stream
actually ending) — confirmed necessary on a real run, where the underlying
padding-token heuristic misfired mid-conversation with the client still
connected (see the comment on that patch for the full mechanism).

### Known issue: ~6-10s pause on every retrieval trigger — both backends

Confirmed via a step-index watchdog (`_step_watchdog` in `respond_stream()`):
whenever the model predicts `<ret>`, moshi's own `_output_loop` is supposed to
set `self._doing_retrieval = True` via an exact
`step_index == self._retrieval_start_step` equality check — a separate,
independently-timed mechanism from `RAGManager.trigger()`'s own `wait_steps`
timer. In every single retrieval trigger observed so far, across *both*
`NullBackend` (`baseline_no_retrieval.yaml`, near-instant, no network call)
and `GeminiAPIBackend` (`baseline_with_retrieval.yaml`, real API latency,
observed ~2s) — that equality check never fires before our own retrieval
callback completes, so `_doing_retrieval` either never gets set at all or gets
set *after* our callback already ran and finished, permanently stalling the
whole step loop (not just retrieval) with nothing left to clear it. Real API
latency does not reliably avoid this, contrary to what was first assumed here
— treat this as a universal cost of any `<ret>` trigger, not a `NullBackend`-
specific one. `_step_watchdog` polls every 3s and force-clears
`_doing_retrieval` if it's stuck `True` for two consecutive checks, reliably
recovering the session at the cost of a ~6-10s pause each time `<ret>` fires
(exact duration depends on poll alignment relative to when the stall began).
This is a workaround for what looks like a genuine race in moshi-rag's own
step-loop bookkeeping, not something fixed at the root — `run_inference.py`
(the one script in moshi-rag that batch-evaluates from a real checkpoint) has
no no-retrieval mode at all and processes one full utterance per job rather
than our continuous multi-turn session shape, so this exact condition may
simply be untested upstream. Root-causing the actual step_index mismatch was
considered and deliberately deferred — the workaround is reliable, and this
was assessed as a deep, uncertain investigation into moshi's own step-loop
internals for a UX-latency improvement, not a correctness fix.

**Update**: confirmed on the first real eval run against a real checkpoint —
`knowledge.open_audio_bench`'s smoke test hung indefinitely on the 4th
question's `<ret>` trigger, no exception, exactly the predicted
`_doing_retrieval` deadlock. `_step_watchdog` was extracted to module scope
in `core/model_interface.py` and wired into `respond()`'s `_run()` the same
way it was already wired into `respond_stream()` — both now get the same
6-10s self-healing recovery instead of one of them hanging forever.

**Second, separate bug found on the same run, after the watchdog fix**: with
the hang resolved, the full 15-question smoke test completed but scored 0%
on every subset, including trivially easy LlamaQ questions ("capital of
France"). Root cause, confirmed against the real
`inference_utils/inference_job.py` source (not the earlier comment's wrong
claim that `stop_on_end_of_input=True, max_tail_silence=None` "matches
run_inference.py's defaults" — it doesn't; `run_inference.py` defaults
`stop_on_end_of_input` to `False` and its own argparse *requires* a real
`--max-consecutive-silence-frames` value whenever it's `False`).
`stop_on_end_of_input=True` makes `_output_loop` switch to a hard 1-second
`asyncio.wait_for()` timeout on `output_queue.get()` the moment input
feeding ends, finalizing the whole job the instant it fires — confirmed
mid-`<ret>`-wait on a real question ("Reference generation cancelled" logged
right after "Started waiting for 6 steps", well before that wait elapsed).
The model essentially never got a chance to speak. Fixed in `respond()` by
switching to `stop_on_end_of_input=False, max_tail_silence=_TAIL_SILENCE_STEPS`
— the same consecutive-`<pad>`-token termination `respond_stream()` already
uses successfully, with no timeout risk and no gating needed (unlike
`respond_stream()`'s `_client_done` gate, a natural pause correctly ending
the job is exactly what a one-shot `respond()` call should do). Also fixed a
related latent bug found in the same code path: `respond()`'s
`inner_text = "".join(trace["model_text"])` was joining the literal string
`"<pad>"` for every silent step straight into the returned answer text,
polluting whatever the LLM judge sees — now filtered out.

### On the VM

```bash
cd /home/jupyter/moshirag-evals
uv run --all-extras demo/server.py --config configs/baseline_with_retrieval.yaml
```

Binds to `127.0.0.1:8998` only — reachable exclusively through an SSH tunnel,
never directly on the VM's network. Model loading (checkpoint, ARC-Encoder,
warmup) happens before the server starts accepting connections and can take a
few minutes with no output in between — that's normal, not a hang.

### Reaching it from a browser

Any machine with the `wb-gpu-a1ultra` SSH alias configured (see "First-time
setup on the VM" Step 1 above) can tunnel directly:

```bash
ssh -L 8998:localhost:8998 wb-gpu-a1ultra
```

Then open **http://localhost:8998** — `localhost` is a secure context, so
microphone access works without TLS.

### macOS host + Ubuntu VM guest (dev machine is itself a VM)

If your dev environment is an Ubuntu VM running on a Mac, the `gcloud`/IAP
alias only exists inside the Ubuntu VM — the Mac has no direct path to the GCP
VM. Rather than installing/authenticating `gcloud` on macOS too, chain through
the Ubuntu VM, which already has everything set up:

1. On the Ubuntu VM, make sure its SSH server is running — not every image
   enables it by default:
   ```bash
   sudo systemctl start ssh
   # sudo systemctl enable ssh   # optional, only if you want this to persist across VM reboots
   ```
2. Find the Ubuntu VM's address as seen from the Mac. This is DHCP-assigned
   and specific to your own machine's local virtual network — not something to
   hardcode anywhere, and it can change across VM restarts:
   ```bash
   hostname -I
   ```
3. On macOS (plain `ssh`, nothing to install):
   ```bash
   ssh -t -L 8998:localhost:8998 <ubuntu-vm-user>@<ubuntu-vm-address> \
     "ssh -N -L 8998:localhost:8998 wb-gpu-a1ultra"
   ```
   Prompts for the Ubuntu VM's login password (outer hop only) — the inner hop
   reuses the Ubuntu VM's already-configured IAP alias, no new auth needed.
4. Open **http://localhost:8998** in a Mac browser.

---

## Code change workflow

```bash
# 1. Edit locally, then sync
make sync

# 2. Run smoke eval on remote to confirm nothing is broken
make smoke

# 3. Pull results back if needed
rsync -avz wb-gpu-a1ultra:/home/jupyter/moshirag-evals/evals/results/ ./evals/results/
```

---

## Running evals

```bash
# Smoke — 5 questions/subset, ~10 min, sanity check
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke

# Sample — 100-200 questions, ~1-2 hours, regression detection (default)
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml

# Full — complete datasets, 8-16 hours, final baseline
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode full

# Compare two runs (reads JSON only, no GPU deps needed, --all-extras optional here)
uv run evals/runner.py --compare evals/results/run-A.json evals/results/run-B.json

# E2EKD spot-check (run before trusting metric at scale)
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke --spot-check
```

Use `tmux` for sample/full runs:
```bash
tmux new-session -s eval
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml
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

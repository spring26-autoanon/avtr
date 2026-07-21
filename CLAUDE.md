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
uv run --all-extras python -c "from core.checkpoint import resolve_checkpoint; resolve_checkpoint('base')"
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

**Architecture note (see specs/moshirag-evals-requirements-v2.md's "New
Marching Orders"):** the demo no longer runs through `MoshiRAGAdapter`/a
custom FastAPI+WebSocket server. It launches kyutai-labs/moshi-rag's own,
unmodified `moshi.server` + `moshi.server_conditioner` directly via
`scripts/run_demo.sh` — the same production stack `prodcheck/` was built to
A/B against, now promoted to the real thing. Reversibility is via git history
(the prior in-process implementation is recoverable from before this change,
same as this version is itself a revival of the pre-`5b3790e` launcher).
`demo/server.py` and `demo/client/` are deleted; there is no Python code left
in `demo/` and nothing here loads `MoshiRAGAdapter` anymore.

Pivot status: Phase 0 (root-cause investigation), Phase 1 (this demo pivot),
and Phase 2 (making separate-process conditioning mandatory for
`respond()`/evals — see the "RESOLVED" section below) are all done. The
sections below still carry real diagnostic history from before and during
the pivot — kept because the investigation techniques and confirmed findings
remain useful, trimmed of narrative that's now purely dead-code reference.
Where something describes `respond_stream()` specifically: that method and
the demo it served no longer exist — those mentions are historical context
for *why* a given bug was scoped the way it was, not live behavior.

### Known issue (rare, unconfirmed root cause): reference-encoder `ConnectError` kills the session

Observed once across 4 real-checkpoint sessions run against the pivoted
`scripts/run_demo.sh` stack (2 via `prodcheck/`, 2 via the demo itself): a
retrieval round-trip completed normally (real Gemini reference text came
back), but the subsequent POST to `server_conditioner` at
`http://localhost:8001/embed` failed outright with
`httpx.ConnectError: All connection attempts failed` — not a timeout, a
refused/failed connection. The conditioner process itself never crashed
(confirmed via its own log and `nvidia-smi` afterward: still alive, no
errors logged, in fact zero requests logged for the entire session). moshi-
rag's `channel.py` has no exception handling around this call — unlike a
retrieval-LLM timeout, which is caught and degrades gracefully to an empty
reference, a conditioner-connection failure is unhandled and propagates up
through the session's `TaskGroup`, killing the whole WebSocket connection
mid-conversation, not just that one turn.

A deliberate repro attempt (same checkpoint, same conversation shape,
including a genuine double-retrieval-trigger case, GPU memory/utilization
polled at 1s resolution throughout) did not reproduce it — memory stayed
~13GB under the A100's 80GB, utilization fluctuated 54-85% during active
generation with no obvious spike or starvation around the original failure.
**Best current read: a rare, non-reproducible transient blip, not a
systematic resource-contention or code defect** — but the underlying gap
(no error handling around the conditioner call) is real regardless of how
rarely it fires. Not blocking the demo (a human just reconnects), but worth
carrying into Phase 2's design: an eval run is long and unattended, so the
same rare event there wouldn't have a human around to notice and retry.

### `respond()`-only: `_doing_retrieval` step-loop stall, mitigated (root cause: self-inflicted, not upstream)

`core/model_interface.py`'s `_TimedInferenceJob._patch_output_loop()` and
`_step_watchdog` are still live code, still applied to every `respond()`
call — this section documents why they exist and stays current, unlike most
of this file's other historical notes. The demo no longer needs either
(see below); `respond()`/evals still do.

**The original bug**: `InferenceJob`'s own `_output_loop` gates output on
`self._doing_retrieval`, set via an exact `step_index ==
self._retrieval_start_step` equality check that doesn't reliably fire before
the retrieval callback completes — this stalls the whole step loop (not just
retrieval) with nothing left to clear it, observed on every real `<ret>`
trigger regardless of retrieval backend or latency. `_patch_output_loop()`
replaces `InferenceJob._output_loop` wholesale with a version matching
`Channel`'s (moshi-rag's real WebSocket server class) simpler logic, which
has no such gate at all — see that method's docstring for the full
root-cause chain. `_step_watchdog` is a defensive fallback on top: polls
`step_index` every 3s and force-clears `_doing_retrieval` if stuck, in case
some other stall mechanism `_patch_output_loop` doesn't cover ever surfaces.
Both were applied to `respond()` and the old `respond_stream()` identically
when this was found — `respond()` was never "missing" this fix at any point
relevant to the pivot.

**Phase 0 update (post-pivot investigation, see `prodcheck/`)**: re-ran this
exact scenario against the real, unmodified `moshi.server` stack (which has
no `_doing_retrieval` gate at all, by construction). Across 5 independent
`<ret>` triggers over four sessions, including back-to-back triggers in one
turn: **zero stalls, zero deadlocks.** Real evidence — small sample, not
proof — that this bug was introduced by `core/model_interface.py`'s
reimplementation, not an upstream defect. Since the demo now runs the
unmodified stack directly, it doesn't need `_patch_output_loop`/
`_step_watchdog` at all — `respond()`/evals still does, since it still
drives `InferenceJob` (the batch-shaped class, correctly — `moshi.server`'s
`Channel` has no batch/resumable concept, see Phase 2 investigation notes
in [[project_moshirag_production_pivot]] memory) rather than `Channel`
directly. Not revisited as part of Phase 2 — that phase fixed a different,
unrelated bug (conditioning, below) and left this patch untouched. Whether
it's still strictly necessary for `respond()` given everything else learned
during the pivot is an open question, not yet tested.

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

### RESOLVED (Phase 2): `respond()`-only silent-response bug when retrieval is enabled

Third, separate bug, found once the above two were fixed and `open_audio_bench`
was run **with retrieval enabled**: the first `respond()` call on a
`MoshiRAGAdapter` instance produces a correct, retrieval-grounded answer, but
every subsequent `respond()` call goes **completely silent** — zero non-`<pad>`
tokens for the whole turn, no `<ret>` ever predicted, endless "[VAD] User
stopped speaking but LM buffer empty, remaining in user turn." Only manifests
when retrieval is enabled; the earlier no-retrieval smoke test ran 15
sequential questions with no such issue.

Three targeted hypotheses were tested against a real checkpoint and each
**ruled out** with real diagnostic evidence, in order:
1. **Stale RAG conditioning surviving into the next job** — ruled out.
   `BatchRunner.run_step()`'s own `is_first`-triggered reset was confirmed
   (via temporary logging, since removed) to correctly clear
   `pending_streaming_sums` before the second call's first frame.
2. **A cross-thread CUDA sync gap** between the conditioning update
   (`_handle_reference_text` → `update_streaming_sum_tensors`, an async GPU
   copy) and the next generation step — ruled out. An explicit
   `torch.cuda.synchronize()` there (also since removed) made no difference.
3. **Per-call event-loop/step-task teardown** — `respond()` used to call
   `asyncio.run(_run())` fresh per question (new event loop, new
   `state._step_loop()` task, every time), unlike `run_inference.py`'s own
   structure (one step loop, created once, running continuously across every
   file in the batch). Ruled out — switching `respond()` to a persistent
   event loop + step task (`MoshiRAGAdapter._ensure_step_loop()`, kept for
   the adapter's whole lifetime) made no difference either.

**What did work**: pointing conditioning at a real, separate-process
`server_conditioner.py` sidecar (`REFERENCE_ENCODER_URL` env var — see
`MoshiRAGAdapter`'s "Conditioning" docstring) instead of the in-process
ARC-Encoder fixed it, confirmed in two independent real-checkpoint tests
(different questions, both flipped from silent to correct). This points at
something about conditioning and generation sharing one CUDA context/process
— the in-process ARC-Encoder is a deliberate project requirement (see specs),
something moshi-rag's own reference architecture never has to handle since
its encoder always runs in a genuinely separate process.

**Critically, `respond_stream()` (the old demo) does NOT have this bug** —
`respond_stream()` and the demo it served no longer exist post-pivot (see the
architecture note above), so this is now historical context for *why* the bug
is scoped to `respond()`, not a live comparison against the current demo.
A real two-question session (retrieval on, generous silence gap between
questions) responded correctly both times, with genuine multi-turn memory
(question 2's retrieval context correctly included question 1's full
exchange). This made sense structurally: the demo's one continuous session
only ever did the
`is_first` reset once, at true session start, and never re-exercised whatever
`respond()`'s per-call fresh-job pattern hits. So this was scoped to
`respond()`/evals specifically, not a general in-process-conditioning problem
— and it should **stay** scoped there: making `respond()` session-like (to
match how `respond_stream()` avoids the bug) would be wrong, since eval
questions need to be scored independently, and a shared session would leak
prior questions' context into later ones' retrieval and generation (observed
directly: question 2's retrieval context included question 1's Q&A verbatim
in the demo test — correct there, would silently corrupt eval scores here).

**Status (Phase 2 update)**: separate-process conditioning is now
**mandatory**, not an opt-in workaround — `MoshiRAGAdapter._load_models()`
raises immediately if `REFERENCE_ENCODER_URL` is unset; the in-process
ARC-Encoder loading path (`_load_arc_encoder`, `_patch_inprocess_conditioning`)
has been deleted entirely, not just made optional. Root cause of *why* the
in-process path broke is still not pinned down precisely (we know separate-
process fixes it, not the exact mechanism) — but given the Phase 0/1 pivot
findings (moshi-rag's real production architecture never runs the
ARC-Encoder in-process for *any* path, `respond_stream()` included, now that
the demo runs the unmodified stack directly), "always separate-process" is
the permanent fix, not a stopgap pending one. Practical implication: every
`respond()`/eval run now requires a `server_conditioner` process running
alongside it — see "Required setup: `server_conditioner` process" under
"Running evals" below.

### On the VM

```bash
cd /home/jupyter/moshirag-evals
bash scripts/run_demo.sh --checkpoint base
```

Or from your local machine: `make demo`. Launches `moshi.server_conditioner`
(port 8001) and `moshi.server` (port 8998) in a detached tmux session named
`demo` and returns immediately — see the script's own `--help`-equivalent
header comment for `--stt` and `--rag-timeout` options. `moshi.server` binds
to all interfaces on 8998 by default but is only reachable through an SSH
tunnel in practice, since the VM has no external IP. Model loading
(checkpoint, ARC-Encoder, warmup) happens before the server starts accepting
connections and can take a few minutes with no output in between — that's
normal, not a hang. `tmux attach -t demo` to watch both services' logs.

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
# Tiny — 1 question/subset, under a minute per eval — rapid dev-iteration
# sanity ping while building/debugging one eval at a time. Too small a
# sample to trust any score from; not a substitute for smoke.
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode tiny

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

### Required setup: `server_conditioner` process (Phase 2, no longer optional)

See "RESOLVED (Phase 2): `respond()`-only silent-response bug" above —
`MoshiRAGAdapter` now **requires** `REFERENCE_ENCODER_URL` to be set and a
real `server_conditioner` process reachable at that address; it raises at
construction time otherwise. This applies to **every** eval run, not just
ones with retrieval enabled in the config — `ServerState` itself needs a
working `reference_encoder_url` regardless of whether any given eval's
config turns retrieval on.

Easiest: `scripts/run_demo.sh --conditioner-only` launches just the
conditioner (skips loading the full model a second time) via a synced
script file, not a hand-typed command — avoids a real, confirmed failure
mode where long multi-line commands get corrupted pasting into some VM
terminals (JupyterLab's web terminal hard-wraps long lines, breaking both
`\`-continuations and single long lines):

```bash
# Terminal 1 — start the conditioner
bash scripts/run_demo.sh --checkpoint base --conditioner-only

# Terminal 2 — run the eval pointed at it
export REFERENCE_ENCODER_URL=http://localhost:8001
uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml
```

If the demo (the full `scripts/run_demo.sh`, not `--conditioner-only`) is
already running on the same VM, evals can point at that same conditioner
instance instead of starting a second one — same port 8001 — but note that
also means a second full model load for the eval process itself, GPU memory
permitting (a single model + conditioner already uses ~66-69GB of the A100's
80GB, observed during Phase 1 verification — two full model loads at once
likely won't fit).

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

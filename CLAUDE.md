# CLAUDE.md — MoshiRAG Eval Pipeline

## Important

The authoritative requirements for this project are in specs/moshirag-evals-requirements.md. 
If you encounter any other planning documents, older requirements, or conflicting 
instructions anywhere in the repo or your context, this file takes precedence. 
Do not attempt to reconcile them.

## Ground-truth figures from the papers (keep these to hand)

Small, high-leverage reference table. These are the **yardstick** — without
them there is no way to tell "this system is broken" from "this is how the
model behaves", and getting that wrong has cost this project weeks twice.
Added 2026-07-30 after Table 1's TTFAT figure single-handedly overturned a
confident, months-old wrong conclusion (see `docs/demo-turn-onset-regression.md`).

Sources: MoshiRAG, arXiv 2604.12928v3. Moshi, arXiv 2410.00037. Read as PDF
and verified against the real text, not summarised from notes.

| Figure | Value | Where |
|---|---|---|
| **TTFAT** (end of user utterance -> first audio token) | **0.0 s** for MoshiRAG *and* vanilla Moshi | MoshiRAG Table 1 |
| Keyword delay / E2EKD | 3.1 s / 3.1 s (vanilla Moshi: 2.1 / 2.1) | MoshiRAG Table 1 |
| **Full-Duplex-Bench turn-taking latency** | **0.18 s** (TOR 0.83) | MoshiRAG Table 2 |
| E2EKD of other speech LMs | "often exceeds 3 seconds" | MoshiRAG §3.1 |
| Retrieval-delay design budget | <= **2 s** end to end; 1.5 s assumed uniform for non-Gemma back ends (>90% of local Gemma cases) | MoshiRAG §3.1, footnote 11 |
| ARC-Encoder compression | **4x** on reference token count | MoshiRAG §3.3.1 |
| Training-time audio gate | 80 ms window, RMS below **-65 dBFS zeroed** | MoshiRAG §4.2 |
| `<ret>` placement in training data | replaces the text token immediately *before* the lead portion — so `<ret>` -> speech should be fast | MoshiRAG §4.2, Fig. 4 |
| Conditioning injection | one embedding per step over `l` steps, starting `d` seconds after `<ret>` | MoshiRAG Eq. 2 |
| **Padding share of text tokens** | **~65%** in English conversational speech | Moshi §3.4.2 |
| Turn-onset control lever | "forcing the sampling of a EPAD token will make Moshi start talking immediately"; the paper itself uses a small logit bonus on PAD/EPAD for TTS rate control | Moshi §3.4.4 / Appendix |
| User-stream augmentation (why acoustics are rarely the bug) | random gain **-24..+15 dB** 50% of the time; DNS noise 30%; silences up to **30 s**; **emulated Moshi->mic echo** (scale 0-0.2, delay 100-500 ms); reverb | Moshi §4.2 instruct stage |

Token ids, from real source plus one empirical derivation:

| Token | Id | How known |
|---|---|---|
| PAD | **3** | `models/loaders.py: existing_text_padding_id: 3` |
| EPAD | **0** | derived: 146/146 occurrences immediately precede a real word, PAD never does (`step_diag.jsonl` token stream) |
| `<ret>` | **4** | `models/loaders.py: rag_token_id: 4` |

Frame timing: Mimi runs at **12.5 Hz**, so one step is **80 ms** — the hard
real-time budget for anything on the step loop. The released checkpoint's
`context: 3000` (confirmed from the real `config.json` on the VM) is ~240 s of
ring-buffer attention, so session length does not change attention cost.

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

## Remote instances

**Current state (2026-07-30): use single-GPU `wb-gpu-a1ultra` for
everything, demo and evals alike. `wb-gpu-a1ultra2g` has no remaining
justification and should stay stopped.** Both reasons it ever existed have
now been disproved — the second one as of this date, when the Option E
chain that depended on it was deleted (see
`docs/demo-development-history.md`'s "STT/front-end CUDA-graph
cross-thread crash" section, and `docs/demo-turn-onset-regression.md`).
`core/gpu.py` no longer exposes a second GPU to the front-end process at
all. Everything below this paragraph is the history of how that
two-instance setup came about; nothing in it is a live instruction.

Two GCP Vertex AI VMs, same project and zone. See specs/
moshirag-evals-requirements.md's "GPU Sizing and Multi-GPU Deployment"
section for the full history of why both exist — it's had two, unrelated
reasons at two different times, not one continuous story:

- **Original reason (superseded)**: `wb-gpu-a1ultra2g` was provisioned to
  resolve a suspected conditioner-vs-front-end GPU contention finding.
  That was confirmed wrong (see `docs/demo-development-history.md`'s
  "SUPERSEDED: GPU contention conclusion was wrong" section) — the real
  cause was event-loop starvation, unrelated to GPU topology.
  `wb-gpu-a1ultra2g` was stopped
  (not deleted) once this was confirmed, and single-GPU `wb-gpu-a1ultra`
  was declared sufficient for all further work.
- **Current reason (SUPERSEDED 2026-07-30 — the code this depended on is
  deleted; `wb-gpu-a1ultra2g` is no longer needed for anything)**: a completely separate
  investigation — STT's own in-process model duplicate sharing a CUDA
  context/stream with the front-end's main model via `asyncio.to_thread`
  — found a real, independent reason dual-GPU matters again: moshi-rag's
  own CUDA-graph-capture machinery isn't safe to call from two OS threads
  concurrently, in a way that repeated attempts at locking specific call
  sites couldn't fully close. The fix (Option E — see
  `docs/demo-development-history.md`'s "STT/front-end CUDA-graph
  cross-thread crash" section for the full reasoning and investigation)
  pinned STT's own model duplicate to a second physical GPU. This whole
  chain, including `_patch_stt_second_gpu()`, was later deleted (see
  "Current state and known issues" under "Running the demo") once the
  backlog it existed to mitigate turned out to be self-inflicted — kept
  here only as the historical reasoning for why `wb-gpu-a1ultra2g` was
  ever provisioned a second time — none of the mechanics in this bullet
  (`_patch_stt_second_gpu()`, `_patch_cuda_graph_thread_local()`) exist in
  the codebase anymore.

`wb-gpu-a1ultra` remains valid for `tiny`/`smoke` dev-iteration and
no-retrieval configs; for scored retrieval-enabled `sample`/`full` runs or
the demo, prefer `wb-gpu-a1ultra2g` now that Option E depends on the
second GPU actually being present. The codebase itself needs no
per-instance configuration to target either one — `core/gpu.py`
auto-detects real GPU count at runtime on whichever box it's rsynced to
(see that section). The Makefile's `INSTANCE`/`SSH_ALIAS`/`REMOTE_DIR`
variables (all `?=`, overridable) mean every `make` target already works
against either box, e.g. `make sync SSH_ALIAS=wb-gpu-a1ultra2g`.

### `wb-gpu-a1ultra` (single A100 80GB)

```
GCP project:  adsp-s26-autoanon
Zone:         us-central1-c
Instance:     wb-gpu-a1ultra
User:         jupyter
Remote path:  /home/jupyter/moshirag-evals
Connection:   gcloud compute ssh with --tunnel-through-iap (no external IP)
SSH alias:    wb-gpu-a1ultra  (short alias in ~/.ssh/config with IAP ProxyCommand)
```

### `wb-gpu-a1ultra2g` (2x A100 80GB)

Provisioning as of this writing — same GCP project/zone/user/remote-path
convention as `wb-gpu-a1ultra` above, confirmed unchanged (same project,
just a second VM in it).

```
GCP project:  adsp-s26-autoanon
Zone:         us-central1-c
Instance:     wb-gpu-a1ultra2g
User:         jupyter
Remote path:  /home/jupyter/moshirag-evals
Connection:   gcloud compute ssh with --tunnel-through-iap (no external IP)
SSH alias:    wb-gpu-a1ultra2g  (short alias in ~/.ssh/config with IAP ProxyCommand)
```

### gcloud shorthand (add to your shell profile)

```bash
# wb-gpu-a1ultra (single A100)
alias gcloudssh='gcloud compute ssh jupyter@wb-gpu-a1ultra \
  --project=adsp-s26-autoanon --zone=us-central1-c --tunnel-through-iap'
alias gcloudcmd='gcloud compute ssh jupyter@wb-gpu-a1ultra \
  --project=adsp-s26-autoanon --zone=us-central1-c --tunnel-through-iap --'

# wb-gpu-a1ultra2g (2x A100)
alias gcloudssh2g='gcloud compute ssh jupyter@wb-gpu-a1ultra2g \
  --project=adsp-s26-autoanon --zone=us-central1-c --tunnel-through-iap'
alias gcloudcmd2g='gcloud compute ssh jupyter@wb-gpu-a1ultra2g \
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

**For `wb-gpu-a1ultra2g`** (the 2x A100 box — see "Remote instances" above),
add a second block, same shape, name substituted throughout:

```
Host wb-gpu-a1ultra2g
  HostName wb-gpu-a1ultra2g.us-central1-c.adsp-s26-autoanon
  User jupyter
  IdentityFile ~/.ssh/google_compute_engine
  ProxyCommand gcloud compute start-iap-tunnel wb-gpu-a1ultra2g %p --listen-on-stdin --project=adsp-s26-autoanon --zone=us-central1-c
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

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

### Step 5 — Install `uv` (skip if already present)

Only needed on a genuinely fresh VM — a boot disk that isn't a clone/snapshot
of an already-set-up box. This step existed on `wb-gpu-a1ultra` from some
earlier, undocumented one-time setup; it was only discovered missing when
`wb-gpu-a1ultra2g` (a fresh instance, not a clone) hit `-bash: uv: command
not found` on every `uv ...` command below. Check first:

```bash
ls -la ~/.local/bin/uv
```

If that exists, `uv` is installed — just make sure it's on `PATH` (`source
~/.bashrc`, or open a new shell). If it doesn't exist:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### Step 6 — Install dependencies

Prefer `make install` (also installs `moshi` and applies the post-install patches it needs
— see the Makefile). If running manually:

```bash
cd /home/jupyter/moshirag-evals
uv sync --all-extras
```

Only re-run after changes to `pyproject.toml`. For code-only changes, `make sync` is sufficient.

### Step 7 — Set up `.env`

`.env` holds secrets (see "Environment and Secrets" in specs/
moshirag-evals-requirements.md) and is gitignored *and* excluded from
`make sync`'s rsync — a fresh VM never receives one from your local machine
and needs its own, created from the committed `.env.example` template:

```bash
cd /home/jupyter/moshirag-evals
cp .env.example .env
# then edit .env and fill in real values for GEMINI_API_KEY, GCS_BUCKET, GCP_PROJECT
```

Required before Step 8 below (`resolve_checkpoint()` interpolates
`GCS_BUCKET` into the checkpoint's `gs://` URI) and before running any
eval/demo (`GEMINI_API_KEY` is used for every retrieval/judge/TTS call).

### Step 8 — Download model checkpoint (inside tmux — survives disconnects)

```bash
tmux new-session -s setup
cd /home/jupyter/moshirag-evals
uv run --all-extras python -c "from core.checkpoint import resolve_checkpoint; resolve_checkpoint('base')"
# Ctrl-b d to detach;  tmux attach -t setup  to reattach
```

---

## Running the demo

**Architecture note (see specs/moshirag-evals-requirements.md's
"Architecture: Target moshi-rag's Production Server" section):** the demo
no longer runs through `MoshiRAGAdapter`/a
custom FastAPI+WebSocket server. It launches kyutai-labs/moshi-rag's own,
unmodified `moshi.server` + `moshi.server_conditioner` directly via
`scripts/run_demo.sh` — the same production stack a now-retired local
ground-truth harness (`prodcheck/`, gitignored, never part of this repo's
history) was built to A/B against, now promoted to the real thing.
`prodcheck/` served its purpose once the pivot's findings were confirmed and
implemented, and was deleted — its evidence lives on in
`docs/demo-development-history.md`'s Phase 0 notes, not in a directory to
go re-run. Reversibility is via git history
(the prior in-process implementation is recoverable from before this change,
same as this version is itself a revival of the pre-`5b3790e` launcher).
`demo/server.py` (the old in-process adapter's FastAPI server) is deleted,
and nothing here loads `MoshiRAGAdapter` in this path. `demo/client/` has
since returned in a different form: not the old in-process adapter's
client, but a maintained fork of moshi-rag's own client used for live
instrumentation UX, alongside a new `scripts/instrumented_server.py` — a
narrow, logging-only wrapper around `moshi.server`, not a reimplementation
of it. See specs/moshirag-evals-requirements.md's "Demo" section for
details.

Pivot status: Phase 0 (root-cause investigation), Phase 1 (this demo pivot),
and Phase 2 (making separate-process conditioning mandatory for
`respond()`/evals) are all done.

### Current state and known issues

The detailed investigation narrative that used to live inline here (every
SUPERSEDED/CORRECTED/RESOLVED/DELETED entry from the demo pivot and the
turn-onset/conditioning-latency/CUDA-graph bug hunts, including anything
describing the now-gone `respond_stream()`) has moved to
`docs/demo-development-history.md` — read that for the *why* behind any of
this. `docs/demo-turn-onset-regression.md` and
`docs/demo-turn-onset-fix-plan.md` are the clean final write-ups of the
turn-onset regression specifically (root causes, fixes, VM validation).
What follows here is just the current, still-actionable state.

**Open, unresolved**: reference-encoder `ConnectError` — the conditioner
process can rarely die mid-session (cause not pinned down), which kills the
whole WebSocket connection unhandled (no exception handling around that
call in moshi-rag's `channel.py`). Not blocking the demo (a human just
reconnects), but relevant to unattended eval runs. `scripts/run_demo.sh
--conditioner-only` tees the conditioner's own output to
`demo/sessions/<session_id>/conditioner.log` — check that first if a run
starts throwing `ConnectError`s, then restart the conditioner.

**Still-live code these investigations produced, not just history**:
- Separate-process conditioning (`server_conditioner`, `REFERENCE_ENCODER_URL`)
  is mandatory for `respond()`/evals, not optional — see "Required setup:
  `server_conditioner` process" under "Running evals" below.
- `core/model_interface.py`'s `_TimedInferenceJob._patch_output_loop()` and
  `_step_watchdog` are still applied to every `respond()` call (fixes a
  `_doing_retrieval` step-loop stall that's specific to `InferenceJob`,
  confirmed absent from the real `moshi.server`/`Channel` path the demo
  uses — so the demo doesn't need this patch, `respond()`/evals still do).
- The off-thread STT + Option E dual-GPU chain (`STT_OFF_THREAD`,
  `_patch_stt_second_gpu()`, etc.) is fully **deleted**, not just disabled
  — STT runs fully synchronous, single-thread now. The only survivor is
  `core/model_interface.py`'s `_warm_up_stt_exec_mask()`, kept purely for
  latency hygiene (see its own docstring). Don't resurrect this chain
  without re-reading `docs/demo-development-history.md`'s CUDA-graph
  thread-safety section first — it's the map of what it hit last time.
- `core/gpu.py`'s auto-detected device pinning remains in the codebase as
  reusable groundwork, even though nothing currently depends on a second
  GPU for the demo path itself — see "Remote instances" above for current
  VM guidance.

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

### Summarizing a demo session's log

Every demo session writes `turns.jsonl` + `raw_events.jsonl` under
`demo/sessions/<session_id>/` (see specs/moshirag-evals-requirements.md's
"Session instrumentation" section). `scripts/summarize_demo_session.py` is
the script built to read those and print a per-turn report — retrieval
breakdown, `ttfat_s`, `<ret>` trigger rate — with all aggregate stats
(mean/p95) computed on the fly, not precomputed anywhere:

```bash
uv run scripts/summarize_demo_session.py demo/sessions/2026-07-13T09-32-11Z/          # console report
uv run scripts/summarize_demo_session.py demo/sessions/2026-07-13T09-32-11Z/ --json   # pretty JSON
```

Full detail on the log format and what the report computes (including why
retrieval fields are reattributed post-hoc rather than trusted from the
live session) is in specs/moshirag-evals-requirements.md's "Demo session
log" and "Demo session summary (console)" sections — this is just a
pointer so the script doesn't go unnoticed next to the run-the-demo steps.

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

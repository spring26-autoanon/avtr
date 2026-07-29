# CLAUDE.md — MoshiRAG Eval Pipeline

## Important

The authoritative requirements for this project are in specs/moshirag-evals-requirements.md. 
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

## Remote instances

Two GCP Vertex AI VMs, same project and zone. See specs/
moshirag-evals-requirements.md's "GPU Sizing and Multi-GPU Deployment"
section for the full history of why both exist — it's had two, unrelated
reasons at two different times, not one continuous story:

- **Original reason (superseded)**: `wb-gpu-a1ultra2g` was provisioned to
  resolve a suspected conditioner-vs-front-end GPU contention finding.
  That was confirmed wrong (see this file's "SUPERSEDED: GPU contention
  conclusion was wrong" section) — the real cause was event-loop
  starvation, unrelated to GPU topology. `wb-gpu-a1ultra2g` was stopped
  (not deleted) once this was confirmed, and single-GPU `wb-gpu-a1ultra`
  was declared sufficient for all further work.
- **Current reason (active as of this writing)**: a completely separate
  investigation — STT's own in-process model duplicate sharing a CUDA
  context/stream with the front-end's main model via `asyncio.to_thread`
  — found a real, independent reason dual-GPU matters again: moshi-rag's
  own CUDA-graph-capture machinery isn't safe to call from two OS threads
  concurrently, in a way that repeated attempts at locking specific call
  sites couldn't fully close. The fix (Option E — see
  `core/model_interface.py`'s `_patch_stt_second_gpu()` docstring for the
  full reasoning, and this file's "STT/front-end CUDA-graph cross-thread
  crash" section below for the investigation) pins STT's own model
  duplicate to a second physical GPU. **`wb-gpu-a1ultra2g` needs to be
  running again for this fix to take effect** — implemented in code as of
  this writing, not yet VM-validated (see that section for current
  status). On single-GPU `wb-gpu-a1ultra`, this specific fix degrades to
  a no-op (STT falls back to sharing the front-end's device, same as
  before Option E) — `_patch_cuda_graph_thread_local()` is the narrower,
  still-applicable single-GPU mitigation for part of the same hazard.

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
implemented, and was deleted — its evidence lives on in this file's Phase 0
notes below, not in a directory to go re-run. Reversibility is via git history
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
`respond()`/evals — see the "RESOLVED" section below) are all done. The
sections below still carry real diagnostic history from before and during
the pivot — kept because the investigation techniques and confirmed findings
remain useful, trimmed of narrative that's now purely dead-code reference.
Where something describes `respond_stream()` specifically: that method and
the demo it served no longer exist — those mentions are historical context
for *why* a given bug was scoped the way it was, not live behavior.

### Known issue (rare, unconfirmed root cause): reference-encoder `ConnectError` kills the session

Observed once across 4 real-checkpoint sessions run against the pivoted
`scripts/run_demo.sh` stack (2 via the now-retired `prodcheck/` harness, 2
via the demo itself): a
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

**Update (2026-07-24, `respond()`/eval path, not the demo)**: hit this same
`ConnectError` far more frequently — 5 of 5 real retrieval-triggering
`respond()` calls in one stretch — while A/B-testing a
`gemini-3.5-flash-lite` retrieval backend (`configs/retrieval_backends.yaml`'s
`gemini_api_flash_lite` entry) against a conditioner that had been running
continuously for a long session. Cross-checking the retrieved reference
text against the model's spoken answer confirmed real damage, not just a
logged error: conditioning silently failed to apply, so the model
hallucinated ungrounded answers (e.g. correctly-retrieved "Aristotle
Onassis" spoken back as "Arthur Schlesinger") while every other signal
(`rag_triggered=True`, a populated reference text) looked like a normal,
successful RAG turn — exactly the "unattended eval run, no human to notice"
risk this section already flagged, now confirmed to actually happen, not
just a theoretical concern.

Initial hypothesis (faster retrieval races the conditioner, landing the
`/embed` call in a bad window given both processes share the VM's single
A100 — unlike the paper's own MoshiRAG setup, which per Table 1's footnote
12 always runs the front-end model and the local retrieval backend on
*separate* GPUs) was **tested directly and did not hold up**: checking the
conditioner's own state found the tmux *server* itself gone entirely (not
just the session) — the conditioner process had died at some earlier,
unknown point in that long session. A fresh conditioner ran `gemini_api`
*and* `gemini_api_flash_lite` back to back with zero `ConnectError`s and
correctly-grounded answers on both. **Best current read, updated: the
conditioner can degrade/die over a long-running session for reasons still
unknown (not yet caught in the act with a real crash log), independent of
which retrieval backend is configured** — the apparent correlation with
`gemini_api_flash_lite` was very likely coincidental timing (it happened to
be under test when the conditioner died), not a causal property of that
backend's speed. `gemini_api_flash_lite` is otherwise validated safe and
meaningfully faster — see its own entry in
`configs/retrieval_backends.yaml`.

Practical fix landed from this: `scripts/run_demo.sh --conditioner-only`
previously ran the conditioner with **no output redirection at all** — its
only record was tmux's own scrollback, which is exactly what vanished when
the tmux server died this round, destroying any chance of finding a crash
traceback after the fact. It now tees to
`demo/sessions/<session_id>/conditioner.log`, matching the full
(non-`--conditioner-only`) path's existing behavior — the next occurrence
should leave real evidence to root-cause from, rather than needing to be
caught live. If a long run starts throwing reference-encoder
`ConnectError`s, check that log first, then restart the conditioner.

### SUPERSEDED: real ARC-encoder conditioning latency (~1.5-2s) is GPU contention with the front-end, not compute cost or IPC overhead

**Correction (2026-07-26): this section's "Confirmed conclusion" and everything
built on it (MIG ruling, `a2-ultragpu-2g` provisioning) turned out to be
diagnosing the wrong mechanism — see "SUPERSEDED: GPU contention conclusion
was wrong" immediately after this section for the real root cause, found by
actually testing on real dual-GPU hardware.** Kept below verbatim, not
deleted or rewritten in place, since it was a real, carefully-reasoned
conclusion from the evidence available on a single-GPU box at the time —
the single-GPU evidence genuinely can't distinguish "conditioner compute
contending with the front-end for the same GPU" from "front-end's own
step loop starving its own concurrent I/O," since both look identical when
there's only one GPU to look at. Only got disentangled once a second,
genuinely separate GPU existed to test against.

Separate from the `ConnectError` investigation above: a live conversation
test (VAD 0.15 + `gemini_api_flash_lite`, see the flash-lite entry in
`configs/retrieval_backends.yaml`) surfaced the model speaking fabricated,
ungrounded content ("Kabul", a fictional-sounding tournament name) on a
question about the history of football, which then got fed back into the
next turn's retrieval context and "confirmed" by the retrieval LLM as if it
were real — a genuine hallucination-feedback loop, not independent noise.
Root cause, found directly in the session's own conditioner/server logs:
the model started speaking its answer within ~40ms of the reference text
coming back, but the ARC-encoder's `/embed` call (which turns that
reference text into the condition tensor actually applied via
`update_streaming_sum_tensors`) took ~2.2s to complete — so the model's
opening tokens were generated before real grounding had landed at all. This
is expected, by-design MoshiRAG behavior, confirmed against the paper's own
text (`arxiv.org/html/2604.12928v1`, fetched directly via `curl` after
`WebFetch` gave two contradictory quotes for footnote 12 across two calls
and couldn't be trusted): "While retrieval is in progress, the front-end
Moshi continues to operate in full duplex... so the conversation proceeds
without interruption" (§3.2) — generation is never gated on conditioning.
The system's correctness depends entirely on the *whole* pipeline (text
retrieval + ARC-encoding + conditioning applied) finishing inside the
paper's own stated budget: "the entire retrieval process completes within
two seconds" (§3.1), with "a sharp decline in accuracy when retrieval
latency exceeds 1.5 seconds" observed empirically in their own data.

This exposed a real, independent bug in our own eval code:
`latency.retrieval_breakdown`'s `context_injection_s` measured a redundant,
no-op `format_context()` call (pure string formatting, no I/O) instead of
the actual `/embed` round trip, so it always silently reported `~0.00s`
while the real ARC-encoder call was taking ~2s. Fixed: `core/
model_interface.py`'s `_TimedInferenceJob` now times `job.
_async_update_reference()` directly (new `conditioning_latency_s` field,
exposed in `respond()`'s metadata dict), and `retrieval_breakdown.py` reads
that real value instead. `total_s` now reflects genuine end-to-end
grounding latency for the first time.

**Isolating cause of the ~2s `/embed` latency itself** (GPU contention vs.
inherent compute cost of the ARC-Encoder, which is `Llama-3.2-3B-Instruct`-
based, not a lightweight embedder, vs. HTTP/serialization overhead):
`scripts/gpu_diag_solo.sh` + `scripts/gpu_diag_contended.sh` ran the same
conditioner process through two conditions — solo (no front-end model
loaded at all) vs. contended (front-end actively generating via a real
`evals/runner.py --mode smoke` run) — while polling `nvidia-smi` at ~100ms
resolution throughout both.

- **Solo** (nothing else on the GPU): steady-state `/embed` calls took
  **~42ms** (first call ~410ms, CUDA warm-up). GPU utilization sat at ~0%
  between brief spikes during the calls themselves. This rules out
  "the 3B-param encoder is just inherently slow" and "HTTP/serialization
  overhead accounts for the ~2s" — both would show up here too, and don't.
- **Contended** (real eval, front-end generating against the same
  conditioner process): `context_injection_mean_s` = **1.58s** (p95 1.68s)
  — a **~37x slowdown** from the solo baseline, with nothing else changed.
  GPU utilization sat at **66-76% for roughly half of all samples**,
  sustained — the front-end's own generation compute, even at
  `MoshiRAGAdapter`'s hardcoded `batch_size=1` (see `core/
  model_interface.py`'s `_load_models()`), already saturates most of the
  VM's single A100 during active generation, leaving the encoder's forward
  pass to queue/time-slice for whatever's left.

**Confirmed conclusion**: this is GPU contention on a shared single-A100
VM, not inherent ARC-Encoder cost, not IPC overhead. moshi-rag's own
separate-process conditioner design (`server_conditioner.py`) is working
as intended — it's just not getting the isolation it's designed to
provide, which assumes either a genuinely idle GPU alongside the front-end
or the encoder's own dedicated second GPU (per the upstream README: "front-
end needs a GPU with a significant amount of memory (24GB)... the reference
encoder can run on a second GPU, or on the same GPU as Moshi if you have
enough VRAM... ideally on the same machine [if separate], to reduce
networking issues"). Note this README guidance is about deployment
flexibility/VRAM headroom, not a latency optimization — it doesn't
contradict this finding, it explains why our single-GPU deployment doesn't
get the benefit the split is designed for. Separately: the paper's own
footnote 12 ("the front-end models run on one GPU, and the local retrieval
back end runs on another") is about the *retrieval back end* (the LLM
generating reference text, e.g. their local Gemma) — a different component
from the ARC-Encoder/reference-encoder. The paper never states where the
ARC-Encoder ran during their own benchmarks; this remains genuinely
unspecified, not inferred either way.

**Real fix requires either a second, genuinely separate GPU for the
conditioner, or MIG (Multi-Instance GPU) partitioning of the current single
A100** — nothing fixable in this codebase alone. Neither has been tried yet;
CUDA MPS was considered and rejected as a candidate fix on reasoning alone
(it reduces context-switch overhead between processes sharing a GPU, but
doesn't add compute capacity — the front-end's own 66-76% utilization during
generation leaves too little headroom for MPS to meaningfully help).

MIG is the untried, no-new-hardware option: the current VM's single A100 is
the 80GB variant (see "Instance sizing" below). Whether it has enough total
VRAM to carve a genuinely isolated partition for each component depends on
the front-end's real footprint, which turned out to be meaningfully larger
than moshi-rag's stated 24GB minimum once actually measured — see the
correction below — so check real MIG profile sizes against that measured
number before assuming this fits, rather than against the vendor's
minimum-case figure.

**Instance sizing, if a second physical GPU is what gets chosen**: `wb-gpu-
a1ultra` is a single A100 **80GB** (`a2-ultragpu-1g` — confirmed by the
~66-69GB combined figure below, which wouldn't fit on a 40GB card at all).
Real GCP specs (`cloud.google.com/compute/docs/gpus`, fetched directly,
not trusted from `WebFetch` synthesis after two unrelated contradictions
this same investigation):

| Machine type      | vCPU | Instance memory | GPUs         | GPU memory (total) |
|--------------------|------|------------------|--------------|---------------------|
| `a2-highgpu-2g`    | 24   | 170 GB           | 2x A100 40GB | 80 GB                |
| `a2-ultragpu-2g`   | 24   | 340 GB           | 2x A100 80GB | 160 GB               |

The ~66-69GB combined figure (front-end + conditioner, one GPU, recorded
under Phase 1) reflects `moshi.server`'s own CLI default of `batch_size=16`
(16 reserved concurrent-conversation KV-cache slots) — `run_demo.sh` never
overrides it. The eval path is not affected by that specific number:
`core/model_interface.py`'s `MoshiRAGAdapter._load_models()` hardcodes
`batch_size=1` unconditionally (line ~903), and the GPU-contention finding
above was measured entirely on that path.

**Measured directly** (`scripts/gpu_diag_per_process.sh`, run against a real
smoke eval): per-process `nvidia-smi --query-compute-apps` turned out to be
blind to the eval's own process for its entire run — every one of 1615
polled samples showed only the conditioner's PID, never the front-end's,
despite the eval completing normally with real results. Cause not pinned
down (not a permissions issue — same user owns both processes; not MIG,
confirmed disabled). Worked around it with the simpler, already-collected
**total**-memory data instead (`scripts/gpu_diag_contended.sh`'s
`memory.used` column, `--query-gpu` rather than `--query-compute-apps`,
which does not have this blind spot): conditioner-alone baseline is a
steady ~23.7GB (`solo_poll.csv`); front-end+conditioner together
(`contended_poll.csv`) ranged from ~46.5GB up to a peak of **71.5GB**,
not a flat number — PyTorch's caching allocator visibly grew its reserved
pool as different questions with different sequence lengths ran through
generation, without releasing memory back down between them. Front-end's
own incremental footprint (contended total minus conditioner-alone
baseline): **~22.8GB at the low end, ~47.8GB at the peak.**

**This corrects the provisional recommendation above** (originally: "`a2-
highgpu-2g` is very likely sufficient," reasoned from moshi-rag's stated
24GB *minimum*, before any real measurement existed) — keeping that
reasoning trail here rather than silently replacing it, since it was a
real, considered guess that real data has now overturned, not a typo. The
measured peak (~47.8GB) **exceeds `a2-highgpu-2g`'s 40GB-per-GPU ceiling**.
It's also a lower bound, not an upper one: this came from a *smoke* run
(25 questions total); a real sample/full eval run (100+ questions) would
very plausibly push the high-water mark higher still, since nothing in
this codebase calls `torch.cuda.empty_cache()` between questions.
**Confirmed recommendation: `a2-ultragpu-2g` (80GB per GPU) is necessary**,
not `a2-highgpu-2g` — 40GB doesn't leave safe headroom above an already-
observed 47.8GB peak. MIG partitioning of the *current* single 80GB A100
remains untried and worth a look before provisioning new hardware at all
(same conclusion as above, unaffected by this correction) — a 47.8GB
partition plus a small one for the conditioner is a tighter fit inside one
80GB card than originally assumed, but may still be workable; check real
MIG profile sizes against this number before ruling it out.

### SUPERSEDED: GPU contention conclusion was wrong — real bottleneck is moshi-rag's own step loop, not GPU sharing

Once `wb-gpu-a1ultra2g` (2x A100 80GB — see specs/moshirag-evals-requirements.md's
"GPU Sizing and Multi-GPU Deployment") was provisioned and `core/gpu.py`'s
auto-detected device pinning put the front-end on physical GPU 0 and the
conditioner genuinely alone on physical GPU 1 (confirmed via `nvidia-smi`:
conditioner process at its known ~23.7GB baseline on GPU 1, GPU 0 otherwise
idle once the front-end's one-shot eval process exited), a real
`gpu_diag_contended.sh` run against this genuinely-separated setup still
measured `context_injection_mean_s` = **1.686s** (p95 1.857s) — statistically
indistinguishable from the original single-GPU finding's 1.58s/1.68s. If GPU
compute sharing were the real mechanism, true physical separation should
have collapsed this back toward the ~42ms solo baseline. It didn't.

**Root cause, traced through real source, not guessed:**

1. `contended_poll.csv` (GPU utilization polled at ~100ms resolution
   throughout the run) shows GPU 1 (conditioner) at **mean 0.08% utilization,
   max 11%, only 29 of 3046 samples showing any activity at all** — consistent
   with ~17 real `/embed` calls each taking their expected ~42-100ms, matching
   the solo baseline exactly. The conditioner's own compute was never slow;
   it sat idle almost the entire run.
2. `core/model_interface.py`'s `conditioning_latency_s` timer wraps the whole
   of moshi-rag's own `InferenceJob._async_update_reference()`:
   ```python
   async def _async_update_reference(self, reference_text: str) -> None:
       streaming_sum_tensor = await get_conditioning_remote_async(...)
       ...
       self.server.runner.lm_gen.update_streaming_sum_tensors(per_slot)  # not awaited
   ```
   `update_streaming_sum_tensors` (moshi-rag's own `LMGen` method, confirmed
   via `inspect.getsource` on the real VM) does one `t.to(device=device,
   dtype=dtype)` call moving a ~200-600KB tensor onto the front-end's own GPU
   — a real candidate for queuing behind the front-end's own generation
   kernels. Ruled out by direct comparison: the eval result JSON's
   `context_injection_values` (the *outer* timer, including this tensor
   copy) match `get_conditioning_remote_async`'s own internally-logged
   `"[Remote Encoder] Received response in X.XXXs"` (the *inner* timer,
   HTTP call only) to the millisecond (e.g. `1.5117...` vs `1.511s`,
   `1.8782...` vs `1.877s`). The tensor copy adds negligible time — the
   entire delay is inside the awaited HTTP call itself, not the subsequent
   GPU-side tensor application.
3. That HTTP call (`get_conditioning_remote_async`, moshi-rag's own code,
   confirmed via `inspect.getsource`) is a plain `httpx.AsyncClient` POST,
   properly `await`ed, no synchronous blocking calls of its own. Combined
   with finding 1 (server-side genuinely fast and idle), the only place left
   for ~1.5s to hide is **the front-end's own event loop not getting back to
   this coroutine promptly** — not because it isn't correctly async, but
   because something else on the same thread isn't yielding.
4. That something else, found in moshi-rag's own `server.py` (cloned
   directly from `github.com/kyutai-labs/moshi-rag`, not guessed):
   ```python
   async def run_one_step(self) -> bool:
       g = self._gather_step_inputs()
       if g is None:
           return False
       return self.runner.run_step(g, self._deliver_step_row)  # sync call, not awaited

   async def _step_loop(self):
       while True:
           ran = await self.run_one_step()
           if ran:
               await asyncio.sleep(0)
           else:
               await asyncio.sleep(0.005)
   ```
   `run_one_step` is declared `async def` but its body never `await`s
   anything — `self.runner.run_step(...)` runs synchronously to completion,
   blocking the entire single-threaded event loop for however long one
   step takes, with the only yield point being `await asyncio.sleep(0)`
   *between* steps. The same contended run's own log independently confirms
   how long that is: `WARNING batched step (1/1 active) took 95.3ms`,
   repeated throughout. ~1.5s / ~95ms per step ≈ 16 consecutive blocking
   steps — a plausible span for the event loop to go without a genuine free
   moment to service the already-completed httpx socket read.

**Confirmed conclusion (revised)**: the bottleneck is moshi-rag's own
`ServerState._step_loop`/`run_one_step` running the real per-step model
computation as a synchronous, unyielding call inside an `async def`
function — starving concurrent I/O (the conditioning HTTP call) of the
event-loop time it needs to complete promptly, regardless of how many
physical GPUs are involved or which one the conditioner runs on. This is
upstream moshi-rag code, not anything introduced in this repo. **GPU count
is not the relevant variable for this specific latency** — `wb-gpu-a1ultra`
(single A100) is sufficient for all further work on this problem;
`wb-gpu-a1ultra2g` was stopped (not deleted) once this was confirmed. See
specs/moshirag-evals-requirements.md's "GPU Sizing and Multi-GPU Deployment"
section for the corresponding correction to the MIG/dual-GPU
recommendation.

**RESOLVED**: before deciding between patching the step loop itself vs.
moving the conditioning call off it, checked whether `run_one_step`'s
synchronous, unyielding call is a genuine bug or a deliberate design
dependency — confirmed the latter, via real moshi-rag source
(`lm.py`'s `apply_pending_streaming_sum_condition`): the conditioning
tensor is consumed **one row per real-time step**
(`state.pending_streaming_sums[b]`, drained one entry per call), in
lockstep with Mimi/Moshi's fixed 12.5Hz frame rate — `run_step`'s own
`elapsed_ms >= 77` warning threshold confirms a hard real-time budget per
step. Patching `run_one_step`/`_step_loop` itself (splitting it across
threads, inserting mid-step yields) would risk a real race on
`pending_streaming_sums` and would fight that real-time constraint — not
attempted, and moshi-rag's `InferenceJob`/`ServerState`/`Channel`/
`BatchRunner` step-loop logic remains completely untouched, per this
project's own rule against reimplementing it.

Fix implemented instead: `core/model_interface.py`'s
`_fetch_and_apply_reference_conditioning()` moves only the conditioning
HTTP fetch (`get_conditioning_remote_async` — pure network I/O, zero
GPU/model-state involvement) off the main event loop, via
`asyncio.to_thread` running it to completion on its own throwaway event
loop in a worker thread. `update_streaming_sum_tensors` (the one piece
that touches live model state, confirmed cheap at ~1ms) stays on the
calling thread, unchanged from upstream. Applied in **two places**, since
this bug affects the live demo identically to respond()/evals (both share
the same `ServerState._step_loop`, confirmed by reading `channel.py`'s own
`Channel._async_update_reference` — the same pattern, same bug): `core/
model_interface.py`'s `_immediate_handle_reference_text` (eval path) and a
new `scripts/instrumented_server.py`'s `_patch_channel_conditioning()`
(demo path, replacing `Channel._async_update_reference` wholesale).

**Verified on real hardware (2026-07-27, `wb-gpu-a1ultra`, eval path)**:
`scripts/gpu_diag_solo.sh` + `scripts/gpu_diag_contended.sh` rerun with the
fix in place (`conditioner_contended: true`, single-GPU box, same as the
original bug report). Solo baseline unchanged at ~41ms steady-state.
Contended `context_injection_mean_s` = **0.196s** (p95 0.199s, n=3 valid
samples out of 5 — 2 excluded by the separately-tracked degenerate-silence
issue), down from the pre-fix 1.58s mean / 1.68s p95 — an ~8x reduction,
from ~37x the solo baseline to ~4.7x it. Confirms the event-loop-starvation
diagnosis and the fix.

**Demo path separately verified (2026-07-27, same VM, real live session, 4
turns / 3 real `<ret>` triggers, `gemini_api`/`gemini-3.5-flash` backend)**:
confirmed active (the `[Reference] ARC encoding received in %.3fs` log line
only exists inside the new shared helper), and real — **1.126s, 0.976s,
0.700s** across the three triggers, vs. the eval path's clean 0.196s.

**Root cause of the 4-6x gap, corrected after actually reading the
per-trigger log lines closely (the earlier "GIL contention from live STT/
VAD" explanation below was a plausible-sounding guess, not checked against
the demo's own log, and turned out to be off-target):**
`get_conditioning_remote_async` logs its own internal
`[Remote Encoder] Received response in Xs` timing *inside* the worker
thread, separately from `_fetch_and_apply_reference_conditioning`'s outer
`[Reference] ARC encoding received in Xs` (logged after `await
asyncio.to_thread(...)` returns on the main thread). Diffing these two
per trigger:

| Trigger | Inner (raw HTTP fetch) | Outer (main thread resumes) | Gap |
|---|---|---|---|
| 1 | 0.505s | 1.126s | 0.621s |
| 2 | 0.116s | 0.976s | 0.860s |
| 3 | 0.104s | 0.700s | 0.596s |

**The raw fetch itself is fast — often faster than the eval path's own
~0.1-0.13s inner fetches, confirming `asyncio.to_thread` is doing exactly
what it's supposed to.** The entire 4-6x gap lives in the **handoff after
the worker thread is already done**: `asyncio.to_thread` wraps
`loop.run_in_executor`, whose completion is delivered back to the awaiting
coroutine via `loop.call_soon_threadsafe` — a callback queued on the
*main* event loop, which still can't run until that loop is free. This is
the same category of bug the fix was written to solve, just moved one
layer down: the fix eliminated the multi-round-trip HTTP exchange sharing
the loop with `run_step()`, but the single "wake up and deliver the
result" handoff is still exposed to the same congestion.

**This is present in the eval path too, at much smaller scale** — checked
`gpu_diag/contended_eval.log` the same way: inner/outer gaps there run
~30-100ms (e.g. 0.132s→0.199s, 0.101s→0.186s), roughly one `run_step()`
step's worth (eval's own log shows frequent `batched step (1/1 active)
took ~95ms` warnings). So it's the same mechanism at both scales, not a
demo-only bug — the demo's gap is just far larger.

**What doesn't explain the size of the demo's gap**: `moshi.server`'s
default `batch_size=16` (vs. eval's hardcoded 1) was the obvious first
suspect — ruled out, `batched step (1/16 active)` timings (81.8-88.5ms)
are the same order of magnitude as eval's.

**Root cause, precisely identified (2026-07-27, `scripts/instrumented_
server.py`'s `_patch_event_loop_diagnostics()`, `DEMO_ASYNCIO_DEBUG=1` —
see that function's docstring for a real monkeypatching pitfall hit and
fixed along the way: the first version hooked `asyncio.set_event_loop`,
the package-level re-export, not `asyncio.events.set_event_loop`, the
name Python 3.11's real `asyncio.run()` internals actually call through —
it silently never fired on the first VM run despite passing a local
sanity test that (wrongly) validated itself by calling the same patched
name directly)**: with the patch actually firing, asyncio's own debug
logging (20ms threshold) shows the front-end step loop firing
`Executing ... took ~0.050s` continuously, back-to-back, throughout the
whole session — real, but individually under the app's own 77ms warning
threshold, which is why `batched step` warnings never appeared near the
gap windows despite the step loop genuinely running near-continuously.
But lining these up precisely against all three gap windows found the
*actual* dominant blocker isn't the step loop at all: **one
`Channel._recv_loop()` execution per window, 0.58-0.80s each — accounting
for nearly the entire gap by itself** (the step loop's own contribution
in the same windows is ~0.05s × 2, comparatively tiny).

Read `Channel._recv_loop` (`moshi/inference_utils/channel.py:255-324`,
real source) to find why: it calls `await self.stt.send_audio(chunk)` per
audio frame. With `--stt local` (the default, used in every session so
far), that's `LocalSpeechToText.send_audio()`
(`moshi/stt/local_stt.py:108-137`) — declared `async def`, but its
`while self._pending.size >= fs:` loop calls `self.mimi.encode(chunk)`
and `self._run_codes(...)` (a real local Mimi + STT-language-model
forward pass) **fully synchronously, no `await`**. The only yield point
inside that loop, `await self._out_queue.put(word)`, only fires when a
word is actually decoded that frame — most frames don't produce one.
`_recv_loop` calls `send_audio` directly, not as its own task, so when
several audio frames back up in `self._pending` (plausible any time the
step loop's own frequent-but-small blocking has just eaten a slice of
event-loop time), `send_audio` drains the whole backlog as one
uninterrupted synchronous burst — exactly what the log shows. Confirmed
this is STT-mode-specific, not inherent to `_recv_loop`: the remote-STT
alternative, `GradiumSpeechToText.send_audio()`
(`moshi/stt/gradium_stt.py:39-46`), is just `await
self.audio_queue.put(audio)` — genuinely fast, no local inference, the
real model work happens in a separate background task talking to a
remote service.

This also explains why eval's own gap is so much smaller: `InferenceJob`'s
feed loop **also** calls `LocalSpeechToText.send_audio()`
(`inference_job.py:240/258`) — `core/model_interface.py:1006/1052`
constructs its own `LocalSpeechToText(deepcopy(self._mimi))` too, so this
was never demo-only. But eval feeds audio self-paced, synchronously, from
the same coroutine that consumes it — no independent network-paced
producer that can race ahead of consumption the way a live WebSocket's
`_recv_loop` can, so no backlog forms; eval's small ~30-100ms gap is fully
explained by the front-end step loop alone. The demo has that mechanism
*plus* this STT one layered on top from real backlog accumulation, and
the STT one dominates.

**Correctness re-assessed and found safer than first flagged**: the
original note here said moving `LocalSpeechToText` inference off-thread
would carry the same risk category as `update_streaming_sum_tensors`
(touches live, shared model state). Checked directly against real
source and found this wrong — `server.py:93`'s `self.mimi_copy =
deepcopy(mimi)` at server startup, plus `server.py:236`'s
`Channel(self, ws, mimi=deepcopy(self.mimi_copy))` per connection (and
this module's own matching `deepcopy(self._mimi)`/
`deepcopy(self._stt_template)` per call), mean `LocalSpeechToText`'s
`mimi`/`_lm_gen` are always genuinely separate Python objects from the
front-end's own live model — nothing shared to race on. Remaining known,
accepted risk: `_run_codes`' inline `self.vad_callback(...)` call (mutates
`TurnManager.vad_history`, a plain list) now fires from the worker thread
instead of the main one — not a crash risk (GIL-atomic list ops, no other
concurrent writer), but VAD updates land at a slightly different cadence
than today. Left as-is rather than reimplementing `_run_codes` to defer
it back to the main thread — more invasive than the risk it guards
against, no observed regression to justify it. Watch for turn-taking/VAD
regressions during VM validation.

**Fix implemented (2026-07-27), not yet VM-verified**:
`core/model_interface.py`'s `_run_stt_frames_sync()` +
`_patched_stt_send_audio()` + `_patch_local_stt_off_thread()` — same
`asyncio.to_thread` shape as the conditioning fix, moving only the
per-frame Mimi-encode + STT-LM compute off the main thread;
buffering/locking/queue-puts stay on the calling thread unchanged.
Applied via class-level monkeypatch (matches the `Channel.
_async_update_reference` precedent) at both construction sites:
`_load_models()` (eval) and `scripts/instrumented_server.py`'s
`apply_patches()` (demo) — one shared implementation, not two that could
drift. Covered by local tests (no GPU needed — `torch` faked via
`sys.modules` injection for `_run_stt_frames_sync`'s own tests, matching
the existing `get_conditioning_remote_async` test precedent):
patch-mechanics/ordering, lock-serialization under concurrent calls, and
the actual mechanism (`test_patched_stt_send_audio_does_not_block_
concurrent_coroutine` — a fake compute doing a real blocking
`time.sleep`, confirming a concurrent lightweight coroutine keeps
ticking on schedule during the call). All pass; full suite (349 tests)
green.

`core/model_interface.py`'s new `_maybe_enable_eval_asyncio_debug()`
(`EVAL_ASYNCIO_DEBUG=1`, see `scripts/gpu_diag_contended.sh`'s header) is
the eval-path counterpart of the demo's `DEMO_ASYNCIO_DEBUG` diagnostic —
simpler to wire than the demo's since `_ensure_step_loop()` builds its
loop directly via `asyncio.new_event_loop()` and never passes it through
`asyncio.set_event_loop()`, so no monkeypatching pitfall to hit here, just
configure the loop object directly. Since eval also drives
`LocalSpeechToText`, this gives a fully scriptable (no live conversation)
VM check of the fix — won't reproduce the demo's full backlog magnitude
(eval's self-paced feeding doesn't create one), but confirms the fix
doesn't crash/misbehave on real hardware before spending a live demo
round trip on the magnitude question. **Not yet run on the VM** — next
session's work: `EVAL_ASYNCIO_DEBUG=1 bash scripts/gpu_diag_contended.sh`
first (fully scripted), then a live demo round trip with
`DEMO_ASYNCIO_DEBUG=1` reusing the same diagnostic already proven to work,
to confirm the `Channel._recv_loop` gap collapses toward eval's baseline.

Not fully collapsed to the ~41ms solo baseline. Initial guess (untested,
now corrected below) was a fixed per-call setup cost from `asyncio.to_thread`
spinning up a new thread and a fresh throwaway event loop every call.

**That guess was wrong, disproven by direct local benchmark** (no GPU/VM
needed — `_run_in_new_loop`'s exact pattern, real `httpx.AsyncClient`,
against a local instant-response HTTP server): a fresh event loop + fresh
`httpx.AsyncClient` per call, with nothing else running, costs only
**~4.3ms mean / 5.6ms p95** above a ~2ms persistent-client baseline — two
orders of magnitude too small to explain a ~150ms residual. (Also
confirmed by reading the real moshi-rag source directly,
`moshi/inference_utils/utils.py`'s `get_conditioning_remote_async`: it
opens a fresh `httpx.AsyncClient` per call on top of `_run_in_new_loop`'s
fresh event loop — two layers of "cold setup," both still cheap.)

**Real explanation, confirmed by the same benchmark**: `asyncio.to_thread`
moves the fetch off the *event loop*, fixing the original starvation bug,
but it still runs on a real Python (GIL-bound) thread. moshi-rag's
`run_one_step()` is still synchronous and GIL-holding for its real ~95ms
logged per-step duration — so the worker thread's own event-loop
iterations (parsing the HTTP response, running httpx's code) still queue
for GIL time behind the main thread, same root mechanism as the original
bug, one layer down. Reproducing this synthetically (same fresh-loop/
fresh-client call, plus a concurrent main-thread loop doing ~95ms
GIL-holding bursts with `sleep(0)` yields between them, mirroring the real
step cadence) reproduced the same order of magnitude: **mean 64.7ms, p95
93.2ms** — a ~15x jump from the no-contention case, consistent with the
real 150-196ms residual (not an exact match — real step counts/timing
vary per question, and the real payload is a much larger safetensors
tensor, not a tiny JSON blob, so more GIL-holding parse time on the worker
side is expected too).

**Implication for any future optimization**: reusing a persistent
worker thread/event loop (the original untested guess's proposed fix)
would only save the ~2-4ms measured for Candidate A — not worth pursuing.
The real lever, if sub-100ms conditioning latency is ever needed, is a
genuinely separate *process* for the fetch (not a thread) — GIL contention
is thread-specific, so a subprocess would be immune to it. Not pursued now
— 196ms is already comfortably inside the paper's 1.5s degradation
threshold and 2s total retrieval budget (see the "real ARC-encoder
conditioning latency" section above). Benchmark script (not checked in,
scratch/local only): `bench_conditioning_overhead.py`, reproducible
without a VM or GPU — pure Python + `httpx` against a local
`http.server`.

`core/gpu.py`'s auto-detected device pinning stays in the codebase — it's
not wrong, just not the fix for this — and remains useful groundwork if a
real reason to use multiple GPUs ever comes up again. (Update: one did —
see this file's "STT/front-end CUDA-graph cross-thread crash" section
below, where this exact groundwork gets extended and put to use for an
entirely unrelated reason.)

**TODO, not yet fixed**: the demo's own `session.last_context_injection_s`
(feeds the live session log's `retrieval_breakdown_s.context_injection_s`)
still comes from the older, separately-known-imprecise `format_context()`
timing in `scripts/instrumented_server.py`'s
`_patch_rag_manager_get_reference_text` — not the real conditioning
latency this section is about. Confirmed why this isn't a trivial
follow-up fix: `Channel._handle_reference_text` schedules
`_async_update_reference` as a **fire-and-forget background task**
(`self._task_group.create_task(...)`, not awaited), so by the time
`RAGManager._background_task` reads `last_context_injection_s` to report
`retrieval_breakdown_s`, the real conditioning fetch this section fixed
hasn't necessarily completed yet — the real timing would need to be
reported asynchronously, after the fact, once that background task
actually finishes, which doesn't fit the current single synchronous
reporting point in `_background_task`. Left alone for now; the eval
path's own `context_injection_s` (via `latency.retrieval_breakdown`) is
unaffected by this and already reports the real value correctly.

### STT/front-end CUDA-graph cross-thread crash: three real crashes, root-caused to moshi-rag's own CUDA-graphing machinery not being thread-safe, fixed via physical GPU separation (Option E)

The STT-off-thread fix above (`_run_stt_frames_sync()` +
`_patched_stt_send_audio()` + `_patch_local_stt_off_thread()`) was
"not yet VM-verified" as of the last section. Running that verification
(`EVAL_ASYNCIO_DEBUG=1 bash scripts/gpu_diag_contended.sh`) surfaced three
distinct, real crashes in sequence — each only reachable once the prior
one was fixed, since moshi's GPU-touching code turned out not to be safe
to call from two threads concurrently in more than one way at once. Full
technical detail for all of this lives in the code itself now (extensively
commented, not just fixed) — `core/model_interface.py`'s
`_run_stt_frames_sync()`, `_patch_compiled_functions_thread_safe()`,
`_patch_stt_no_cuda_graph()`, `_patch_stt_second_gpu()`,
`_patch_cuda_graph_thread_local()`, and `_warm_up_stt_exec_mask()`
docstrings, and `_COMPILED_FUNCTIONS_LOCK`'s own module-level docstring —
this section is the narrative summary, not a duplicate of that detail.

**Crash 1 — dynamo lazy-compile race**: `torch_compile_lazy`
(moshi/utils/compile.py) has a bare, unlocked `nonlocal` lazy-compile
cache per decorated function (`apply_rope`, `_rms_norm`,
`gating_forward_kernel`), shared at the module level regardless of
deepcopy. `RuntimeError("Detected that you are using FX to symbolically
trace a dynamo-optimized function...")`. Fixed by
`_patch_compiled_functions_thread_safe()` — a narrow lock around exactly
these three leaf functions.

**Crash 2 — CUDA graph capture race**: Mimi's encode/decode and the LM's
own forward pass are wrapped in moshi's own `CUDAGraphed`, which needs
true stream-level exclusivity for its entire capture window.
`AcceleratorError("CUDA error: operation not permitted when stream is
capturing")`. A first attempt (one coarse lock around the whole of
`BatchRunner.run_step()` vs. the whole of one STT frame's compute) *did*
eliminate the crash — but a live demo test caught what eval-path testing
couldn't: the front-end's own generation went near-silent for seconds at
a time during real retrieval waits, because the demo's continuous live
audio keeps the STT thread almost always active, unlike eval's
self-paced feeding. Rejected. Fixed instead by `_patch_stt_no_cuda_graph()`
— disabling CUDA graphing for STT's own model instances entirely
(`LMGen(profile=True)`, `mimi.set_profile(True)`), so there's no capture
to race in the first place, rather than locking around it.

**Crash 3 — `set_exec_mask`'s own capture, and the real, deeper root
cause**: `set_exec_mask`'s own `CUDAGraphed` isn't gated by `self.profile`
at all — the one CUDAGraphed construction site in the whole codebase that
isn't, confirmed via an exhaustive grep. Same `AcceleratorError`, on a
turn that had run clean for 5-6 turns prior. Three fixes attempted here,
each teaching something real:
1. Adding `set_exec_mask` to the existing narrow lock **deadlocked**
   immediately — `LMGen.set_exec_mask`'s own callback recurses into a
   *nested* `StreamingModule.set_exec_mask` on the *same* thread, and a
   plain `threading.Lock` isn't reentrant. Found via `py-spy dump` on the
   real hung process (installed with permission —
   `uv tool install py-spy`), which showed the exact nested frame twice.
2. Switching to an `RLock` fixed the deadlock but **not the crash** — the
   same error recurred a couple turns later. Root cause, confirmed by
   reading real moshi-rag source: the *main* step-loop thread's own
   encode/decode/step calls are plain graph *replays* by that point
   (captured once during `_state.warmup()`), not calls to any of the
   four locked functions — nothing serialized them against the STT
   thread's own `set_exec_mask` *capture*.
3. Pre-warming (calling `set_exec_mask` once per turn before the
   concurrent window opens, to turn the capture into a safe replay)
   revealed the real, deeper problem on closer reading of
   `moshi/utils/compile.py`: `CUDAGraphed.__call__` wraps *every*
   call — warmup, capture, and ordinary replay alike — in
   `with _set_in_cuda_graph(): assert not _in_cuda_graph; ...`, and
   `_in_cuda_graph` is a bare, **process-wide module global, not a
   `threading.local()`**. Two threads racing a check-then-set on that
   shared flag is a real hazard for *any* `CUDAGraphed` call from either
   thread — not just the handful of call sites this project could
   enumerate and lock one at a time. moshi-rag's own CUDA-graphing
   utility was simply never designed to be called from more than one OS
   thread at once, for anything. (The pre-warming call itself also had a
   real, separate off-by-one bug, since fixed: `CUDAGraphed`'s default
   `warmup_steps=1` means the *first* call only primes that buffer,
   uncaptured — the real capture happens on the *second* call. Calling
   once left the real capture to happen during the live per-frame call
   anyway.)

**Fix — Option E, implemented, not yet VM-validated**: no amount of
locking specific call sites can close a hazard that's baked into
*every* `CUDAGraphed` call in the model, so the fix sidesteps sharing
instead of trying to guard it:
- `_patch_stt_second_gpu()` moves STT's own model duplicate onto a
  genuinely separate physical GPU when one is visible to this process
  (`cuda:1`), so its CUDA stream never overlaps with the front-end's at
  all, regardless of timing. Requires `core/gpu.py`'s
  `ensure_cuda_visible_devices("frontend")` to expose *both* GPUs to the
  front-end process now (`"0,1"`, not just `"0"`) — see that module's
  own `resolve_devices()` docstring. Degrades to a no-op wherever a
  second GPU isn't visible (single-GPU `wb-gpu-a1ultra`, or a manually
  restricted `CUDA_VISIBLE_DEVICES`).
- `_patch_cuda_graph_thread_local()` closes the remaining, far more
  benign residual risk — the shared `_in_cuda_graph` Python global itself
  — by making it `threading.local()`.
- `_warm_up_stt_exec_mask()` is kept on top of both (fixed to call twice,
  per the off-by-one above), now for latency cleanliness rather than
  correctness: with device separation and the thread-local flag both in
  place, a capture during the live window is safe either way, but doing
  it upfront avoids inflating the first timed frame of every turn with a
  real capture's cost instead of a fast replay's.

**Status as of this writing**: implemented in code, covered by local
tests (no GPU needed — every patch mechanism, including a real two-thread
test proving the thread-local flag actually isolates two threads without
losing genuine same-thread reentrancy protection), full suite green.
**Not yet run on the VM.** Needs `wb-gpu-a1ultra2g` (see "Remote
instances" above — stopped, not deleted, now needed again for this
unrelated reason) restarted, then: `make sync`, then
`EVAL_ASYNCIO_DEBUG=1 bash scripts/gpu_diag_contended.sh` for 15+ turns
(the last crash hit on turn 7, so a materially longer run than that is
the actual bar for confidence this time), checking `nvidia-smi` shows
STT's own memory footprint landing on GPU 1 (not GPU 0) alongside
confirming no more `AcceleratorError`s. Once that's clean: the live-demo
generation-fluency check and the `gemini_api_flash_lite` backend test
that were paused for this whole investigation are still outstanding.

**Update (2026-07-27), eval-path VM validation — DONE, confirmed
correct**: `wb-gpu-a1ultra2g` restarted (needed a manual `make install`-
equivalent first — the fresh boot disk had no `moshi`/`transformers`
installed at all, unrelated to this fix, see the VM-environment note
this update's session left in `project_stt_cuda_graph_thread_safety`
memory), then `EVAL_ASYNCIO_DEBUG=1 bash scripts/gpu_diag_contended.sh`
run for 23 real `respond()` calls (`evals/results/run-2026-07-27T23-30-13Z.json`)
— zero `AcceleratorError`s, zero crashes of any kind.
`gpu_diag/contended_poll.csv` confirms device separation: GPU 0
(front-end) peaked at ~17.9GB, GPU 1 (STT) ranged 23.7GB→47.1GB — STT's
memory genuinely landed on the second physical GPU, not shared with the
front-end. This is well past the "15+ turns, longer than the turn-7
crash" bar set above. **The crash is fixed; treat Option E as validated
on the eval path.**

**Live-demo generation-fluency check — done, but the result reframes the
question**: a real live demo session was run post-fix
(`demo/sessions/2026-07-27T23-44-30Z/`) and subjectively did not feel
fluent. Directly measuring it (inter-token gaps from `[Buffered
Model]`/`[Display Model]` timestamps, retrieval `backend_latency=`,
VAD-state timestamps) found **no evidence pointing back at Option E**:
within-generation word cadence (0.309s mean) was close to the pre-fix
baseline (0.243s) and nowhere near the actual known-bad coarse-lock
session's pathology (that session's problem was isolated multi-second
dead-air stalls, not per-word cadence — its own within-generation
cadence measured fine, 0.229s). Retrieval latency (1.83s/3.56s
`backend_latency=`) and conditioning (`ARC encoding received in`
0.454s/0.123s) both measured as expected, not regressed.

What the log data did surface, unprompted: a dead-air gap between the
model finishing a turn (`[Display Model]`'s last token) and VAD detecting
the next user utterance, growing turn-over-turn within the same session
— **9.6s, then 23.8s** — far longer than the actual greeting/answer
content takes to say. Neither gap is step-loop starvation (callbacks
were a steady 55-59ms throughout, no warnings) or STT lag (transcription
tracks live speech within ~100-250ms once audio starts arriving). Given
the growing-over-session shape, the leading hypothesis is the
already-known, separately-tracked client jitter-buffer overrun (see
"Demo audio quality" investigation / `project_demo_audio_quality_investigation`
memory) — the client may still be draining a backlog of the model's own
audio well after the server considers the turn finished, so what looks
like server-side dead air is partly (or entirely) the user still hearing
the previous turn. **Not yet confirmed** — requires client-side
playback-timing evidence, which didn't exist before this session. See
"Client-side audio jitter-buffer diagnostic" below for the logging just
added to get it. Until that's collected, don't read the eval-path fix
above as in question — it's on separate, direct, server-side-only
evidence (a stopwatch around the actual coroutine, plus a crash that
reproduces identically on the client-free eval path) that this perceived
demo-fluency gap doesn't touch.

### Client-side audio jitter-buffer diagnostic (added 2026-07-27, not yet used on a real session)

To test the jitter-buffer hypothesis above, `demo/client/` now records a
timestamped log of the browser's own audio-playback queue depth
(`audio-processor.ts`'s existing `delay` figure — mic-duration minus
played-stream-time — already computed for the live "Latency" stat, just
not persisted over time before this), plus structured buffer-drop/
underrun/resume events, in `useServerAudio.ts`'s new `getAudioDiagLog()`.
Timestamps are `t_rel_s` relative to the client's own WebSocket-open
moment (not epoch time — client and server may be different machines
with unsynced clocks, but socket-open and server-side `Channel`
construction happen within one handshake of each other, close enough to
line up against `server.log`/`raw_events.jsonl`'s own `t_rel_s`).

**To use**: open the demo's audio-stats panel (the same one showing
"Latency"/"Min/Max buffer" today) and click the new "Download audio
diagnostics" button — appears whenever `getAudioDiagLog` is wired in —
at the end of a session. It downloads `client_audio_diag_<timestamp>.json`,
an array of `{t_rel_s, delay, actualAudioPlayed, totalAudioPlayed,
events?}` samples (one per audio frame, ~80ms cadence during active
playback). Save it as `demo/sessions/<session_id>/client_audio.json`
(matching the session directory the server side already writes) before
the next analysis pass — it isn't part of `make sync`'s rsync since it
never leaves the client machine's browser.

**Not yet build/lint-verified** — implemented and manually reviewed line
by line, but no Node.js toolchain was available to run `npm run build`/
`tsc` this session. Run that first, before spending a live session on
it, in case of a typo this review missed.

**Next session's analysis, once a real session's `client_audio.json`
exists**: plot/diff its `delay` values against the server's own
`t_rel_s` VAD/RAG timestamps for the same session. If `delay` is
elevated and climbing through the 9.6s/23.8s dead-air windows identified
above (rather than sitting near the worklet's own ~10-80ms target
buffer range), that confirms client playback backlog as the real
mechanism behind the perceived demo unfluency — independent of, and not
a regression from, Option E.

### Real-time step pacing (added 2026-07-27) and the STT-off-thread A/B toggle

Two follow-ups from re-examining the whole STT-off-thread/Option E arc
against the actual upstream source (a clone in scratch, not just prior
notes), prompted by stepping back to ask whether that entire investment
was earning its keep.

**Root cause of the jitter-buffer overrun, now confirmed directly against
real moshi-rag source (not just the client-side symptom)**:
`batch_runner.py`'s `BatchRunner.run_step()` measures `elapsed_ms` and
only *warns* past ~77ms — there is no complementary sleep anywhere when a
step finishes faster than the real-time budget one step represents
(~80ms at Mimi's 12.5Hz). `_deliver_step_row()` does an uncapped
`output_queue.put_nowait()`; nothing downstream throttles it either. On
hardware fast enough to reliably beat 80ms/step (this project's A100s
routinely log ~50-95ms steps), the server ships audio strictly faster
than real-time, continuously, with no mechanism anywhere to slow back
down — precisely the "genuine average-rate mismatch, not ordinary
jitter" signature the audio-quality investigation already found from the
client side. **Fix**: `scripts/instrumented_server.py`'s new
`_patch_server_state_step_pacing()` wraps (does not reimplement)
`ServerState.run_one_step()` — after a step actually runs, if it
completed faster than `1 / mimi.frame_rate`, sleeps the remainder before
the next step. Demo-only: `run_one_step` has exactly one caller in all
of moshi-rag (`ServerState._step_loop()`, confirmed via grep across the
real source), so this cannot affect `respond()`/evals, which never call
it and want maximum throughput, not real-time pacing, anyway. Gated by
`DEMO_STEP_PACING=0` to disable, for A/B against the pre-pacing
baseline. **Implemented, not yet VM-verified.**

**Independent re-verification (not just re-citing the earlier claim) that
eval never needed STT-off-thread**: read `inference_job.py`'s
`_feed_loop()` and `channel.py`'s `_recv_loop()` directly, side by side.
`_feed_loop()` gates every chunk on `await
self._wait_step_index_at_least(feed_step - max_stream_delay)` —
structurally caps how far its self-paced feeding can get ahead of the
model's own step progress (bounded by the model's small internal codec
delay), so `LocalSpeechToText.send_audio()`'s synchronous per-frame
compute cannot form a backlog there, no matter how long it blocks.
`_recv_loop()` has no equivalent gate anywhere — it's driven by `async
for message in self.ws`, i.e. the live client's own network-paced audio,
completely independent of the model's progress; confirmed independently
in `local_stt.py` that `send_audio()`, despite being `async def`, runs
`mimi.encode()`/`_run_codes()` synchronously with no internal `await`, so
a live backlog can and does queue at the transport layer while one call
blocks. This means the entire 3-crash chain (dynamo race → CUDA-graph
race → `set_exec_mask` race) and Option E's 2-GPU requirement exist
solely to serve a mechanism that's real for the demo but was never
present for eval — eval only inherited the risk because STT-off-thread
was applied there "for consistency," not because it needed it.

**Fix**: `core/model_interface.py`'s new `_stt_off_thread_enabled()`
gates the entire Option E chain (`_patch_local_stt_off_thread()`,
`_patch_compiled_functions_thread_safe()`,
`_patch_cuda_graph_thread_local()`, `_patch_stt_no_cuda_graph()`,
`_patch_stt_second_gpu()`) behind `STT_OFF_THREAD=0` to disable (default
enabled, preserving current behavior) — applied identically in both
`_load_models()` (eval) and `scripts/instrumented_server.py`'s
`apply_patches()` (demo), one shared helper so they can't drift. With it
disabled, STT reverts to fully synchronous, single-thread, no CUDA-graph
hazard at all — single GPU sufficient for both paths. **Implemented, not
yet VM-verified.**

**Update (2026-07-28) — the 2×2 test above was overtaken by events; see the
new section immediately below.** `STT_OFF_THREAD=0` was run for real on
single-GPU `wb-gpu-a1ultra` (with `DEMO_STEP_PACING=1`) — zero crashes, a
full multi-turn session, no CUDA-graph surface at all in this mode. That
answers this section's own open question: **single GPU is safe once
STT-off-thread is disabled.** But the live-fluency question this test was
really chasing turned out to have a different, much bigger answer than
GPU count or STT threading — see "Real root cause of demo response lag"
below. `wb-gpu-a1ultra2g`/Option E have not been proven necessary by
anything found this session; they remain implemented and available, but
the case for keeping them running by default is weak pending the new
investigation.

**Update (2026-07-29) — default flipped to disabled.** A real demo A/B
mistake: two sessions (`demo/sessions/2026-07-29T05-50-41Z/`,
`demo/sessions/2026-07-29T06-18-06Z/`, both testing unrelated `temp_text`
values) were run without explicitly exporting `STT_OFF_THREAD=0`, silently
defaulting to the (at-the-time) enabled off-thread chain — costing a real
round trip before the missing export was noticed. Both showed a genuine
dropped-user-utterance bug (confirmed via the recorded client audio
containing real speech with zero corresponding server-side VAD/STT
activity) that the one session with it explicitly disabled
(`demo/sessions/2026-07-28T07-23-04Z/`, referenced above) never exhibited.
**Not proven as the definitive root cause of the dropped utterance** — the
mechanism linking off-thread STT to a dropped utterance specifically
(rather than just the already-known CUDA-graph-crash risk this section is
about) is still a hypothesis, not confirmed — but the correlation is real,
and disabling it is already independently validated as safe on single-GPU
hardware per the update immediately above. Given that, defaulting to the
safer, already-proven mode is the right call while the hypothesis gets
tested properly, rather than leaving a footgun that depends on everyone
remembering to export a variable. `core/model_interface.py`'s
`_stt_off_thread_enabled()` now returns `os.environ.get("STT_OFF_THREAD",
"0") == "1"` (was `!= "0"`, default `"1"`) — set `STT_OFF_THREAD=1` to
explicitly opt back into the off-thread + Option E chain.

### Real root cause of demo response lag: model-generation `<pad>` sampling drift, not retrieval/GPU/pacing (2026-07-28)

**This section supersedes the framing of every section above it in this
file that attributed demo dead-air to retrieval latency, GPU/CUDA-graph
work, or client-side jitter buffering.** All of that investigation was
real and its fixes are real (retrieval genuinely was slow before
flash-lite; the CUDA-graph crash genuinely needed Option E if STT stays
off-thread; the jitter-buffer/pacing work genuinely closed a real gap in
upstream moshi-rag) — but none of it was the dominant contributor to what
the user actually perceives as demo lag. Found via the two ground-truth
tools built this session (real acoustic analysis of the saved conversation
recording — `scripts/analyze_recorded_audio.py` — and direct source
reading of the generation code), not inferred from proxies.

**The user's own report, checked directly against a real recording and
confirmed exactly**: at 18s into a saved session recording, the user's
first question ends; the model doesn't start responding until 23s — a
genuine ~5s gap. Cross-referencing against `server.log` for that same
session (`demo/sessions/2026-07-28T07-23-04Z/`, `STT_OFF_THREAD=0`,
`DEMO_STEP_PACING=1`, flash-lite retrieval already active) found:

- Retrieval and ARC-encoding are both genuinely fast now (0.36-0.66s each,
  confirmed via `backend_latency=`/`ARC encoding received in` log lines) —
  ruled out as the cause of a 5s gap.
- The real mechanism: `moshi/inference_utils/turn_manager.py`'s
  `TurnManager._update_active_speaker()` only initiates the switch-to-model
  countdown once `self.model_text_buffer` is non-empty (line ~90); while
  it's empty, it just logs `"[VAD] User stopped speaking but LM buffer
  empty, remaining in user turn"` and loops, once per real step (~80ms).
  Counting consecutive occurrences of that exact line immediately before
  each of this session's 5 `switch to model` events: **21, 63, 58, 69, 80**
  — present on turn 1 (no retrieval at all) and growing across the whole
  session, not specific to `<ret>` turns. Retrieval turns only *look*
  worse because they tend to occur later in a conversation, where this
  effect has already compounded — confirmed by checking the exact same
  stall-count-before-switch-to-model metric on non-retrieval turns too,
  in the same session, in `git log`-independent real data, not assumed.
- Ruled out via direct source reading, not guessed: `RingKVCache`
  (`modules/transformer.py`) is a genuinely fixed-capacity ring buffer
  (`context: 3000` in the real checkpoint's `config.json` — confirmed by
  the user directly from the VM — i.e. ~240s at 12.5Hz), so attention
  cost/memory is constant regardless of session length; our own
  `_fetch_and_apply_reference_conditioning()` (`core/model_interface.py:220`)
  replaces `pending_streaming_sums` fresh every turn via direct overwrite,
  no accumulation. Neither explains a growing stall.
- What's actually happening: `LMGen._step()`'s text token
  (`lm.py:834`, `sample_token(text_logits, use_sampling=True, temp=0.7,
  top_k=25)`) is a **genuine stochastic sample** from the model's own
  logits every single step — no hardcoded silence threshold anywhere in
  this path. Confirmed `use_sampling=True`/`temp_text=0.7`/`top_k_text=25`
  are truly in effect (not a checkpoint override) by having the user `cat`
  the real checkpoint's `config.json` directly from the VM: **no
  `lm_gen_config` key at all**, so `LMGen.__init__`'s hardcoded Python
  defaults apply unmodified. The noisy-but-growing stall-count pattern is
  exactly what a repeated-Bernoulli-trial process looks like when its
  underlying "chance of not drawing `<pad>` this step" is slowly
  decreasing — consistent with, though not yet proven to be caused by,
  something about accumulating conversation context/conditioning shifting
  the model's own logits toward `<pad>` the longer a session runs.

**Not yet done, in either direction — the two open levers for next
session**:
1. **Try raising `temp_text`/`top_k_text`** on the main model's `LMGen`
   construction (`inference_utils/utils.py:153-159`, currently unwired to
   any of our config — would need a narrow patch, analogous in shape to
   existing patches in this file, not an upstream reimplementation) and
   test empirically whether the stall shrinks. Cheap, reversible, directly
   tests whether this is a sampling-conservativeness problem rather than
   something deeper.
2. **Controlled experiment**: does the stall track conversation
   depth/content, or raw step count since session start (`state.offsets`
   in `lm.py`, never reset except on a true `is_first` slot reset)? Ask a
   `<ret>`-triggering question as literally the *first* turn of a fresh
   session and see if the stall is already small — isolates which
   mechanism is really drifting.

**Tooling built this session to make this investigable at all** (previous
sessions' server-log-only/client-buffer-only diagnostics couldn't have
found this):
- `scripts/analyze_recorded_audio.py` — decodes the client's saved
  conversation recording (`useRecording.ts`'s "Save audio", confirmed to
  capture the same post-worklet audio that drives the speakers, mixed with
  the user's mic) via `ffmpeg`, computes a self-calibrating RMS energy
  envelope, and merges real acoustic onset/offset/glitch events with
  `server.log`'s own timestamps into one chronological timeline. Alignment
  is anchored to `server.log`'s literal `"new WebSocket client connected"`
  line plus the recording filename's embedded `_startT<X>s` tag (both
  `useServerAudio.ts` and `useRecording.ts` read the same
  `performance.now()` clock) — **not** `raw_events.jsonl`'s own `t_rel_s`,
  which is anchored to `SessionLog.__init__`'s `time.perf_counter()` at
  server-process startup (before model loading even begins), a materially
  different zero point than the client's WebSocket-open anchor. An earlier
  version of this file's own text incorrectly claimed these two `t_rel_s`
  values lined up — corrected after checking `SessionLog.__init__` directly.
  End-to-end verified against a real synthesized `.webm` (not just unit
  tests against synthetic WAV data) before trusting it on real session data.
- `demo/client/src/audio-processor.ts`/`useServerAudio.ts`'s
  `getAudioDiagLog()`/`liveBufferS` — the corrected jitter-buffer
  diagnostic (see "Real-time step pacing" section above for the first,
  wrong version of this field). On the real `STT_OFF_THREAD=0` +
  `DEMO_STEP_PACING=1` session, `liveBufferS` sat at 0ms for 99.3% of
  samples, max 37ms, zero real buffer-drop events — the jitter buffer
  itself is genuinely healthy now. This is *not* what's causing the
  perceived lag; the pad-sampling drift above is a separate, downstream-
  of-VAD, upstream-of-audio issue.
- `ffmpeg` had to be installed into a user-writable, isolated `apk --root`
  prefix (`~/.local/ffmpeg-root`, wired via `~/.bashrc`) in the environment
  this analysis runs in — no system package-manager privileges available
  there; a static-binary/apk-root workaround, not an `apt`/`apk` install
  in the usual sense. Not relevant to the VM/demo environment itself, only
  to wherever `analyze_recorded_audio.py` actually gets run.

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

**Phase 0 update (post-pivot investigation, via the now-retired `prodcheck/`
ground-truth harness)**: re-ran this exact scenario against the real,
unmodified `moshi.server` stack (which has
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

### RESOLVED: `ttfat_s` always exactly 0.0 (found validating `latency.ttfat`)

Every real VM run of `respond()` (any eval, any checkpoint) reported
`ttfat_s=0.000` in its `[respond]` log line, and `latency.ttfat`'s
aggregated mean/p50/p95 were all `0.00s` — including turns with substantial
real generation, not just degenerate-silence ones. Root cause, confirmed
against real `inference_utils/inference_job.py` source (cloned directly,
not guessed): `_TimedInferenceJob.run()` used to set `self._t_question_end
= time.time()` right after `await original_feed()` (the real
`_feed_loop()`) *returned*. But under `stop_on_end_of_input=False`
(`respond()`'s own deliberate setting — see the earlier
`stop_on_end_of_input` bug above), `_feed_loop()` does **not** return when
the user's real audio ends — it falls into an unconditional loop feeding
zero-padding silence until the whole turn's `_shutdown_event` fires (i.e.
near end-of-turn, not end-of-question). Since first real audio output
always precedes end-of-turn, `t_first_audio < t_question_end` by
construction, and `ttfat_s = max(0.0, t_first_audio - t_question_end)`
clamped to exactly `0.0` on every single call.

**Fix**: `_feed_loop()`'s own `feed_step` counter (source of
`trace["question_end_step"]`, moshi-rag's own correct index for the last
*real* input frame) increments exactly once per `input_queue.put()` call,
in both its real-input loop and its trailing-silence loop. `run()` now
wraps `input_queue.put()` (not `_feed_loop` itself — a single-method,
observation-only patch, no behavior change) to record a
`feed_step -> wall-clock time` list, then a new `_finalize_ttfat()`,
called once `job.run()` fully returns, looks up the wall-clock time at
`trace["question_end_step"]` and uses *that* as `t_question_end` — the
precise moment the user's real question finished being fed, independent of
whatever `_feed_loop` does afterward.

**Second, independent bug found re-verifying the fix above on a real VM
run**: `ttfat_s` was *still* exactly `0.000` after the `t_question_end` fix
landed. Added temporary diagnostic logging (`[TTFAT-DIAG]`) rather than
guessing again, and it was conclusive: `t_first_audio` (the "first audio
token" side of the calculation, set via `output_queue.get()`'s
`pcm.abs().max() > 1e-6` check) fired at `step_index=0` with
`pcm_max_amp=0.01241324` — identical to 8 decimal places — across five
completely different real questions/responses in the same run. A value
that doesn't vary at all with content that different is definitionally not
content-dependent speech onset; it's a fixed artifact (most likely a Mimi
decoder warm-up transient) that happens to clear a `1e-6` threshold on the
very first decoded frame, every time, regardless of what's actually being
said. **Fix**: `t_first_audio` is now set on the first non-pad *text*
token instead of PCM amplitude — MoshiRAG generates text and audio in
lockstep, and the step-loop already distinguishes real tokens from
`<pad>` (`_decode_text_token(text_token) is None`) for other bookkeeping,
so this reuses an existing, already-correct signal rather than tuning a
new amplitude threshold empirically. Lives inside the *already-patched*
`_patched_output_loop` (the same function `asr_wait_s`/`rag_trigger_step`
already extend this session), not a new patch surface. The old
`output_queue.get` wrapping in `run()` is gone entirely — first-audio
detection no longer needs it.

Both fixes are unit-tested (fake job, no GPU needed to verify the wiring)
but **not yet re-verified against a real checkpoint** as of this writing —
do that before trusting any `latency.ttfat` numbers from a run predating
this second fix, including runs made *after* the first fix but before this
one (they'd still show `ttfat_s=0.000` for the reason described above).

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

#!/bin/bash
# Start the full moshi-rag demo stack on the GCP VM.
#
# Launches two services in a tmux session "demo":
#   window 0 (conditioner): reference encoder conditioner on port 8001
#   window 1 (server):      moshi-rag main server on port 8998, wrapped by
#                            scripts/instrumented_server.py (mostly a narrow
#                            logging patch, plus one deliberate retrieval-
#                            routing exception — see that file's own
#                            docstring), serving the built demo/client/ fork
#
# Each launch (except --conditioner-only) creates a fresh session directory
# at demo/sessions/<session_id>/, tees both processes' raw stdout/stderr into
# {conditioner,server}.log there (so logs survive independent of tmux's
# lifecycle/scrollback), and points instrumented_server.py at it via
# DEMO_SESSION_DIR to write turns.jsonl — see specs/
# moshirag-evals-requirements.md's "Session instrumentation" section. Review
# a session afterward with:
#   uv run scripts/summarize_demo_session.py demo/sessions/<session_id>/
#
# Connect from macOS once the server is ready:
#   ssh -L 8998:localhost:8998 wb-gpu-a1ultra
#   Then open http://localhost:8998 in your browser.
#
# Usage:
#   bash scripts/run_demo.sh [--checkpoint <alias>] [--stt local|gradium] [--config <path>] [--rag-timeout <seconds>] [--batch-size <n>] [--conditioner-only]
#
# Options:
#   --checkpoint  Checkpoint alias from configs/checkpoints.yaml (default: base)
#   --stt         local  = moshi built-in STT, no credentials needed (default)
#                 gradium = Gradium streaming ASR, lower latency, requires
#                           STT_URL and STT_API_KEY in .env
#   --config      Path to the same YAML config shape evals/runner.py uses
#                 (default: configs/baseline_with_retrieval.yaml). Derives
#                 the retrieval backend's env vars and the model.generation
#                 block's CLI flags via scripts/print_demo_env.py — see
#                 specs/moshirag-evals-requirements.md's "Config-derived
#                 environment" section under "Demo" for the full design.
#   --rag-timeout Explicit override for the retrieval-LLM timeout, layered
#                 on top of whatever --config's model.generation.rag_timeout
#                 produced (default: unset, config value wins). moshi.server's
#                 own hardcoded default is 1.5s, which CLAUDE.md's Phase 0
#                 prodcheck investigation confirmed is shorter than real
#                 Gemini round-trip latency (observed 2.5-3s) — see
#                 CLAUDE.md's retrieval-latency notes before lowering this.
#   --batch-size  moshi.server's reserved conversation slots (default: 1,
#                 overriding moshi's own default of 16). Unused slots are not
#                 free -- BatchRunner.run_step() computes the full batch
#                 tensor every step regardless of how many slots are active,
#                 so 16 costs ~16x the necessary work against an 80ms
#                 real-time budget. See docs/demo-turn-onset-regression.md
#                 §4.6. Deliberately a CLI flag here rather than a
#                 model.generation config field -- see specs/
#                 moshirag-evals-requirements.md's "Generation parameters".
#   --conditioner-only  Launch only server_conditioner (port 8001), skip
#                 moshi.server entirely. For pointing evals/runner.py at a
#                 real conditioner without also loading the full model a
#                 second time (MoshiRAGAdapter loads its own copy) — see
#                 CLAUDE.md's Phase 2 "mandatory separate-process
#                 conditioning" section. Also sidesteps having to hand-type
#                 a long server_conditioner invocation into a terminal that
#                 hard-wraps long pasted lines (confirmed to corrupt
#                 multi-line commands on at least one real VM session).
#
# Env vars (forwarded to the server window only if set in the invoking
# shell, opt-in, off by default):
#   DEMO_ASYNCIO_DEBUG=1              Turn on asyncio debug mode on the real
#                                      server event loop for one diagnostic
#                                      session — logs any callback exceeding
#                                      DEMO_ASYNCIO_DEBUG_THRESHOLD_S (default
#                                      0.02s) with a repr identifying it. See
#                                      scripts/instrumented_server.py's
#                                      _patch_event_loop_diagnostics() and
#                                      CLAUDE.md's "SUPERSEDED: GPU contention
#                                      conclusion was wrong" section for why
#                                      this exists. Adds real per-callback
#                                      overhead — don't leave this on for
#                                      normal use.
#
# This is a straight revival of the production-server-based launcher this
# repo used before 5b3790e switched the demo to a custom in-process
# FastAPI/WebSocket adapter — see specs/moshirag-evals-requirements-v2.md's
# "New Marching Orders" section for why. Reversibility is via git history
# only: this file and its pre-5b3790e predecessor are both in `git log`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────────────
CHECKPOINT_ALIAS=base
STT_MODE=local
CONFIG_PATH=configs/baseline_with_retrieval.yaml
RAG_TIMEOUT_OVERRIDE=""
CONDITIONER_ONLY=0
FORWARDED_ENV=""
# moshi.server's own default is 16 (reserved concurrent-conversation KV-cache
# slots). We override to 1 because unused slots are NOT free: BatchRunner's
# run_step() computes the full batch tensor every step -- exec_mask gates
# state updates, not compute -- so batch 16 costs ~16x the necessary per-step
# work against a hard 80ms real-time budget (1/12.5Hz). This demo is
# single-session by design (see the spec's "Explicitly Out of Scope"), and
# the eval path already hardcodes batch_size=1. Raise it only if genuinely
# serving concurrent users, and expect to pay for it in step time. See
# docs/demo-turn-onset-regression.md §4.6.
BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)       CHECKPOINT_ALIAS="$2";       shift 2 ;;
        --stt)              STT_MODE="$2";               shift 2 ;;
        --config)           CONFIG_PATH="$2";            shift 2 ;;
        --rag-timeout)      RAG_TIMEOUT_OVERRIDE="$2";   shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";             shift 2 ;;
        --conditioner-only) CONDITIONER_ONLY=1;           shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$STT_MODE" != "local" && "$STT_MODE" != "gradium" ]]; then
    echo "--stt must be 'local' or 'gradium'" >&2; exit 1
fi

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set in .env}"

if [[ "$STT_MODE" == "gradium" ]]; then
    : "${STT_URL:?STT_URL must be set in .env for --stt gradium}"
    : "${STT_API_KEY:?STT_API_KEY must be set in .env for --stt gradium}"
fi

# ── Resolve checkpoint ────────────────────────────────────────────────────────
echo "Resolving checkpoint '$CHECKPOINT_ALIAS'..."
CKPT=$(uv run --all-extras python -c "
from dotenv import load_dotenv; load_dotenv()
from core.checkpoint import resolve_checkpoint
print(resolve_checkpoint('$CHECKPOINT_ALIAS'))
")
echo "  → $CKPT"

MOSHI_WEIGHT="$CKPT/model.safetensors"
MIMI_WEIGHT=$(ls "$CKPT"/tokenizer-*.safetensors 2>/dev/null | head -1)
TOKENIZER=$(ls "$CKPT"/*.model 2>/dev/null | head -1)

[[ -f "$MOSHI_WEIGHT" ]] || { echo "model.safetensors not found at $CKPT" >&2; exit 1; }
[[ -f "$MIMI_WEIGHT"  ]] || { echo "tokenizer safetensors not found at $CKPT" >&2; exit 1; }
[[ -f "$TOKENIZER"    ]] || { echo "tokenizer .model not found at $CKPT" >&2; exit 1; }

# ── Config-derived environment (LLM_BASE_URL/LLM_MODEL_NAME/LLM_API_KEY,
# DEMO_GENERATION_FLAGS) ───────────────────────────────────────────────────
# See scripts/print_demo_env.py's own docstring and specs/
# moshirag-evals-requirements.md's "Config-derived environment" section for
# why LLM_BASE_URL/LLM_MODEL_NAME/LLM_API_KEY are needed at all (vestigial —
# only to keep ServerState.warmup() from raising) and why
# MOSHI_RETRIEVAL_LLMS_JSON is deliberately never set.
echo "Deriving environment from '$CONFIG_PATH'..."
eval "$(uv run --all-extras python scripts/print_demo_env.py --config "$CONFIG_PATH")"

# ── GPU device assignment ─────────────────────────────────────────────────────
# Auto-detected from real hardware -- the same script, same rsynced files,
# and same commands run unchanged on the current single-A100 box and a
# future dual-A100 one; nothing here branches on hostname or machine type.
# See core/gpu.py and specs/moshirag-evals-requirements.md's "GPU Sizing and
# Multi-GPU Deployment" section for the resolution rule and why an explicit
# CUDA_VISIBLE_DEVICES already set in this shell's environment always wins
# over auto-detection (each process still gets --cuda-device 0/--device
# cuda below regardless -- once CUDA_VISIBLE_DEVICES scopes a process to one
# physical GPU, that GPU is always addressed as index 0 from inside it).
FRONTEND_CUDA_VISIBLE_DEVICES=$(uv run --all-extras python -m core.gpu frontend)
COND_CUDA_VISIBLE_DEVICES=$(uv run --all-extras python -m core.gpu conditioner)
GPU_CONTENDED=0
[[ "$FRONTEND_CUDA_VISIBLE_DEVICES" == "$COND_CUDA_VISIBLE_DEVICES" ]] && GPU_CONTENDED=1

# ── Build env prefix (passed into each tmux window) ──────────────────────────
ENV_PREFIX="REFERENCE_ENCODER_URL=http://localhost:8001"
ENV_PREFIX+=" LLM_BASE_URL=$LLM_BASE_URL"
ENV_PREFIX+=" LLM_API_KEY=$LLM_API_KEY"
ENV_PREFIX+=" LLM_MODEL_NAME=$LLM_MODEL_NAME"
ENV_PREFIX+=" OPENAI_API_KEY=$GEMINI_API_KEY"

if [[ "$STT_MODE" == "gradium" ]]; then
    ENV_PREFIX+=" STT_URL=$STT_URL STT_API_KEY=$STT_API_KEY"
fi

# ── Build service commands ────────────────────────────────────────────────────
CONDITIONER_CMD="$ENV_PREFIX CUDA_VISIBLE_DEVICES=$COND_CUDA_VISIBLE_DEVICES uv run --all-extras python -m moshi.server_conditioner \
  --config '$CKPT/config.json' \
  --moshi-weight '$MOSHI_WEIGHT' \
  --cuda-device 0 \
  --conditioner reference_with_time \
  --port 8001"

SESSION=demo

if [[ "$CONDITIONER_ONLY" == "1" ]]; then
    # ── Launch conditioner alone ────────────────────────────────────────────
    # Tee'd to a persistent on-disk log, same as the full (non-conditioner-only)
    # path below — a real gap found the hard way: this branch used to run the
    # conditioner with no redirection at all, so its only record was tmux's
    # own scrollback. When the tmux *server* itself died (not just this
    # session — e.g. a VM hiccup, an OOM kill, anything that takes the whole
    # server process down), that scrollback was gone with it, along with any
    # crash traceback that would have explained why the conditioner stopped
    # responding. A conditioner crash mid-eval-run is exactly the kind of
    # unattended failure this project's evals are most exposed to (see
    # CLAUDE.md's reference-encoder ConnectError notes) — worth a durable log
    # regardless of how rarely it fires.
    CONDITIONER_ONLY_SESSION_ID=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
    CONDITIONER_ONLY_SESSION_DIR="$PROJECT_DIR/demo/sessions/$CONDITIONER_ONLY_SESSION_ID"
    mkdir -p "$CONDITIONER_ONLY_SESSION_DIR"

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -n conditioner
    tmux send-keys -t "$SESSION:conditioner" \
        "cd $PROJECT_DIR && $CONDITIONER_CMD 2>&1 | tee '$CONDITIONER_ONLY_SESSION_DIR/conditioner.log'" Enter

    echo ""
    echo "Conditioner-only launching in tmux session '$SESSION' (port 8001, CUDA_VISIBLE_DEVICES=$COND_CUDA_VISIBLE_DEVICES)."
    echo "Log: $CONDITIONER_ONLY_SESSION_DIR/conditioner.log"
    if [[ "$GPU_CONTENDED" == "1" ]]; then
        echo ""
        echo "⚠ single-GPU mode: whatever process you point at this conditioner will"
        echo "  share this same physical GPU with it -- known conditioner contention,"
        echo "  do not trust retrieval-latency or grounding-dependent scores from"
        echo "  that run (see specs/moshirag-evals-requirements.md's \"GPU Sizing and"
        echo "  Multi-GPU Deployment\" section)."
    fi
    echo ""
    echo "  tmux attach -t $SESSION   — watch logs"
    echo ""
    # Same correction as the main path below: aiohttp, not uvicorn.
    echo "Once it prints '[Startup] HTTP server listening', point evals at it:"
    echo ""
    echo "  export REFERENCE_ENCODER_URL=http://localhost:8001"
    echo ""
    echo "To stop: tmux kill-session -t $SESSION"
    exit 0
fi

# ── Session directory ─────────────────────────────────────────────────────────
SESSION_ID=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
SESSION_DIR="$PROJECT_DIR/demo/sessions/$SESSION_ID"
mkdir -p "$SESSION_DIR"
echo "Session log directory: $SESSION_DIR"

# ── Build the web client (demo/client/, forked + rebranded moshi-rag client) ──
echo "Building web client (demo/client/)..."
(cd "$PROJECT_DIR/demo/client" && npm install --no-audit --no-fund && npm run build)
CLIENT_DIST="$PROJECT_DIR/demo/client/dist"
[[ -f "$CLIENT_DIST/index.html" ]] || { echo "Client build did not produce $CLIENT_DIST/index.html" >&2; exit 1; }

SERVER_CMD="$ENV_PREFIX CUDA_VISIBLE_DEVICES=$FRONTEND_CUDA_VISIBLE_DEVICES DEMO_SESSION_DIR='$SESSION_DIR' DEMO_CHECKPOINT='$CHECKPOINT_ALIAS' DEMO_CONFIG='$CONFIG_PATH' \
  uv run --all-extras python scripts/instrumented_server.py \
  --moshi-weight '$MOSHI_WEIGHT' \
  --mimi-weight '$MIMI_WEIGHT' \
  --tokenizer '$TOKENIZER' \
  --device cuda \
  --port 8998 \
  --init-active-speaker model \
  --batch-size $BATCH_SIZE \
  $DEMO_GENERATION_FLAGS \
  --static '$CLIENT_DIST'"

# Explicit --rag-timeout, if given, is appended last so it wins over
# whatever $DEMO_GENERATION_FLAGS produced (argparse: later flag wins) —
# an ad-hoc override layered on top of the config, not the sole source.
[[ -n "$RAG_TIMEOUT_OVERRIDE" ]] && SERVER_CMD+=" --rag-timeout $RAG_TIMEOUT_OVERRIDE"

# ── Forward opt-in env vars explicitly into the tmux command ─────────────────
#
# These MUST be spliced into SERVER_CMD, not merely exported by the caller.
# The server runs via `tmux send-keys` into a fresh shell owned by the tmux
# *server* process, which inherits its environment once — whenever that server
# first started. If a tmux server is already running (a leftover `eval`/`setup`
# session is enough), a freshly exported var in the invoking shell silently
# never reaches the demo, and the run quietly uses defaults instead.
#
# This is not hypothetical: it is the confirmed mechanism behind the 2026-07-29
# A/B mistake where two sessions believed to be testing STT_OFF_THREAD=0 in
# fact ran with the off-thread chain enabled, costing a real VM round trip
# (see CLAUDE.md's "Update (2026-07-29) — default flipped to disabled").
# DEMO_ASYNCIO_DEBUG was already handled this way and worked, which is exactly
# why it was the one diagnostic that never mysteriously failed to fire.
#
# Only vars actually set in the invoking shell are forwarded, so unset means
# "use the code's own default" rather than an empty string overriding it.
for _var in DEMO_QUEUE_DIAG DEMO_QUEUE_DIAG_EVERY DEMO_STEP_PACING STT_OFF_THREAD \
            DEMO_ASYNCIO_DEBUG DEMO_ASYNCIO_DEBUG_THRESHOLD_S; do
    if [[ -n "${!_var:-}" ]]; then
        SERVER_CMD="$_var=${!_var} $SERVER_CMD"
        FORWARDED_ENV+=" $_var=${!_var}"
    fi
done
# DEMO_ASYNCIO_DEBUG's threshold has a default that only applies when the
# debug flag itself is on, and is otherwise unset above — keep that behaviour.
if [[ -n "${DEMO_ASYNCIO_DEBUG:-}" && -z "${DEMO_ASYNCIO_DEBUG_THRESHOLD_S:-}" ]]; then
    SERVER_CMD="DEMO_ASYNCIO_DEBUG_THRESHOLD_S=0.02 $SERVER_CMD"
    FORWARDED_ENV+=" DEMO_ASYNCIO_DEBUG_THRESHOLD_S=0.02(default)"
fi
[[ "$STT_MODE" == "gradium" ]] && SERVER_CMD+=" --gradium-stt"

# ── Launch tmux session ───────────────────────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n conditioner

# Window 0: reference encoder conditioner — raw log tee'd, unpatched otherwise
tmux send-keys -t "$SESSION:conditioner" \
    "cd $PROJECT_DIR && $CONDITIONER_CMD 2>&1 | tee '$SESSION_DIR/conditioner.log'" Enter

# Window 1: main server — wait 20s for conditioner to finish loading
tmux new-window -t "$SESSION" -n server
tmux send-keys -t "$SESSION:server" \
    "cd $PROJECT_DIR && echo 'Waiting 20s for conditioner to load...' && sleep 20 && $SERVER_CMD 2>&1 | tee '$SESSION_DIR/server.log'" Enter

# ── Instructions ─────────────────────────────────────────────────────────────
RAG_TIMEOUT_NOTE="from $CONFIG_PATH"
[[ -n "$RAG_TIMEOUT_OVERRIDE" ]] && RAG_TIMEOUT_NOTE="${RAG_TIMEOUT_OVERRIDE}s (--rag-timeout override)"
echo ""
echo "Demo launching in tmux session '$SESSION' (STT: $STT_MODE, config: $CONFIG_PATH, rag-timeout: $RAG_TIMEOUT_NOTE)."
echo "Session dir: $SESSION_DIR"
echo "batch-size: $BATCH_SIZE"
# Echoed so an A/B run can be verified from this output alone, without
# attaching to tmux and reading the server's own startup banner — the
# forwarding bug this guards against was invisible precisely because nobody
# checked which variables actually reached the process.
if [[ -n "$FORWARDED_ENV" ]]; then
    echo "Forwarded diagnostic env:$FORWARDED_ENV"
else
    echo "Forwarded diagnostic env: none (all defaults)"
fi
# Informational, not a warning about data validity. The original text here
# told the operator "do not trust retrieval-latency or grounding-dependent
# behavior from this session" on the strength of the GPU-contention finding —
# a conclusion CLAUDE.md's own "SUPERSEDED: GPU contention conclusion was
# wrong" section later disproved on real dual-GPU hardware (true physical
# separation measured 1.686s vs single-GPU's 1.58s, i.e. no improvement; the
# real cause was event-loop starvation, fixed in
# _fetch_and_apply_reference_conditioning). Leaving that text in place was
# actively harmful: it discredits perfectly good single-GPU sessions. Real
# single-GPU numbers from 2026-07-30T19-12-46Z: retrieval 0.455-0.691s, ARC
# encoding 0.165-0.593s — comfortably inside the paper's 1.5s degradation
# threshold. What IS still true on one GPU is the VRAM ceiling, so that is
# what this now says.
if [[ "$GPU_CONTENDED" == "1" ]]; then
    echo ""
    echo "ⓘ single-GPU mode: conditioner and front-end share one physical GPU."
    echo "  This is fine for latency -- the old \"GPU contention\" finding was"
    echo "  disproved (see CLAUDE.md's \"SUPERSEDED: GPU contention conclusion"
    echo "  was wrong\"); conditioning is ~0.2-0.6s here. The real single-GPU"
    echo "  constraint is VRAM: front-end + conditioner reach ~66-69GB of 80GB"
    echo "  at --batch-size 16, so do not also start an eval against this"
    echo "  conditioner, and drop --batch-size if you hit OOM."
fi
echo ""
echo "  tmux attach -t $SESSION              — watch all logs"
echo "  tmux select-window -t $SESSION:0     — conditioner (port 8001)"
echo "  tmux select-window -t $SESSION:1     — server (port 8998)"
echo ""
# 'Uvicorn running' never appears -- moshi.server is aiohttp, not uvicorn.
# Confirmed against a real server.log: the ready lines are "Access the Web UI
# directly at ..." and "[Startup] HTTP server listening". The old string sent
# operators looking for output that does not exist.
echo "Once the server prints '[Startup] HTTP server listening', run from your machine:"
echo ""
echo "  ssh -L 8998:localhost:8998 wb-gpu-a1ultra"
echo ""
echo "  Then open: http://localhost:8998"
echo ""
echo "Session logs: $SESSION_DIR"
echo "  turns.jsonl, raw_events.jsonl, conditioner.log and server.log persist"
echo "  there regardless of tmux's lifecycle (plus step_diag.jsonl when"
echo "  DEMO_QUEUE_DIAG=1). Review with:"
echo ""
# --all-extras per CLAUDE.md's uv rule for the VM: a plain `uv run` only
# reconciles the base dependency group and can leave gpu-extra packages
# silently drifted.
echo "  uv run --all-extras scripts/summarize_demo_session.py $SESSION_DIR"
echo "  uv run --all-extras scripts/count_pad_stalls.py $SESSION_DIR"
echo ""
echo "To stop: tmux kill-session -t $SESSION"

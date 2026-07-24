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
#   bash scripts/run_demo.sh [--checkpoint <alias>] [--stt local|gradium] [--config <path>] [--rag-timeout <seconds>] [--conditioner-only]
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)       CHECKPOINT_ALIAS="$2";       shift 2 ;;
        --stt)              STT_MODE="$2";               shift 2 ;;
        --config)           CONFIG_PATH="$2";            shift 2 ;;
        --rag-timeout)      RAG_TIMEOUT_OVERRIDE="$2";   shift 2 ;;
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
CONDITIONER_CMD="$ENV_PREFIX uv run --all-extras python -m moshi.server_conditioner \
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
    echo "Conditioner-only launching in tmux session '$SESSION' (port 8001)."
    echo "Log: $CONDITIONER_ONLY_SESSION_DIR/conditioner.log"
    echo ""
    echo "  tmux attach -t $SESSION   — watch logs"
    echo ""
    echo "Once it prints 'Uvicorn running', point evals at it:"
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

SERVER_CMD="$ENV_PREFIX DEMO_SESSION_DIR='$SESSION_DIR' DEMO_CHECKPOINT='$CHECKPOINT_ALIAS' DEMO_CONFIG='$CONFIG_PATH' \
  uv run --all-extras python scripts/instrumented_server.py \
  --moshi-weight '$MOSHI_WEIGHT' \
  --mimi-weight '$MIMI_WEIGHT' \
  --tokenizer '$TOKENIZER' \
  --device cuda \
  --port 8998 \
  --init-active-speaker model \
  $DEMO_GENERATION_FLAGS \
  --static '$CLIENT_DIST'"

# Explicit --rag-timeout, if given, is appended last so it wins over
# whatever $DEMO_GENERATION_FLAGS produced (argparse: later flag wins) —
# an ad-hoc override layered on top of the config, not the sole source.
[[ -n "$RAG_TIMEOUT_OVERRIDE" ]] && SERVER_CMD+=" --rag-timeout $RAG_TIMEOUT_OVERRIDE"
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
echo ""
echo "  tmux attach -t $SESSION              — watch all logs"
echo "  tmux select-window -t $SESSION:0     — conditioner (port 8001)"
echo "  tmux select-window -t $SESSION:1     — server (port 8998)"
echo ""
echo "Once the server prints 'Uvicorn running', run from your machine:"
echo ""
echo "  ssh -L 8998:localhost:8998 wb-gpu-a1ultra"
echo ""
echo "  Then open: http://localhost:8998"
echo ""
echo "Session logs: $SESSION_DIR"
echo "  turns.jsonl, conditioner.log, and server.log persist there regardless"
echo "  of tmux's lifecycle. Review with:"
echo ""
echo "  uv run scripts/summarize_demo_session.py $SESSION_DIR"
echo ""
echo "To stop: tmux kill-session -t $SESSION"

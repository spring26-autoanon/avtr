#!/bin/bash
# Start the full moshi-rag demo stack on the GCP VM.
#
# Launches two services in a tmux session "demo":
#   window 0 (conditioner): reference encoder conditioner on port 8001
#   window 1 (server):      moshi-rag main server on port 8998
#
# Connect from macOS once the server is ready:
#   ssh -L 8998:localhost:8998 wb-gpu-a1ultra
#   Then open http://localhost:8998 in your browser.
#
# Usage:
#   bash scripts/run_demo.sh [--checkpoint <alias>] [--stt local|gradium] [--rag-timeout <seconds>] [--conditioner-only]
#
# Options:
#   --checkpoint  Checkpoint alias from configs/checkpoints.yaml (default: base)
#   --stt         local  = moshi built-in STT, no credentials needed (default)
#                 gradium = Gradium streaming ASR, lower latency, requires
#                           STT_URL and STT_API_KEY in .env
#   --rag-timeout Seconds to wait for the retrieval LLM before giving up
#                 (default: 8). moshi.server's own default is 1.5s, which
#                 CLAUDE.md's Phase 0 prodcheck investigation confirmed is
#                 shorter than real Gemini round-trip latency (observed
#                 2.5-3s) — the default here is raised accordingly. See
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
RAG_TIMEOUT=8
CONDITIONER_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)       CHECKPOINT_ALIAS="$2"; shift 2 ;;
        --stt)              STT_MODE="$2";         shift 2 ;;
        --rag-timeout)      RAG_TIMEOUT="$2";       shift 2 ;;
        --conditioner-only) CONDITIONER_ONLY=1;     shift 1 ;;
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

# ── Build env prefix (passed into each tmux window) ──────────────────────────
ENV_PREFIX="REFERENCE_ENCODER_URL=http://localhost:8001"
ENV_PREFIX+=" LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/"
ENV_PREFIX+=" LLM_API_KEY=$GEMINI_API_KEY"
ENV_PREFIX+=" LLM_MODEL_NAME=gemini-3.5-flash"
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
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -n conditioner
    tmux send-keys -t "$SESSION:conditioner" "cd $PROJECT_DIR && $CONDITIONER_CMD" Enter

    echo ""
    echo "Conditioner-only launching in tmux session '$SESSION' (port 8001)."
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

SERVER_CMD="$ENV_PREFIX uv run --all-extras python -m moshi.server \
  --moshi-weight '$MOSHI_WEIGHT' \
  --mimi-weight '$MIMI_WEIGHT' \
  --tokenizer '$TOKENIZER' \
  --device cuda \
  --port 8998 \
  --init-active-speaker model \
  --rag-timeout $RAG_TIMEOUT"

[[ "$STT_MODE" == "gradium" ]] && SERVER_CMD+=" --gradium-stt"

# ── Launch tmux session ───────────────────────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n conditioner

# Window 0: reference encoder conditioner
tmux send-keys -t "$SESSION:conditioner" "cd $PROJECT_DIR && $CONDITIONER_CMD" Enter

# Window 1: main server — wait 20s for conditioner to finish loading
tmux new-window -t "$SESSION" -n server
tmux send-keys -t "$SESSION:server" \
    "cd $PROJECT_DIR && echo 'Waiting 20s for conditioner to load...' && sleep 20 && $SERVER_CMD" Enter

# ── Instructions ─────────────────────────────────────────────────────────────
echo ""
echo "Demo launching in tmux session '$SESSION' (STT: $STT_MODE, rag-timeout: ${RAG_TIMEOUT}s)."
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
echo "To stop: tmux kill-session -t $SESSION"

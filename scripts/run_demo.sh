#!/bin/bash
# Start the full moshi-rag demo stack on the GCP VM.
#
# Launches two services in a tmux session "demo":
#   window 0 (conditioner): reference encoder conditioner on port 8001
#   window 1 (server):      moshi-rag main server on port 8998
#
# Connect from macOS once the server is ready:
#   gcloud compute start-iap-tunnel wb-gpu-a1ultra 8998 \
#     --local-host-port=localhost:8998 \
#     --project=adsp-s26-autoanon --zone=us-central1-c
#   Then open http://localhost:8998 in your browser.
#
# Usage:
#   bash scripts/run_demo.sh [--checkpoint <alias>] [--stt local|gradium]
#
# Options:
#   --checkpoint  Checkpoint alias from configs/checkpoints.yaml (default: base)
#   --stt         local  = moshi built-in STT, no credentials needed (default)
#                 gradium = Gradium streaming ASR, lower latency, requires
#                           STT_URL and STT_API_KEY in .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────────────
CHECKPOINT_ALIAS=base
STT_MODE=local

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT_ALIAS="$2"; shift 2 ;;
        --stt)        STT_MODE="$2";         shift 2 ;;
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
CKPT=$(uv run python -c "
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
ENV_PREFIX+=" LLM_MODEL_NAME=gemini-2.0-flash"
ENV_PREFIX+=" OPENAI_API_KEY=$GEMINI_API_KEY"

if [[ "$STT_MODE" == "gradium" ]]; then
    ENV_PREFIX+=" STT_URL=$STT_URL STT_API_KEY=$STT_API_KEY"
fi

# ── Build service commands ────────────────────────────────────────────────────
CONDITIONER_CMD="$ENV_PREFIX uv run python -m moshi.server_conditioner \
  --config '$CKPT/config.json' \
  --moshi-weight '$MOSHI_WEIGHT' \
  --cuda-device 0 \
  --conditioner reference_with_time \
  --port 8001"

SERVER_CMD="$ENV_PREFIX uv run python -m moshi.server \
  --moshi-weight '$MOSHI_WEIGHT' \
  --mimi-weight '$MIMI_WEIGHT' \
  --tokenizer '$TOKENIZER' \
  --device cuda \
  --port 8998 \
  --init-active-speaker model"

[[ "$STT_MODE" == "gradium" ]] && SERVER_CMD+=" --gradium-stt"

# ── Launch tmux session ───────────────────────────────────────────────────────
SESSION=demo
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
echo "Demo launching in tmux session '$SESSION' (STT: $STT_MODE)."
echo ""
echo "  tmux attach -t $SESSION              — watch all logs"
echo "  tmux select-window -t $SESSION:0     — conditioner (port 8001)"
echo "  tmux select-window -t $SESSION:1     — server (port 8998)"
echo ""
echo "Once the server prints 'Uvicorn running', run on macOS:"
echo ""
echo "  gcloud compute start-iap-tunnel wb-gpu-a1ultra 8998 \\"
echo "    --local-host-port=localhost:8998 \\"
echo "    --project=adsp-s26-autoanon --zone=us-central1-c"
echo ""
echo "  Then open: http://localhost:8998"
echo ""
echo "To stop: tmux kill-session -t $SESSION"

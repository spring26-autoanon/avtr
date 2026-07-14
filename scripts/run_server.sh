#!/bin/bash
# Start the moshi-rag WebSocket server on the VM.
# Connect from macOS via IAP tunnel:
#   gcloud compute start-iap-tunnel wb-gpu-a1ultra 8998 \
#     --local-host-port=localhost:8998 \
#     --project=adsp-s26-autoanon --zone=us-central1-c
# Then open http://localhost:8998 in your browser.
#
# REFERENCE_ENCODER_URL is set to a dummy address so the server starts
# without a live reference encoder. RAG retrieval will time out quickly
# and the model runs unaugmented — useful for baseline audio checks.

set -euo pipefail

CKPT=/home/jupyter/moshirag-evals/checkpoint_cache/adsp-s26-autoanon-bucket/checkpoints/base/moshirag-base-bf16

REFERENCE_ENCODER_URL=http://localhost:9999 \
uv run python -m moshi.server \
  --moshi-weight "$CKPT/model.safetensors" \
  --mimi-weight "$CKPT/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer "$CKPT/tokenizer_spm_32k_3.model" \
  --device cuda \
  --port 8998 \
  --rag-timeout 0.1

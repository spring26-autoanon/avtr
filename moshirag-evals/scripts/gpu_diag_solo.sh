#!/bin/bash
# GPU-contention diagnostic, part 1/2: solo baseline.
#
# Launches server_conditioner ALONE (scripts/run_demo.sh --conditioner-only
# — no front-end model loaded, so the GPU has nothing else on it), then
# fires 5 timed /embed calls against it while polling nvidia-smi at ~100ms
# resolution. Compare this output against gpu_diag_contended.sh's to
# isolate whether the ~2s ARC-encoder latency observed live is dominated
# by genuine compute cost (this script would already show ~1.5-2s here,
# with nothing else running) or by contention with a concurrently
# generating front-end (this script would show sub-second calls, only
# blowing up once gpu_diag_contended.sh's front-end is also active).
#
# Leaves the conditioner running afterward — gpu_diag_contended.sh reuses
# this same process deliberately, so "front-end generating or not" is the
# only variable that changes between the two runs.
#
# On a real dual-GPU box (see specs/moshirag-evals-requirements.md's "GPU
# Sizing and Multi-GPU Deployment" section): run_demo.sh --conditioner-only
# auto-assigns this conditioner to physical GPU 1 rather than sharing GPU 0
# with whatever front-end gpu_diag_contended.sh later loads. Neither script
# sets CUDA_VISIBLE_DEVICES itself to force sharing -- this pair was
# actually run this way on a real dual-GPU box (wb-gpu-a1ultra2g) to test
# whether separation would eliminate the conditioning latency. **It
# didn't**: gpu_diag_contended.sh's context_injection_s came back
# statistically unchanged from the original single-A100 figure (~1.5-1.9s)
# even with the conditioner genuinely isolated on GPU 1, confirmed idle via
# nvidia-smi throughout. Real root cause: moshi-rag's own real-time step
# loop blocking the event loop, not GPU sharing — since fixed. This pair
# stays useful as a real diagnostic tool (e.g. a regression check that the
# fix stays fixed), just not as a way to prove GPU topology matters — it
# doesn't. To force both processes back onto one GPU deliberately (e.g. to
# reproduce the single-GPU numbers for comparison), export
# CUDA_VISIBLE_DEVICES=0 before running both scripts.
#
# Usage: bash scripts/gpu_diag_solo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUT_DIR="$PROJECT_DIR/gpu_diag"
mkdir -p "$OUT_DIR"

echo "Launching conditioner-only (no front-end model loaded)..."
RUN_OUTPUT=$(bash scripts/run_demo.sh --checkpoint base --conditioner-only)
echo "$RUN_OUTPUT"

CONDITIONER_LOG=$(echo "$RUN_OUTPUT" | grep -oE '/[^ ]+/conditioner\.log' | head -1)
if [[ -z "$CONDITIONER_LOG" ]]; then
    echo "Could not find conditioner.log path in run_demo.sh output — aborting." >&2
    exit 1
fi

echo "Waiting for conditioner to finish loading (watching $CONDITIONER_LOG)..."
until curl -sf -o /dev/null http://localhost:8001/health 2>/dev/null; do
    sleep 2
done
echo "Conditioner ready."

REF_TEXT='Football in Kabul started in the nineteen-thirties with the Bagh-e-Bagh-e-Sabur, a game popular with British officials, while the general history of the sport is relatively new and complex.'

echo "Starting GPU utilization poll -> $OUT_DIR/solo_poll.csv"
: > "$OUT_DIR/solo_poll.csv"
( while true; do
    printf '%s,%s\n' "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> "$OUT_DIR/solo_poll.csv"
    sleep 0.1
  done ) &
POLL_PID=$!
trap 'kill "$POLL_PID" 2>/dev/null || true' EXIT

echo "Firing 5 timed /embed calls, 2s apart..." | tee "$OUT_DIR/solo_timings.txt"
for i in 1 2 3 4 5; do
    curl -s -o /dev/null -w "solo call $i: %{time_total}s\n" \
        -X POST http://localhost:8001/embed \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$REF_TEXT\"}" | tee -a "$OUT_DIR/solo_timings.txt"
    sleep 2
done

kill "$POLL_PID" 2>/dev/null || true
wait "$POLL_PID" 2>/dev/null || true
trap - EXIT

echo ""
echo "Done. Conditioner left running (tmux session 'demo') for gpu_diag_contended.sh."
echo "Results:"
echo "  $OUT_DIR/solo_timings.txt"
echo "  $OUT_DIR/solo_poll.csv"

#!/bin/bash
# GPU per-process VRAM peak measurement.
#
# Answers the a2-highgpu-2g vs a2-ultragpu-2g sizing question directly:
# how much VRAM does the front-end (eval path, batch_size=1 — see
# core/model_interface.py's MoshiRAGAdapter._load_models()) actually use on
# its own, separate from the conditioner's own footprint, when both are
# co-resident on the same GPU (today's only deployment). The ~66-69GB
# combined figure on record is from the demo path's batch_size=16 default,
# not this one — that distinction matters here.
#
# Polls `nvidia-smi --query-compute-apps` (per-process memory, not just
# total GPU memory) at ~0.5s resolution and reports peak usage per process
# once stopped — a single snapshot can catch either process mid-allocation
# or between generation bursts, so peak-over-a-real-run is what actually
# answers the sizing question.
#
# Usage:
#   1. Conditioner already running (e.g. via scripts/gpu_diag_solo.sh, or
#      bash scripts/run_demo.sh --conditioner-only).
#   2. In a separate terminal, start the eval (loads the front-end at
#      batch_size=1):
#        export REFERENCE_ENCODER_URL=http://localhost:8001
#        uv run --all-extras evals/runner.py --config configs/_scratch_validate_no_gsm8k.yaml --mode smoke
#   3. Run this script in a third terminal (before or right as the eval
#      starts), then Ctrl-C it once the eval finishes:
#        bash scripts/gpu_diag_per_process.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$PROJECT_DIR/gpu_diag"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/per_process_mem.csv"
: > "$LOG"

echo "Polling per-process GPU memory -> $LOG (Ctrl-C to stop and summarize)"

cleanup() {
    echo ""
    echo "=== Peak VRAM per process ==="
    awk -F',' '{
        pid = $2; mem = $3 + 0
        if (mem > max[pid]) max[pid] = mem
    } END {
        for (pid in max) print pid, max[pid]
    }' "$LOG" > "$OUT_DIR/per_process_peaks.txt"

    while read -r pid mem; do
        cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | cut -c1-70)
        printf "  pid %-8s peak %6s MiB   %s\n" "$pid" "$mem" "${cmd:-<process exited>}"
    done < "$OUT_DIR/per_process_peaks.txt"

    exit 0
}
trap cleanup INT TERM

while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | sed "s/^/$(date +%s.%N),/" >> "$LOG"
    sleep 0.5
done

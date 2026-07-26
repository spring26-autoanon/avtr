#!/bin/bash
# GPU-contention diagnostic, part 2/2: contended case.
#
# Reuses the SAME already-running conditioner process from
# scripts/gpu_diag_solo.sh (run that first) — the only variable that
# changes here is a front-end model actively generating against it, via a
# real evals/runner.py smoke run, so front-end GPU activity is the only
# difference from the solo baseline. Polls nvidia-smi throughout and tees
# the eval's own output, which (as of the conditioning_latency_s fix)
# reports the real per-turn ARC-encoder round trip directly in
# latency.retrieval_breakdown's context_injection_s — no manual
# conditioner.log timestamp-parsing needed.
#
# Deliberately does NOT launch its own conditioner: relaunching one here
# would make "which conditioner process" a second uncontrolled variable
# alongside "is the front-end generating," defeating the point of the
# comparison.
#
# On a real dual-GPU box, this script's own name is no longer literally
# accurate by default: MoshiRAGAdapter.__init__ auto-assigns the front-end
# to physical GPU 0, and gpu_diag_solo.sh's conditioner already auto-landed
# on GPU 1 -- so this run is genuinely NOT GPU-contended unless
# CUDA_VISIBLE_DEVICES was forced beforehand (see gpu_diag_solo.sh's
# header). That was the whole point of running it this way on
# wb-gpu-a1ultra2g: to test whether genuine separation would collapse
# context_injection_s back toward gpu_diag_solo.sh's solo numbers. **It
# didn't** -- context_injection_s came back statistically unchanged from
# the original single-A100 figure (~1.5-1.9s), while GPU 1 (the
# conditioner) sat ~0% utilized throughout per nvidia-smi. Real root cause,
# confirmed against moshi-rag's own source: `ServerState._step_loop` runs
# the real per-step model computation synchronously inside an `async def`,
# blocking the event loop and starving this exact conditioning HTTP call —
# not GPU contention, and not something GPU topology can fix. See
# CLAUDE.md's "SUPERSEDED: GPU contention conclusion was wrong" section for
# the full evidence and the fix (core/model_interface.py's
# _fetch_and_apply_reference_conditioning, applied on both the eval and
# demo paths). If rerunning this pair after that fix, expect
# context_injection_s to land close to the solo baseline regardless of
# whether the conditioner shares a GPU with the front-end or not — the fix
# targets the event-loop starvation directly, not GPU placement.
#
# Usage: bash scripts/gpu_diag_contended.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUT_DIR="$PROJECT_DIR/gpu_diag"
mkdir -p "$OUT_DIR"

if ! curl -sf -o /dev/null http://localhost:8001/health; then
    echo "No conditioner responding on localhost:8001." >&2
    echo "Run 'bash scripts/gpu_diag_solo.sh' first — it launches the conditioner" >&2
    echo "and leaves it running. This script deliberately reuses that same" >&2
    echo "process rather than starting a fresh one, so front-end activity is the" >&2
    echo "only variable that changes between the two runs." >&2
    exit 1
fi
echo "Conditioner confirmed reachable on localhost:8001."

echo "Starting GPU utilization poll -> $OUT_DIR/contended_poll.csv"
: > "$OUT_DIR/contended_poll.csv"
( while true; do
    printf '%s,%s\n' "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> "$OUT_DIR/contended_poll.csv"
    sleep 0.1
  done ) &
POLL_PID=$!
trap 'kill "$POLL_PID" 2>/dev/null || true' EXIT

echo "Running smoke eval against the shared conditioner (loads its own front-end model)..."
export REFERENCE_ENCODER_URL=http://localhost:8001
uv run --all-extras evals/runner.py --config configs/_scratch_validate_no_gsm8k.yaml --mode smoke \
    2>&1 | tee "$OUT_DIR/contended_eval.log"

kill "$POLL_PID" 2>/dev/null || true
wait "$POLL_PID" 2>/dev/null || true
trap - EXIT

echo ""
echo "Done. Results:"
echo "  $OUT_DIR/contended_eval.log   — full eval output"
echo "  $OUT_DIR/contended_poll.csv   — GPU utilization trace"
echo "  evals/results/*.json          — latency.retrieval_breakdown scores"
echo "    (context_injection_mean_s / p95_s = real ARC-encoder round trip,"
echo "     per-question values also in that result's transcript entries)"

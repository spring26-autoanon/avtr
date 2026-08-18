# Demo sprint runbook

## 1. Provision wb-gpu-training (Thu afternoon)

# Mac — push the repo state and the data Track A needs
rsync -avP --exclude runs --exclude runs_backup --exclude .git \
  --exclude finetune/data/datastereo --exclude finetune/data/prepared \
  /Users/leenatantawy/Downloads/capstone/moshi-finetune/ wb-gpu-training:~/moshi-finetune/

# Box — env (uv sync reads the fork-pinned pyproject)
cd ~/moshi-finetune && uv sync
uv pip install soundfile   # build_manifest needs it; not in pyproject
uv run python -c "import moshi; print(moshi.__version__)"   # expect 0.2.13
uv run hf auth login   # only if a download 401s — plain moshika is public ("huggingface-cli" was renamed to "hf")
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('kyutai/moshika-pytorch-bf16')"           # prefetch ~8 GB

# stripped config — REQUIRED: the fork's default arch includes the ARC conditioner (meta crash)
mkdir -p checkpoints/moshika-rag
uv run python -c "from huggingface_hub import hf_hub_download as d; import shutil; \
shutil.copy(d('kyutai/moshika-rag-pytorch-bf16','config.json'),'checkpoints/moshika-rag/config.json')"
uv run python scripts/strip_config.py

# nomarkers manifest is built on the box, not rsynced (absolute /home paths)
uv run python scripts/build_manifest.py \
  --wav-dir replay/retrieval_nomarkers --out-dir replay/retrieval_nomarkers --no-eval

# Box — verify Track A's manifests resolve (paths are absolute /home/leenatantawy/...)
for f in finetune/data/prepared_dialogue/train.jsonl replay/retrieval_nomarkers/train.jsonl; do
  echo "$f: $(wc -l < ~/moshi-finetune/$f) rows"; done
ls finetune/data/prepared_dialogue/*.wav | head -2
ls replay/retrieval_nomarkers/*.wav | head -2

# Box — 60-second smoke: config loads, one forward+backward
rm -rf runs/moshika_voice_max   # trainer refuses an existing run dir
sed 's/max_steps:.*/max_steps: 1/; s/do_ckpt:.*/do_ckpt: false/' \
  example/moshika_voice_max.yaml > /tmp/smoke.yaml
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train /tmp/smoke.yaml
rm -rf runs/moshika_voice_max   # smoke occupies the run dir; clear it so tonight's real launch starts clean

⚠️ If `uv sync` fails on setuptools or the box lacks uv, fall back to the ultra-box
pattern: create `~/fork-venv` and `pip install -e .` — see `docs/stage3_a100_runbook.md`
preamble for the interpreter-not-uv rule. Verified 2026-08-07: smoke passed, peak 35 GB
at rank 128 / batch 16 — no batch halving needed.

## 5. Session-liveness hardening (STT) — boot preflight + stall shrinking

Two failure classes wear the "timeout" costume. Keep them separate:

**HARD DROPS — Gradium hosted STT kills the session at 35–75 s.** Cured by local DSM STT
(launch lines already omit `--gradium-stt`). Only remaining risk is reintroduction. Gate
EVERY serve start (audition and demo):

```bash
# 1. no gradium/STT env may be set in the serving shell
env | grep -i -E "gradium|STT_URL" && echo "!! STT ENV PRESENT — unset before serving" || echo "env clean"
# 2. the serve command line must not contain --gradium-stt (read it before pressing enter)
# 3. after boot, confirm the local STT loaded (no gradium URL in the log):
grep -i -m2 -E "stt|asr" ~/serve.log
```

**STALLS — 2–18 s silences with the session alive**: the audible "timeout". Attack order:
1. **Checkpoint choice** — the audition harness scores `longest_stall` first-class; stalls
   outrank trigger rate in the verdict (decision of record).
2. **stt-wait step-down on the LOCKED checkpoint**: `--stt-wait-time` 2.0 → 1.0 → 0.5.
   The paper waits 0.5 s and shows an accuracy cliff at 1.5 s total retrieval delay; our
   2.0 s wait alone can be most of the excess. Three sessions per setting; keep the lowest
   whose grounded_rate matches 2.0's. Compare `round_trips` in the scorecards.
3. **Clear speech at the mic** — trigger rate tracks user-speech WER (paper Fig. 6a):
   directional/headset mic, quiet room, fully-phrased questions.

**Live recovery drill (rehearse once):** if a session ever drops mid-demo — private browser
window, reconnect, `grep -c "acquired slot" ~/serve.log` to confirm the slot is real, resume
the script at the current segment. Target < 20 s of stage time.

**Escalation (do NOT attempt on demo eve otherwise):** only if scorecards show retrieval
failing on garbled context (`Generated reference` lines quoting mis-transcribed questions),
consider an STT model swap — new moving part, measured risk only.

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

⚠️ If `uv sync` fails on setuptools or the box lacks uv, fall back to the ultra-box
pattern: create `~/fork-venv` and `pip install -e .` — see `docs/stage3_a100_runbook.md`
preamble for the interpreter-not-uv rule. Verified 2026-08-07: smoke passed, peak 35 GB
at rank 128 / batch 16 — no batch halving needed.

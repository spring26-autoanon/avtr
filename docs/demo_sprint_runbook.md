# Demo sprint runbook

## 1. Provision wb-gpu-training (Thu afternoon)

# Mac — push the repo state and the data Track A needs
rsync -avP --exclude runs --exclude runs_backup --exclude .git \
  --exclude finetune/data/datastereo --exclude finetune/data/prepared \
  /Users/leenatantawy/Downloads/capstone/moshi-finetune/ wb-gpu-training:~/moshi-finetune/

# Box — env (uv sync reads the fork-pinned pyproject)
cd ~/moshi-finetune && uv sync
uv run python -c "import moshi; print(moshi.__version__)"   # expect 0.2.13
huggingface-cli login    # must reach kyutai/moshika-pytorch-bf16
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('kyutai/moshika-pytorch-bf16')"           # prefetch ~8 GB

# Box — verify Track A's manifests resolve (paths are absolute /home/leenatantawy/...)
for f in finetune/data/prepared_dialogue/train.jsonl replay/retrieval_nomarkers/train.jsonl; do
  echo "$f: $(wc -l < ~/moshi-finetune/$f) rows"; done
ls finetune/data/prepared_dialogue/*.wav | head -2
ls replay/retrieval_nomarkers/*.wav | head -2

# Box — 60-second smoke: config loads, one forward+backward
sed 's/max_steps:.*/max_steps: 1/; s/do_ckpt:.*/do_ckpt: false/' \
  example/moshika_voice_max.yaml > /tmp/smoke.yaml
uv run torchrun --nproc-per-node 1 -m train /tmp/smoke.yaml

⚠️ If `uv sync` fails on setuptools or the box lacks uv, fall back to the ultra-box
pattern: create `~/fork-venv` and `pip install -e .` — see `docs/stage3_a100_runbook.md`
preamble for the interpreter-not-uv rule.

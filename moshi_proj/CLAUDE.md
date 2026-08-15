# CLAUDE.md — moshi-finetune → moshi-rag voice LoRA

## ⚠️ CURRENT STATUS — read `NEXT_STEPS.md` first (updated 2026-07-20)
`NEXT_STEPS.md` is the live source of truth; parts of the original plan below are
superseded. Quick state:
- **Voice cloning works.** First run (full LoRA on **plain moshika**, mainline moshi)
  cloned the voice, but **monologued / ignored the user** (turn-taking broken).
- **Root cause:** the data is monologue with a **silent user channel** (Option A), so the
  model learned the user never speaks. Voice AND turn-taking both live in the **attention**
  layers (proven by a freeze-attention experiment), so the fix is **data**, not a layer trick.
- **Can't fine-tune moshika-rag directly** — the trainer needs the moshi-rag fork, which
  crashes on the RAG ARC-encoder. **Path B is what we use:** train on plain moshika, apply to
  moshika-rag at serve time.
- **Verified:** training under the **fork** (moshi 0.2.13, now pinned in `pyproject.toml`) on
  plain moshika wraps attention and produces a moshika-rag-**compatible** adapter.
- **Next:** record 2-speaker **dialogue** data (her isolated left / partner right), rewrite
  `prepare_stereo.py`, retrain full LoRA under the fork, then serve (patch `get_lora_moshi`
  meta bug for the moshi-rag stack). Full detail + recording spec in `NEXT_STEPS.md`.

## Goal
Fine-tune a **LoRA adapter** that makes Moshi speak with the **voice and conversational
personality** of the recordings in `finetune/data/datastereo/`, and use that adapter with
**moshi-rag** (`kyutai/moshika-rag-pytorch-bf16`) — *not* plain base Moshi.

## Key facts & decisions
- **Base checkpoint = `kyutai/moshika-rag-pytorch-bf16`** (what moshi-rag serves). A LoRA is a
  delta welded to its base, so we train against these exact weights — NOT the repo default
  `kyutai/moshiko-pytorch-bf16`.
- **Compute:** training/annotation/serving run on a remote **GCP A100-80GB** (`a2-ultragpu-1g`).
  macOS is prep-only (no CUDA).
- **Data:** 9 single-speaker monologues, ~57.8 min total, consented/licensed voice. These 9 are
  the entire dataset.
- **Stereo strategy = Option A** (voice-left / silent-right).
- **Packaging = Option A** (apply `lora.safetensors` over the intact moshika-rag stack via a
  `--lora-weight` flag added to moshi-rag's server).

## How the pipeline works (grounding)
- Launch: `uv run torchrun --nproc-per-node 1 -m train <config>.yaml` (`train.py:360`).
- Model load: `loaders.CheckpointInfo.from_hf_repo(...)` (`train.py:128-134`) via `moshi_paths`
  keys `hf_repo_id / moshi_path / mimi_path / tokenizer_path / config_path` (`finetune/args.py:50-56`).
- **Critical:** `train.py:136-140` uses the loaded `config.json` as the model spec. moshika-rag's
  config carries RAG conditioners (`reference_with_time`, `first_speaker`, `fuser`, `rag_token_id`).
  `config_path` MUST be set or the default (non-RAG) architecture loads (`args.py:58-63`).
- Data: `data.train_data` → `.jsonl` of `{"path","duration"}`. Each `X.wav` needs a sibling
  `X.json` with `{"alignments": [[text,[start,end],"SPEAKER_MAIN"], ...]}` — an **unconditional
  open** (`interleaver.py:267-270`); training crashes without it. Left channel (0) = Moshi/main.
- Transcripts come from `annotate.py` (whisper_timestamped, reads channel 0, 16 kHz for Whisper).
- LoRA: only params named `"lora"` train (`wrapped_model.py:173-178`). Output with
  `save_adapters:true` → `runs/<run_dir>/checkpoints/checkpoint_*/consolidated/lora.safetensors`.

## Gotchas (why this repo needs care)
1. Source files are **mono** but the pipeline needs **stereo** → fixed by `scripts/prepare_stereo.py`.
2. Manifest + per-file transcripts are **required and were missing** → `scripts/build_manifest.py`
   + remote `annotate.py`.
3. LoRA is **base-specific** → train against moshika-rag, not moshiko.
4. Filenames had spaces/camelCase → normalized to snake_case during prep.

---

## Workflow

### Done locally (macOS, already run)
```bash
# mono -> stereo (voice-left/silent-right), snake_case rename, 24 kHz / PCM_24
python scripts/prepare_stereo.py --src finetune/data/datastereo --dst finetune/data/prepared
# manifests (train=8, eval=1 held-out movies.wav, all=9)
python scripts/build_manifest.py --wav-dir finetune/data/prepared --eval-file movies.wav
```
Output: `finetune/data/prepared/*.wav`, `train.jsonl`, `eval.jsonl`, `all.jsonl`.
Config: `example/moshika_rag_voice.yaml`.

### On the A100 box
**0. Setup**
```bash
uv sync
uv pip install whisper_timestamped submitit
huggingface-cli login                       # must reach kyutai/moshika-rag-pytorch-bf16
# fetch the RAG config so config_path resolves:
python -c "from huggingface_hub import hf_hub_download as d; import os,shutil; \
os.makedirs('checkpoints/moshika-rag',exist_ok=True); \
shutil.copy(d('kyutai/moshika-rag-pytorch-bf16','config.json'),'checkpoints/moshika-rag/config.json')"
```

**1. Transcribe (generates the required X.json next to each wav)**
```bash
python annotate.py finetune/data/prepared/train.jsonl   # use medium whisper (stereo)
python annotate.py finetune/data/prepared/eval.jsonl
# verify every wav has a sibling .json before training
```

**2. Smoke test the moshika-rag config (validates the conditioner risk)**
Copy the config, set `max_steps: 1`, `do_ckpt: false`, run once:
```bash
uv run torchrun --nproc-per-node 1 -m train example/moshika_rag_voice_smoke.yaml
```
- Loads + steps cleanly → use `moshika_rag_voice.yaml` as-is.
- Fails building conditioners → generate the stripped config and point `config_path` at it:
  ```bash
  python scripts/strip_config.py \
    --in  checkpoints/moshika-rag/config.json \
    --out checkpoints/moshika-rag/config.stripped.json
  ```
  It removes `conditioners`, `fuser`, `rag_token_id` and keeps all transformer/depformer dims.
  Then set `moshi_paths.config_path: "checkpoints/moshika-rag/config.stripped.json"`. The
  resulting LoRA is still valid over the full moshika-rag weights.

**3. Train**
```bash
uv run torchrun --nproc-per-node 1 -m train example/moshika_rag_voice.yaml
```
Watch train vs eval loss; take the earlier checkpoint if eval loss rises. Under-imprinted voice →
raise `lora.rank` (96/128) or `max_steps` before touching `lr`. Result:
`runs/moshika_rag_voice/checkpoints/checkpoint_*/consolidated/lora.safetensors`.

### In the moshi-rag repo — Packaging Option A (`--lora-weight`)
moshi-rag's `server_conditioner.py` CLI only exposes `--config`/`--moshi-weight`, but its vendored
`moshi/models/loaders.py` already supports `get_moshi_lm(..., lora_weights=, fuse_lora=)` and
`CheckpointInfo.lora_weights`. Add a flag and thread it through (verify exact call site against your
checkout — line numbers vary):
```python
# in parse_args():
parser.add_argument("--lora-weight", type=str, default=None,
                    help="LoRA safetensors to apply over the base weights")
# where CheckpointInfo is built / get_moshi is called, pass through:
#   checkpoint_info.lora_weights = Path(args.lora_weight) if args.lora_weight else None
# (or pass lora_weights=... / fuse_lora=True into get_moshi_lm)
```
Serve with moshi-rag's **original** config + weights (keeps all RAG conditioners live) plus the adapter:
```bash
python -m moshi.moshi.server_conditioner \
  --config      hf://kyutai/moshika-rag-pytorch-bf16/config.json \
  --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
  --conditioner reference_with_time \
  --lora-weight /path/to/lora.safetensors
```
**Fallback (Option B):** if editing moshi-rag is undesirable, merge the LoRA delta into a copy of
moshika-rag's `model.safetensors` (preserving all conditioner tensors) and serve that with
`--moshi-weight`. Do NOT reuse moshi-finetune's `save_adapters:false` output for this if you trained
with a stripped config — it would omit the conditioner weights.

## Verification checklist
1. Every prepared wav: 2ch / 24 kHz / has non-empty `.json` alignments / listed in a manifest.
2. Smoke run loads moshika-rag and completes forward+backward (or stripped-config fallback does).
3. Training loss decreases; `lora.safetensors` written.
4. moshi-rag boots with the adapter and stays responsive (RAG + turn-taking intact).
5. Live conversation: voice/personality matches the source; not monologuing over the user. If it
   monologues, retrain lighter (fewer steps / lower rank).

## File map
- `scripts/prepare_stereo.py` — mono→stereo + rename (done)
- `scripts/build_manifest.py` — manifests (done)
- `scripts/strip_config.py` — RAG-conditioner-free config fallback (run on A100 only if smoke test fails)
- `example/moshika_rag_voice.yaml` — training config
- `finetune/data/datastereo/` — ORIGINAL mono files (never modify)
- `finetune/data/prepared/` — stereo wavs + manifests (+ `.json` transcripts after step 1)
- Plan of record: `~/.claude/plans/joyful-noodling-conway.md`

# A100 runbook — dialogue data prep → fork Path-B retrain

Prereqs: the 5 stereo WAVs exist locally in `finetune/data/prepared_dialogue/`
(produced by `scripts/pair_dialogue_stereo.py`). Do these ON THE A100 box.

## 1. Copy the WAVs to the box
From the Mac:
```bash
rsync -avP finetune/data/prepared_dialogue/*.wav \
  wb-gpu-training:~/moshi-finetune/finetune/data/prepared_dialogue/
```

## 2. Build manifests (absolute paths — must run on the box)
```bash
cd ~/moshi-finetune
uv run python scripts/build_manifest.py \
  --wav-dir finetune/data/prepared_dialogue \
  --out-dir finetune/data/prepared_dialogue \
  --eval-file danielle_joshuarhodes_eval.wav
```
Expect: `train.jsonl` = 4 files, `eval.jsonl` = 1 file, `all.jsonl` = 5 files.

## 3. Transcribe channel 0 (Danielle) — generates the required X.json
`annotate.py` reads channel 0, needs CUDA, and uses `-l` for local (no Slurm).
```bash
CUDA_VISIBLE_DEVICES=0 uv run python annotate.py \
  finetune/data/prepared_dialogue/all.jsonl -l --whisper_model medium --lang en
```
The 86-min file is within annotate.py's 4-hour guard. Each `X.wav` gets a
sibling `X.json` with `{"alignments": [[text,[start,end],"SPEAKER_MAIN"], ...]}`.

## 4. Verify every wav has a non-empty transcript
```bash
for w in finetune/data/prepared_dialogue/*.wav; do
  j="${w%.wav}.json"
  if [ ! -s "$j" ]; then echo "MISSING/EMPTY: $j"; fi
done
echo "check complete"
```
Expect: no "MISSING/EMPTY" lines. (Training crashes without a sibling .json —
interleaver.py:267-270.)

## 5. Train (fork stack, full LoRA on plain moshika)
```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train \
  example/moshika_rag_voice.yaml
```
Watch train vs eval loss. Take the earlier checkpoint if eval loss climbs.
Result: `runs/moshika_rag_voice/checkpoints/checkpoint_*/consolidated/lora.safetensors`.

> Data prep ends here. Serving the adapter on moshika-rag is Part 3 of NEXT_STEPS.
> Hyperparameter tuning for the larger (~2.4 hr) dataset — e.g. raising `max_steps`
> beyond the monologue-era 400 — is a separate follow-up.

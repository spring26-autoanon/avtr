# A100 runbook — Stage 3 retrieval data + train

Everything GPU-side runs on **`wb-gpu-a1ultra2g`** through **`~/fork-venv/bin/python`**.
That box has no repo `.venv`, so `uv run` / `uv run --no-sync` will create an empty one and
fail on imports — call the interpreter directly.

Mac-side steps use system `python3` (not `.venv`, which lacks soundfile).

Prereq: `finetune/data/prepared_retrieval/danielle_joshuarhodes.wav` exists locally —
84.65 min, 2ch, 24 kHz, PCM_24, Danielle left / Joshua right, produced by
`scripts/pair_dialogue_stereo.py --no-eval`.

## 1. Upload the master and the changed scripts (Mac)
```bash
cd /Users/leenatantawy/Downloads/capstone/moshi-finetune
rsync -avP annotate.py train.py wb-gpu-a1ultra2g:~/moshi-finetune/
rsync -avP scripts/ wb-gpu-a1ultra2g:~/moshi-finetune/scripts/
rsync -avP finetune/data/reference_io.py finetune/data/reference_injection.py \
           finetune/data/interleaver.py \
  wb-gpu-a1ultra2g:~/moshi-finetune/finetune/data/
rsync -avP example/moshika_rag_stage3.yaml wb-gpu-a1ultra2g:~/moshi-finetune/example/
rsync -avP --progress finetune/data/prepared_retrieval/danielle_joshuarhodes.wav \
  wb-gpu-a1ultra2g:~/moshi-finetune/finetune/data/prepared_retrieval/
```

## 2. Transcribe both channels (box)
```bash
cd ~/moshi-finetune

# annotate.py imports submitit at module scope even in --local mode
~/fork-venv/bin/python -c "import submitit" 2>/dev/null \
  || uv pip install --python ~/fork-venv/bin/python submitit

~/fork-venv/bin/python scripts/build_manifest.py \
  --wav-dir finetune/data/prepared_retrieval \
  --out-dir finetune/data/prepared_retrieval \
  --no-eval
```
Expect `all.jsonl : 1 files, 84.65 min`.

```bash
loginctl enable-linger "$USER"

CUDA_VISIBLE_DEVICES=0 setsid nohup ~/fork-venv/bin/python annotate.py \
  finetune/data/prepared_retrieval/all.jsonl -l --whisper_model medium --lang en \
  --channel 0 --out-suffix .json > ~/annotate_ch0.log 2>&1 &
```
Wait for it (`tail -f ~/annotate_ch0.log`) before starting the second — they would otherwise
contend for the GPU.
```bash
CUDA_VISIBLE_DEVICES=0 setsid nohup ~/fork-venv/bin/python annotate.py \
  finetune/data/prepared_retrieval/all.jsonl -l --whisper_model medium --lang en \
  --channel 1 --out-suffix .ch1.json > ~/annotate_ch1.log 2>&1 &
```

Two traps. `annotate.py` skips a file whose output already exists, so if `.ch1.json` is
somehow already present the second pass silently does nothing — check its mtime, not just
that it exists. And this is a **Vertex AI Workbench** instance whose 180-minute idle timer
watches Jupyter kernels, not terminals, so it can shut down mid-transcription if you are
otherwise idle.

## 3. Pull the transcripts back (Mac)
```bash
rsync -avP 'wb-gpu-a1ultra2g:~/moshi-finetune/finetune/data/prepared_retrieval/*.json' \
  finetune/data/prepared_retrieval/
```

## 4. Segment, filter and cut (Mac)
```bash
source .env                     # GEMINI_API_KEY, rotated
python3 scripts/segment_retrieval_audio.py \
  --ch0 finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
  --ch1 finetune/data/prepared_retrieval/danielle_joshuarhodes.ch1.json \
  --out replay/retrieval_segments.jsonl

python3 scripts/filter_retrieval_segments.py \
  --in replay/retrieval_segments.jsonl \
  --out replay/retrieval_segments.filtered.jsonl \
  --dropped replay/retrieval_dropped.jsonl

python3 scripts/cut_retrieval_clips.py \
  --master finetune/data/prepared_retrieval/danielle_joshuarhodes.wav \
  --alignments finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
  --segments replay/retrieval_segments.filtered.jsonl \
  --outdir replay/retrieval --manifest replay/retrieval_manifest.jsonl
```

**Record the `her turns:` line** from the segmenter. Its `retrieval_share` sets
`RAG_TOKEN_WEIGHT` in step 7.

Audition two or three clips before uploading; each should open on a question, not mid-sentence:
```bash
open replay/retrieval/$(ls replay/retrieval | head -1)
```

## 5. Upload the clips (Mac)
```bash
rsync -avP replay/retrieval/ wb-gpu-a1ultra2g:~/moshi-finetune/replay/retrieval/
```

## 6. Encode references, build the manifest, run the gate (box)
The `:8001` reference-encoder service must be up on GPU 1
(`CUDA_VISIBLE_DEVICES=1 ... --cuda-device 0`, so attn_bias and the model share a device).
`build_manifest.py` must run here because manifest paths are absolute and machine-specific.
```bash
cd ~/moshi-finetune
~/fork-venv/bin/python scripts/precompute_references.py \
  --manifest replay/retrieval_manifest.jsonl --audiodir replay/retrieval

~/fork-venv/bin/python scripts/build_manifest.py \
  --wav-dir replay/retrieval --out-dir replay/retrieval --no-eval

~/fork-venv/bin/python scripts/verify_retrieval_clips.py --clips replay/retrieval
```
The gate must print `gate passed.` If it reports marker/tensor mismatches, send the output
back rather than training around them — a mismatch means a retrieval turn would be trained
with no conditioning at all.

## 7. Smoke test, then train (box)
The multi-`⟨ret⟩` injection is unit-tested, but nothing on the Mac can load moshi, so prove
it on the real stack for two steps first. Give the smoke run its own `run_dir` — the real
config omits `overwrite_run_dir`, so a leftover directory would make the real run refuse to
start.
```bash
cd ~/moshi-finetune
sed -e 's/^max_steps: .*/max_steps: 2/' \
    -e 's/^do_ckpt: .*/do_ckpt: false/' \
    -e 's#^run_dir: .*#run_dir: "/tmp/stage3_smoke"#' \
    example/moshika_rag_stage3.yaml > /tmp/stage3_smoke.yaml
printf 'overwrite_run_dir: true\n' >> /tmp/stage3_smoke.yaml

RAG_TOKEN_WEIGHT=15 CUDA_VISIBLE_DEVICES=0 \
  ~/fork-venv/bin/torchrun --nproc-per-node 1 -m train /tmp/stage3_smoke.yaml
```
Expect two steps with a finite loss. A crash inside `build_reference_condition`, or a
`reference_with_time` shape error, means the Task 9 change needs fixing before the real run.

Then the real run. Set `RAG_TOKEN_WEIGHT` from step 4: **15** if smalltalk+decline are at
least 30% of her turns, otherwise **8-10**.
```bash
RAG_TOKEN_WEIGHT=15 CUDA_VISIBLE_DEVICES=0 setsid nohup \
  ~/fork-venv/bin/torchrun --nproc-per-node 1 -m train example/moshika_rag_stage3.yaml \
  > ~/train_stage3.log 2>&1 &
```
Checkpoints land in
`runs/moshika_rag_stage3/checkpoints/checkpoint_*/consolidated/lora.safetensors`.
`do_eval` is off — eval_loss goes NaN below `batch_size` windows — so pick by ear.

## 8. Audition against the success criterion
Judge switching, not either mode alone:

1. **Into retrieval:** several turns of chit-chat, *then* a factual question answerable only
   from the corpus. This is the case trial 4 failed — chit-chat primed persona mode and it
   confabulated.
2. **Back out:** after a successful retrieval, return to casual talk; she must not stay in
   formal-assistant register (`checkpoint_000600`'s failure).
3. **No spurious triggering:** a wholly casual conversation should emit no `⟨ret⟩`.
4. **Voice:** she should sound like herself in both modes.

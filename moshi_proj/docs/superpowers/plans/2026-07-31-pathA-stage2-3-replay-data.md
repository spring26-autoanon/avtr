# Path A Stage 2/3 — Replay Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`). Runs on the 2-GPU box `wb-gpu-a1ultra2g` (fork env `~/fork-venv`) + the Mac, guided live. Verification is a **runtime checkpoint** (renders / injects / retrieves), not a unit test. Fork internals aren't visible from the Mac → **INSPECT ON BOX** steps probe first, then act.

**Goal:** Restore retrieval (the "emit ⟨ret⟩ → speak from the reference" behavior that voice-only Path A killed) by mixing synthetic RAG replay data into training, keeping the Danielle voice.

**Architecture:** Generate grounded QA → conversationalize → render stereo (her voice + partner TTS) → precompute `reference_with_time` conditioning via the `:8001` ARC encoder → place ⟨ret⟩ → faithfulness-filter → mix with voice-dialogue → retrain on moshika-rag.

**Tech stack:** fork moshi 0.2.13 / torch 2.9.1 (`~/fork-venv`), moshika-rag, `checkpoint_700` (voice adapter, on the Mac + box), Gemini/Claude for QA-gen, Gradium (STT + voice-clone fallback), the `:8001` `server_conditioner`.

## Global Constraints

- Two spikes (Tasks 1–2) are **hard gates** — do NOT generate a large dataset (Task 4+) until both pass. If a spike fails, stop and re-scope.
- Idle timeout: set Workbench "Time of inactivity before shutdown" to **1440** (the UI field) before any long run, so training/rendering isn't idle-killed. **Stop the 2×A100 box manually when done** (cost).
- Long jobs: `loginctl enable-linger "$USER"` + `setsid nohup … &`.
- Fork-code edits (interleaver, config): keep `.orig` backups; they're box-local (site-packages), note them in the writeup.
- The Stage-1 stripped config `checkpoints/moshika-rag/config.norefenc.json` stripped `reference_with_time` from **both** `conditioners` and `fuser`. Replay training needs a **different** config: `reference_with_time` in the **fuser** but **not** as a conditioner module (Task 2 determines the exact shape).

---

### Task 1 — SPIKE: `checkpoint_700` as a text→her-voice TTS (gate)

**Files:** Create `~/moshi-finetune/scratch/tts_probe.py` (or reuse a fork inference entrypoint).

- [ ] **Step 1: INSPECT ON BOX — does the fork expose forced-text generation / TTS?**
```bash
FORKPKG=$(~/fork-venv/bin/python -c "import moshi,os;print(os.path.dirname(moshi.__file__))")
ls "$FORKPKG" | grep -iE "tts|infer|generate|run_"
grep -n "def main\|argparse\|forced\|text\|prompt\|tts\|LMGen\|generate" "$FORKPKG/run_inference.py" 2>/dev/null | head -40
grep -rn "class LMGen\|def step\|forced_text\|text_or_audio\|def generate" "$FORKPKG/models/lm.py" "$FORKPKG"/models/*.py | head -30
```
Record: is there an entrypoint that takes text and forces the inner-monologue stream? Paste findings.

- [ ] **Step 2: Prototype rendering one sentence in her voice.** Using the entrypoint/LMGen found in Step 1, load moshika-rag + `checkpoint_700`'s `lora_serve.safetensors` (the proven serve recipe), force the text stream to a target sentence (e.g. "Hello, this is a test of my voice."), generate audio, decode with Mimi, write a wav. (Exact code depends on Step-1 API; I'll hand it to you from the findings.)

- [ ] **Step 3: Listen (gate).** scp the wav to the Mac and play it.
  - **PASS:** clean, recognizably-Danielle audio for arbitrary text → this is the assistant-turn renderer. Proceed.
  - **FAIL** (no forced-text API, or bad audio): fall back to **Gradium voice-clone** for her turns — clone her voice in the Gradium console from a few `danielle_*` clips, render assistant turns via the Gradium TTS API. Record which route.

**Checkpoint:** a working "text → Danielle audio" renderer (Moshi-forced-text OR Gradium clone).

---

### Task 2 — SPIKE: inject a precomputed `reference_with_time` tensor into a training step (THE gate)

**Files:** Create `~/moshi-finetune/scratch/cond_inject_probe.py`; possibly edit fork `moshi/conditioners`/`fuser` + `finetune/data/interleaver.py` (INSPECT first).

- [ ] **Step 1: INSPECT ON BOX — how serve injects the precomputed reference tensor.** Understand the mechanism we must mirror:
```bash
FORKPKG=$(~/fork-venv/bin/python -c "import moshi,os;print(os.path.dirname(moshi.__file__))")
grep -rn "def get_condition_tensors\|streaming_sum\|prepend\|ConditionType\|class ConditionFuser\|def forward" "$FORKPKG/conditioners"/*.py "$FORKPKG"/inference_utils/utils.py | head -50
sed -n '/def get_condition_tensors/,/return/p' "$FORKPKG/inference_utils/utils.py" | head -60
grep -rn "reference_with_time\|streaming_sum\|condition_tensors\|ConditionTensors\|prepare" "$FORKPKG"/inference_utils/channel.py | head -30
```
Record: the type/shape of the `reference_with_time` streaming-sum tensor, how `get_condition_tensors` builds a `condition_tensors` object the model forward consumes, and how `channel.py` swaps in a new reference tensor mid-stream.

- [ ] **Step 2: Precompute one reference tensor via the `:8001` encoder.** Start the encoder (Task-3 serve block), POST a reference string to `/embed`, save the returned tensor:
```bash
curl -s -X POST http://localhost:8001/embed -H 'content-type: application/json' \
  -d '{"text":"Reference: The 2026 World Cup was held in North America."}' -o ~/ref_tensor.safetensors
~/fork-venv/bin/python -c "from safetensors.torch import load_file; t=load_file('$HOME/ref_tensor.safetensors'); print({k:v.shape for k,v in t.items()})"
```
(Adjust the endpoint/payload/response format to what the `:8001` service actually expects — from `server_conditioner.py`.)

- [ ] **Step 3: INSPECT the trainer's conditioning path + interleaver.**
```bash
grep -n "condition_tensors\|condition_provider\|condition_attributes\|prepare" ~/moshi-finetune/train.py
sed -n '250,290p' ~/moshi-finetune/finetune/data/interleaver.py   # Sample(codes, text_conditions=...)
grep -rn "text_conditions\|condition\|ConditionAttributes\|prepare" ~/moshi-finetune/finetune/data/*.py
```
Determine how a per-example condition flows from the `.json` → `Sample` → `batch` → `model(codes, condition_tensors=…)`.

- [ ] **Step 4: Build the model keeping `reference_with_time` in the FUSER but not as a module + inject the tensor into ONE forward/backward.** In `cond_inject_probe.py`: build moshika-rag with a config that keeps `reference_with_time` in `fuser.streaming_sum` but removes only the ARC **conditioner module** (so no meta crash), construct a `condition_tensors` object carrying the precomputed tensor (shape/type from Step 1), run a single `model(codes=<dummy>, condition_tensors=…)` + `.backward()`.
  - **PASS:** no crash, the model consumes the tensor, loss is finite and *changes* when the tensor changes (vs zeros). → the replay format = audio + alignments + this tensor + ⟨ret⟩ position; the interleaver gets a small change to carry it (Task 5).
  - **FAIL:** record where it breaks. Fallback options (in order): (a) minimal interleaver/fuser surgery to accept the raw tensor; (b) the heavier "ARC-in-trainer" path — un-strip `reference_with_time`, patch the meta autocast, feed reference *text*. Re-scope if neither is tractable.

**Checkpoint:** a training step that consumes a precomputed `reference_with_time` tensor, with a written recipe for how replay examples carry it. **This is the make-or-break of the whole stage.**

---

### Task 3 — Bring up the `:8001` reference encoder (dependency for Tasks 2 & 5)

- [ ] **Step 1:** Launch it (GPU 1), same as the serving stack:
```bash
cd ~/moshi-finetune
CUDA_VISIBLE_DEVICES=1 setsid nohup ~/fork-venv/bin/python -m moshi.server_conditioner \
  --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \
  --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
  --conditioner reference_with_time --cuda-device 0 --port 8001 > ~/refenc.log 2>&1 &
echo "PID $!"; sleep 40; tail -5 ~/refenc.log
```
**Checkpoint:** `:8001` serving; a test POST returns a tensor (Task 2 Step 2).

---

### Task 4 — Generate grounded QA + conversationalize (parallelizable; run while spikes happen)

**Files:** Create `scripts/gen_replay_qa.py` (Mac), output `replay/qa.jsonl`.

- [ ] **Step 1:** Script an LLM (Claude/Gemini) to emit **diverse general QA triples**: `{question, reference (~50 words, the "Reference:" line), answer (faithful, her-style)}`. Include **hard/negative** cases: irrelevant reference→decline; no-retrieval smalltalk turns; multi-reference pick-right. Target a **few hundred** for the Stage-2b batch.
- [ ] **Step 2:** **Conversationalize** each into a short 2-speaker dialogue (user asks → assistant answers from the reference), with natural turn-taking, marking where the ⟨ret⟩ fires (before the assistant's grounded answer).
- [ ] **Step 3: Faithfulness filter** — LLM-judge each answer against its reference; drop unsupported ones.
**Checkpoint:** `replay/qa.jsonl` with a few hundred filtered, conversationalized examples.

---

### Task 5 — Render + assemble the replay dataset (needs Task 1 + Task 2 outcomes)

- [ ] **Step 1:** Render each dialogue to **stereo**: assistant (ch0) via the Task-1 renderer (her voice), user (ch1) via a varied TTS; time-aligned → `replay/<id>.wav`.
- [ ] **Step 2:** **Precompute conditioning** — POST each reference to `:8001` → save the `reference_with_time` tensor per example (Task-2 format).
- [ ] **Step 3:** **Alignments + ⟨ret⟩** — align the assistant channel → `SPEAKER_MAIN` json (Whisper, as with dialogue data); record the ⟨ret⟩/rag_token frame position.
- [ ] **Step 4:** Package each example as {stereo wav, alignments json, reference tensor, ⟨ret⟩ position} in the format Task-2 proved the trainer can ingest; write a manifest.
**Checkpoint:** a small replay dataset the trainer can load.

---

### Task 6 — Stage 2b: retrain on dialogue + small replay batch, validate retrieval restored

- [ ] **Step 1:** Config = the Task-2 config (`reference_with_time` in fuser, not module) + train_data = dialogue manifests **+** replay manifest (mixed). rank 64, ~800 steps, disconnect-proof, idle timeout 1440.
- [ ] **Step 2:** Train (Jupyter cell or terminal), watch loss.
- [ ] **Step 3: Serve + probe retrieval.** Re-pad the checkpoint, relaunch the 2 services, ask a factual question. **Success = the model emits `[RAG] model emitted RAG token` + a `[Remote Encoder]` call + an answer reflecting the reference** (vs Stage-1's "answered from its own head"), with voice + turn-taking intact, no screech.
**Checkpoint:** retrieval demonstrably restored on a small batch = the mechanism works.

---

### Task 7 — Stage 3: scale + final retrain

- [ ] **Step 1:** Expand replay diversity (more topics/hard cases) via Task 4–5.
- [ ] **Step 2:** Retrain a **fresh** adapter on the voice-dialogue + replay **mix** (~50/50 start; tune the ratio = voice-vs-retrieval dial).
- [ ] **Step 3:** Judge **both** voice and retrieval live; back up the keeper adapter to the Mac; record outcome in `NEXT_STEPS.md` + memory.
**Checkpoint:** the Stage-3 deliverable — Danielle voice + turn-taking + no screech + working retrieval.

---

## Notes for the executor

- **Tasks 1 and 2 gate everything** — run them first; Task 4 (QA-gen) can run in parallel since it needs neither. Do not render a big dataset (Task 5) until both spikes pass.
- Task 2 is the genuine research risk; budget for it not resolving cleanly on the first pass, and use its fallbacks before escalating to the heavy ARC-in-trainer path.
- The fork internals drive every INSPECT step — probe, paste, then act; never guess shapes/APIs.
- Keep the 2×A100 idle timeout at 1440 during runs and **stop the box when done**.

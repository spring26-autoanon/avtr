# Phase 2 — Retrieval QA + persona on moshika-rag (papers corpus) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to work this task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a **box-side integration runbook** run on the 2-GPU box `wb-gpu-a1ultra2g`, guided live (the fork source + services aren't reachable from the mac). Verification at each step is a **runtime checkpoint** (loads / serves / retrieves / sounds right), not a unit test. Several steps are **INSPECT-ON-BOX** — read the fork code first, then act; never guess.

**Goal:** Stand up the moshi-rag 3-service stack so Danielle's voice adapter answers questions grounded in the two Moshi papers, with a consistent persona.

**Architecture:** gemma-3-27b (vLLM) on GPU 0 = retrieval LLM; Moshi + `lora_padded` adapter + reference/ARC encoder on GPU 1; Gradium STT transcribes the user. ⟨ret⟩ token → transcript → retrieve from papers → gemma reference → ARC encoder → grounded spoken answer.

**Tech Stack:** moshi-rag fork (moshi 0.2.13 / torch 2.9.1+cu128), vLLM + gemma-3-27b-it, `meta-llama/Llama-3.2-3B` (ARC dep), Gradium STT, `uv`.

## Global Constraints

- Run on `wb-gpu-a1ultra2g` (2×A100-80GB). Pin GPUs explicitly with `CUDA_VISIBLE_DEVICES` per service (gemma→0, Moshi/encoder→1).
- Long-running services: launch detached — `loginctl enable-linger "$USER"` + `CUDA_VISIBLE_DEVICES=… setsid nohup … > ~/<svc>.log 2>&1 &` (the Phase-1 lesson: systemd SIGTERMs the user slice on full logout otherwise).
- Reuse the Phase-1 serving recipe verbatim: `lora_padded.safetensors` + `get_moshi(fuse_lora=True, lm_kwargs_overrides={"lora":True,"lora_rank":64,"lora_scaling":2.0})`.
- Path B overlay only. No Path A / replay training, no Route B persona data (both out of scope).
- Where the fork code drives the exact command, the step says **INSPECT ON BOX** and gives the probe first.
- Keep `.orig` backups of any fork file edited.

---

### Task 1: Provision the 2-GPU box (fork env + adapter + access)

**Files:** none (environment). Uses the pushed branch for scripts.

- [ ] **Step 1: Clone the repo (branch) onto the box**

```bash
ssh wb-gpu-a1ultra2g
git clone https://github.com/spring26-autoanon/moshi_proj.git ~/moshi-finetune
cd ~/moshi-finetune && git checkout dialogue-data-prep
```
Checkpoint: repo present, on branch `dialogue-data-prep` (has `scripts/pad_adapter_for_ragbase.py`).

- [ ] **Step 2: Build the fork env (mirror Phase 1)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
uv venv ~/fork-venv --python 3.12
VIRTUAL_ENV=~/fork-venv uv pip install --python ~/fork-venv/bin/python \
  "torch==2.9.1" \
  "moshi @ git+https://github.com/kyutai-labs/moshi-rag.git#subdirectory=moshi" \
  "sphn>=0.2.0,<0.3.0"
~/fork-venv/bin/python -c "import moshi,torch;print(moshi.__version__,torch.__version__)"
```
Checkpoint: prints `0.2.13 2.9.1+cu128`.

- [ ] **Step 3: Verify GPUs + a real CUDA op**

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
~/fork-venv/bin/python -c "import torch; print((torch.randn(2048,device='cuda')@torch.randn(2048,2048,device='cuda')).sum().item(), torch.cuda.device_count())"
```
Checkpoint: **two** A100-80GB rows; CUDA op prints a number and device count `2`.

- [ ] **Step 4: HF login + copy the adapter from the mac**

```bash
~/fork-venv/bin/huggingface-cli login          # paste token
```
From the **mac**:
```bash
ssh wb-gpu-a1ultra2g 'mkdir -p ~/moshi-finetune/runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated'
rsync -avP ~/Downloads/capstone/moshi-finetune/adapters/checkpoint_000700/ \
  wb-gpu-a1ultra2g:~/moshi-finetune/runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/
```
Checkpoint: `lora.safetensors`, `lora_padded.safetensors`, `config.json` on the box.

- [ ] **Step 5: Confirm the padded adapter loads on THIS box**

```bash
CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python - <<'PY'
import torch; from pathlib import Path
from moshi.models.loaders import CheckpointInfo
P=Path.home()/"moshi-finetune/runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora_padded.safetensors"
m=CheckpointInfo.from_hf_repo("kyutai/moshika-rag-pytorch-bf16", lora_weights=P).get_moshi(
   device="cuda", dtype=torch.bfloat16, fuse_lora=True,
   lm_kwargs_overrides={"lora":True,"lora_rank":64,"lora_scaling":2.0})
print("LOAD OK", sum(p.numel() for p in m.parameters()))
PY
```
Checkpoint: `LOAD OK 10730668080` (matches Phase 1). If `lora_padded` is somehow missing/stale, regenerate: `~/fork-venv/bin/python scripts/pad_adapter_for_ragbase.py --lora <…>/lora.safetensors --out <…>/lora_padded.safetensors`.

---

### Task 2: Learn moshi-rag's retrieval pipeline + index the two papers (THE CRUX)

**Files:**
- Create: `~/moshi-finetune/scratch/extract_papers.py` (PDF → chunked text)
- Create: whatever corpus/profile artifact Step 2 reveals moshi-rag needs.

- [ ] **Step 1: INSPECT ON BOX — how retrieval + corpus are configured**

```bash
FORKPKG=$(~/fork-venv/bin/python -c "import moshi,os;print(os.path.dirname(moshi.__file__))")
ls "$FORKPKG/reference" "$FORKPKG/inference_utils"
echo "== retrieval profiles =="; sed -n '1,200p' "$FORKPKG/inference_utils/retrieval_profiles.py"
echo "== reference generator =="; sed -n '1,160p' "$FORKPKG/reference/llm_reference_generator.py"
grep -rn "corpus\|index\|embed\|retriev\|profile\|documents\|passage\|faiss\|bm25\|\.jsonl\|load_retrieval_env" "$FORKPKG/reference" "$FORKPKG/inference_utils" | head -60
```
Record: how a **retrieval profile** points at documents, the **corpus format** (jsonl? embeddings? raw docs handed to the LLM?), and whether retrieval is vector-based or LLM-over-docs. **This determines Step 3.** Paste findings; we design the corpus artifact together.

- [ ] **Step 2: Extract the two papers to chunked text**

```bash
cat > ~/moshi-finetune/scratch/extract_papers.py <<'PY'
import json, re, sys
from pathlib import Path
from pypdf import PdfReader   # uv pip install pypdf into fork-venv if missing
OUT = Path.home()/"moshi-finetune/scratch/papers_corpus.jsonl"
docs = []
for pdf in ["base-moshi-paper.pdf", "moshi-rag-paper.pdf"]:
    r = PdfReader(str(Path.home()/"moshi-finetune"/pdf))
    for pi, pg in enumerate(r.pages):
        txt = re.sub(r"\s+", " ", (pg.extract_text() or "")).strip()
        # ~1000-char chunks with light overlap
        for ci in range(0, len(txt), 800):
            chunk = txt[ci:ci+1000]
            if len(chunk) > 120:
                docs.append({"id": f"{pdf}:{pi}:{ci}", "source": pdf, "page": pi+1, "text": chunk})
with open(OUT, "w") as f:
    for d in docs: f.write(json.dumps(d) + "\n")
print(f"wrote {len(docs)} chunks -> {OUT}")
PY
~/fork-venv/bin/pip install pypdf -q 2>/dev/null; ~/fork-venv/bin/python ~/moshi-finetune/scratch/extract_papers.py
```
Checkpoint: `papers_corpus.jsonl` with a few hundred chunks. (Chunk size/overlap adjustable once Step 1 tells us the expected format.)

- [ ] **Step 3: Build the retrieval profile/index over the corpus (format per Step 1)**

Adapt the corpus into moshi-rag's expected form — a retrieval profile pointing at
`papers_corpus.jsonl`, or an embedded index, or the doc set the reference LLM reads.
**Decision branch:** if moshi-rag has **no clean custom-corpus hook**, add a minimal
retrieval profile (or a thin retrieval shim: embed chunks + top-k, hand results to
`LLMReferenceGenerator`). Exact commands come from Step 1 findings.
Checkpoint: a test query ("What is Inner Monologue in Moshi?") returns a relevant
chunk / reference from the papers.

---

### Task 3: Bring up the reference-encoder + retrieval-LLM services

**Files:** none (service launches).

- [ ] **Step 1: Reference-encoder service (GPU 1)**

INSPECT the flags first: `grep -n "add_argument\|conditioner\|port\|reference" "$FORKPKG/server_conditioner.py" | head`. Then (adjust flags to match):
```bash
cd ~/moshi-finetune
CUDA_VISIBLE_DEVICES=1 setsid nohup ~/fork-venv/bin/python -m moshi.server_conditioner \
  --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \
  --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
  --conditioner reference_with_time --port 8001 \
  > ~/refenc.log 2>&1 &
echo "PID $!"; sleep 30; tail -20 ~/refenc.log
```
Checkpoint: service listening on :8001, no traceback.

- [ ] **Step 2: gemma-3-27b via vLLM (GPU 0)**

```bash
VIRTUAL_ENV=~/fork-venv uv pip install --python ~/fork-venv/bin/python vllm 2>&1 | tail -3
CUDA_VISIBLE_DEVICES=0 setsid nohup ~/fork-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-27b-it --port 8002 --gpu-memory-utilization 0.9 \
  > ~/vllm.log 2>&1 &
echo "PID $!"; sleep 90; tail -30 ~/vllm.log
curl -s localhost:8002/v1/models | head
```
Checkpoint: vLLM serves `gemma-3-27b-it` on :8002 (the `/v1/models` curl returns it). **Decision branch:** if gemma-27b OOMs or is gated/unavailable, fall back to a smaller instruct model on :8002 (retrieval quality drops but pipeline stands).

---

### Task 4: Source + wire the Gradium STT

**Files:** none (env vars).

- [ ] **Step 1: Obtain the Gradium key (USER, external)** and set env

INSPECT what the server reads: `grep -rn "STT_URL\|STT_API\|gradium\|GRADIUM\|getenv" "$FORKPKG" | head`. Then export the exact vars it expects:
```bash
export STT_URL="<gradium endpoint>"; export STT_API_KEY="<key>"   # names per the grep
```
Checkpoint: a quick STT smoke (per the client in `$FORKPKG`) transcribes a test clip. **Blocker:** if no key yet, Steps 5–7 can boot without STT to check voice/turn-taking, but retrieval needs it — note as pending.

---

### Task 5: Wire `--lora-weight` into `server.py` + boot the main server (GPU 1)

**Files:** Modify (INSPECT first): fork `moshi/server.py` (arg parser + LM load site).

- [ ] **Step 1: INSPECT ON BOX — find the LM load site**

The Phase-1 grep showed `server.py` builds the LM in a helper, not inline. Find it:
```bash
grep -rn "get_moshi\b\|CheckpointInfo\|from_hf_repo\|lm_kwargs_overrides\|fuse_lora\|lora" "$FORKPKG/server.py" "$FORKPKG"/inference_utils/*.py | head -40
```
Record the exact call site + module.

- [ ] **Step 2: Add `--lora-weight` and thread the Phase-1 overrides**

Backup then edit: `cp "$FORKPKG/server.py" "$FORKPKG/server.py.orig"`. Add `--lora-weight` to the parser, and at the LM build pass `lora_weights=<padded>` + `fuse_lora=True` + `lm_kwargs_overrides={"lora":True,"lora_rank":64,"lora_scaling":2.0}` (mirroring the proven `serve_check.py` load). Edit the helper if that's where the build lives.

- [ ] **Step 3: Boot the main server (GPU 1), env pointing at services 3 & 4**

```bash
cd ~/moshi-finetune
CUDA_VISIBLE_DEVICES=1 REFERENCE_ENCODER_URL=http://localhost:8001 \
  RETRIEVAL_LLM_URL=http://localhost:8002/v1 STT_URL="$STT_URL" STT_API_KEY="$STT_API_KEY" \
  setsid nohup ~/fork-venv/bin/python -m moshi.server \
  --lora-weight runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora_padded.safetensors \
  --port 8998 > ~/mainserver.log 2>&1 &
echo "PID $!"; sleep 40; tail -40 ~/mainserver.log
```
(Env var **names** per the Step-1/Task-4 grep — adjust.) Checkpoint: server accepts a WebSocket connection with the adapter fused and services wired.

---

### Task 6: Persona (Route A — inference-time grounding)

**Files:** Create: `~/moshi-finetune/scratch/persona.txt` (Danielle bio/character).

- [ ] **Step 1: Author the persona** — role, background, personality, self-facts, answer style (a short character card).
- [ ] **Step 2: Inject it through the channel found in Task 2** — e.g. as a always-present reference/context document, a retrieval-profile system prompt, or the reference generator's prompt style (`prompt_style` seen in `server.py:123`). Exact hook per Task-2 findings.
- [ ] **Step 3: Checkpoint** — across ~3 fresh conversations she states a **consistent** identity (no "voice AI" vs "data scientist" flip).

---

### Task 7: Live end-to-end test — the Phase-2 verdict

- [ ] **Step 1: Tunnel + converse**

```bash
# mac:
ssh -L 8998:localhost:8998 wb-gpu-a1ultra2g   # -> http://localhost:8998
```
- [ ] **Step 2: Judge all four and record**
  1. Voice = Danielle. 2. Turn-taking intact. 3. **Papers-only retrieval probe** answered correctly (ask something only the corpus supplies — a specific method/number from a paper). 4. Persona consistent.
- [ ] **Step 3: Record outcome** in `NEXT_STEPS.md` + memory: what worked, retrieval latency, and whether retrieval sounded smeared (→ Path A trigger) or clean.

---

## Notes for the executor

- The fork source + services aren't visible from the mac — every **INSPECT ON BOX** step reveals the real API/flags/env-var names before acting.
- Task 2 (custom corpus into moshi-rag's retrieval) is the biggest unknown; expect to design the corpus artifact live from the Step-1 findings.
- Bring services up in order 3 → 4 → 5 (dependencies: main server needs the encoder + retrieval + STT URLs).
- If the Gradium key isn't ready, you can still validate voice + turn-taking on the moshika-rag base (boot the main server without STT); retrieval QA waits on the key.
- Keep `.orig` backups; the branch is pushed, so code edits to fork site-packages are box-local (not in git) — note them in the outcome writeup.

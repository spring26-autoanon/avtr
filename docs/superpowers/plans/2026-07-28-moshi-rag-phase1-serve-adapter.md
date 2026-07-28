# Phase 1 — Serve checkpoint_000700 on moshika-rag (fork) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to work this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a **box-side integration runbook** run on the A100 (`wb-gpu-training`), not the mac — the fork source isn't on the mac, so several steps are *inspect-then-act* with the exact edit determined on the box. Verification at each step is a **runtime checkpoint** (does it load / boot / sound right), not a unit test.

**Goal:** Prove `checkpoint_000700` (mainline-trained voice+turn-taking LoRA) overlays onto the fork-built moshika-rag, patch whatever blocks the load, wire `--lora-weight` into the real `server.py`, and confirm live conversation preserves her voice + turn-taking.

**Architecture:** Path B overlay under the fork — load moshika-rag's full weights/config (conditioners live) and fuse the LoRA delta at load. Boot the real `server.py` in minimal config (neutral conditioning, no gemma/STT). Retrieval + persona are Phase 2.

**Tech Stack:** moshi-rag fork (moshi 0.2.13 / torch 2.9.1 / sphn 0.2.x), moshika-rag-pytorch-bf16, `meta-llama/Llama-3.2-3B-Instruct` (ARC encoder dep, HF access approved), `uv`.

## Global Constraints

- Run everything **on the box** `wb-gpu-training`, prefix GPU work with `CUDA_VISIBLE_DEVICES=0`.
- **Do not modify** the mainline serving path or the native venv used for plain `moshi.server` — the fork work goes in a **separate fork env**.
- Long/interactive processes: launch detached (`setsid nohup … &`) and keep `loginctl enable-linger "$USER"` on, per the training-run lesson (systemd SIGTERM'd the last run on logout).
- Adapter under test: `~/moshi-finetune/runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/{lora.safetensors,config.json}` (backup on mac: `adapters/checkpoint_000700/`).
- **Path B only** here (overlay). Path A exact-training is Phase 2 / NEXT_STEPS Part 4 — out of scope.
- When an exact line/API can't be seen from the mac, the step says **INSPECT ON BOX** and gives the probe to reveal it; do not guess line numbers.

---

### Task 1: Fork environment ready (moshi 0.2.13)

**Files:** none (environment only).

- [ ] **Step 1: Look for an existing fork env from the earlier smoke test**

Run:
```bash
ssh wb-gpu-training
ls -d ~/*venv* ~/.venv* ~/moshi-finetune/.venv* 2>/dev/null
# check which moshi each env has
for v in ~/*venv* ~/moshi-finetune/.venv; do
  [ -x "$v/bin/python" ] && echo "== $v ==" && "$v/bin/python" -c "import moshi,torch;print(moshi.__version__,torch.__version__)" 2>/dev/null
done
```
Expected: identify any env reporting `0.2.13 / 2.9.1`. If one exists, note its path as `$FORKVENV` and skip Step 2.

- [ ] **Step 2: If none exists, build the fork env fresh**

The repo `pyproject.toml` already declares the fork stack. Create a dedicated fork venv so the mainline serve/train path is untouched:
```bash
cd ~/moshi-finetune
uv venv ~/fork-venv --python 3.12
VIRTUAL_ENV=~/fork-venv uv pip install --python ~/fork-venv/bin/python \
  "torch==2.9.1" "moshi @ git+https://github.com/kyutai-labs/moshi-rag.git#subdirectory=moshi" "sphn>=0.2.0,<0.3.0"
```
(If a wheel/build error appears, capture it — resolve exactly as encountered; do not proceed on a broken env.)

- [ ] **Step 3: Verify the fork env**

Run (use `$FORKVENV` if found in Step 1, else `~/fork-venv`):
```bash
~/fork-venv/bin/python -c "import moshi, torch, sphn; print('moshi', moshi.__version__, '| torch', torch.__version__, '| sphn', sphn.__version__)"
```
Expected: `moshi 0.2.13 | torch 2.9.1 | sphn 0.2.x`. **Checkpoint met when this prints the fork versions.**

---

### Task 2: Stage moshika-rag assets + confirm the gated dep pulls

**Files:** none (downloads only).

- [ ] **Step 1: Confirm HF login and pull moshika-rag config + weights**

```bash
~/fork-venv/bin/huggingface-cli whoami
~/fork-venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download as d
for f in ["config.json", "model.safetensors"]:
    p = d("kyutai/moshika-rag-pytorch-bf16", f)
    print("got", f, "->", p)
PY
```
Expected: both download, no 401. Note the local cache path of `config.json`.

- [ ] **Step 2: Confirm the gated ARC-encoder dependency resolves**

```bash
~/fork-venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download as d
p = d("meta-llama/Llama-3.2-3B-Instruct", "config.json")
print("Llama-3.2-3B reachable ->", p)
PY
```
Expected: downloads (access is approved). If 401/gated → stop and resolve HF access before continuing (it's a hard dep of moshika-rag's reference encoder). **Checkpoint: both moshika-rag and Llama-3.2 assets resolve.**

---

### Task 3: Loader probe — build moshika-rag + overlay 700, find & patch the meta bug

**Files:**
- Create: `~/moshi-finetune/scratch/loader_probe.py` (throwaway probe)
- Modify (on box, INSPECT first): the fork's `moshi/models/loaders.py` (the `get_lora_moshi` meta-tensor path, ~L560)

**Interfaces:**
- Produces: a fork `loaders.py` that can build moshika-rag with a fused LoRA on CUDA without a meta-tensor error; and the observed adapter/model attention key names for Task 4.

- [ ] **Step 1: INSPECT ON BOX — get the real loader API + meta-bug site**

The exact signatures/lines aren't visible from the mac. Reveal them:
```bash
FORKPKG=$(~/fork-venv/bin/python -c "import moshi,os;print(os.path.dirname(moshi.__file__))")
echo "fork moshi at: $FORKPKG"
grep -n "def get_moshi_lm\|def get_lora_moshi\|class CheckpointInfo\|lora_weights\|fuse_lora\|\.to(\|to_empty\|autocast" "$FORKPKG/models/loaders.py" | head -60
```
Record: the `get_moshi_lm` / `get_lora_moshi` signatures, and the `model.to(...)` / meta / autocast lines. These drive Steps 2–4.

- [ ] **Step 2: Write the loader probe**

Create `~/moshi-finetune/scratch/loader_probe.py` (adjust the two flagged kwargs to the signatures found in Step 1):
```python
import os
from pathlib import Path
from moshi.models import loaders

RAG = "kyutai/moshika-rag-pytorch-bf16"
LORA = Path.home() / "moshi-finetune/runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora.safetensors"

ci = loaders.CheckpointInfo.from_hf_repo(RAG)
try:
    ci.lora_weights = LORA            # per NEXT_STEPS: CheckpointInfo.lora_weights exists
except Exception as e:
    print("NOTE: setting lora_weights attr failed:", e)

# ADJUST kwargs to the real get_moshi_lm signature from Step 1:
lm = loaders.get_moshi_lm(ci, device="cuda", fuse_lora=True, lora_weights=LORA)
print("OK: moshika-rag built + LoRA fused on cuda")

# dump attention + lora param names for the Task-4 compat check
shown = 0
for name, p in lm.named_parameters():
    if ("in_proj" in name) or ("lora" in name):
        print(f"  [{'meta' if p.is_meta else 'ok'}] {name} {tuple(p.shape)}")
        shown += 1
        if shown >= 20:
            break
```

- [ ] **Step 3: Run it, expect the meta-tensor crash, pinpoint the site**

```bash
cd ~/moshi-finetune
CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python scratch/loader_probe.py 2>&1 | tail -40
```
Expected (per NEXT_STEPS Part 4 Stage 0): `RuntimeError: Cannot copy out of meta tensor` (from `model.to(device)` on a meta model) and/or an ARC-encoder `unsupported autocast device_type 'meta'`. The traceback names the exact file+line in the fork's `loaders.py`.
- **If it loads with no error** → skip to Step 5 (the meta bug may already be absent in this fork build).

- [ ] **Step 4: Patch the meta-init in the fork's loaders.py**

At the traceback site, replace the meta→device move with an empty-then-load, and materialize conditioners on a real device (exact edit depends on Step 1/3 findings). Pattern:
```python
# BEFORE (crashes on a meta-device model):
#   model.to(device)
# AFTER:
model.to_empty(device=device)              # allocate real storage, no copy-from-meta
missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
# ensure any conditioner/ARC submodules built under autocast('meta') are constructed
# on a real device instead (drop autocast(device_type='meta') at their ctor site).
```
Keep a copy of the original: `cp "$FORKPKG/models/loaders.py" "$FORKPKG/models/loaders.py.orig"` before editing.

- [ ] **Step 5: Re-run the probe to confirm a clean build**

```bash
CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python scratch/loader_probe.py 2>&1 | tail -30
```
Expected: `OK: moshika-rag built + LoRA fused on cuda`, followed by attention/lora param lines. **Checkpoint: model builds and the adapter fuses with no meta-tensor error.** Save the printed param-name list for Task 4.

---

### Task 4: Key-compat verdict (does the mainline-trained adapter fill the fork model's LoRA slots?)

**Files:** Create `~/moshi-finetune/scratch/key_compat.py`

**Interfaces:**
- Consumes: the built model from Task 3.
- Produces: a verdict — **MATCH** (Path B overlay valid) or **MISMATCH** (remap or fork-retrain).

- [ ] **Step 1: Dump adapter keys vs the model's LoRA-bound modules**

Create `~/moshi-finetune/scratch/key_compat.py`:
```python
from pathlib import Path
from safetensors import safe_open

LORA = Path.home() / "moshi-finetune/runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora.safetensors"
with safe_open(str(LORA), framework="pt") as f:
    adapter_keys = list(f.keys())
print(f"adapter: {len(adapter_keys)} tensors")
att = [k for k in adapter_keys if "in_proj" in k or "self_attn" in k]
print(f"  attention-related adapter keys: {len(att)}; sample:")
for k in att[:10]:
    print("   ", k)
```
```bash
~/fork-venv/bin/python scratch/key_compat.py
```

- [ ] **Step 2: Cross-check against the model + read the probe's `meta` flags**

From Task 3 Step 5 output: **any attention/LoRA param still printed as `[meta]` means an unfilled slot → the adapter did NOT fill it → key mismatch.** All `[ok]` (and the fuse succeeding) means the names line up.

- [ ] **Step 3: Record the verdict and branch**

- **MATCH** (fuse succeeded, no `[meta]` LoRA/attention params) → Path B overlay is valid; proceed to Task 5.
- **MISMATCH** (meta slots remain / adapter keys don't correspond to model modules) → **do not force it.** Choose:
  - **(a) Remap** if the difference is purely a naming convention (e.g. mainline `in_projs.N` vs a fork alias) — write a small key-rename over `lora.safetensors` and re-run Task 3 Step 5.
  - **(b) Fork retrain** if the attention layout genuinely differs — retrain the adapter under the fork (the fork *training* path is smoke-tested green; reuse `example/moshika_rag_voice.yaml` under `~/fork-venv`, same dialogue data), then re-enter at Task 3. Record which path taken and why.

**Checkpoint: a written MATCH/MISMATCH verdict, and if MISMATCH, a working overlay produced via (a) or (b).**

---

### Task 5: Integrate `--lora-weight` into the real `server.py` and boot minimal

**Files:** Modify (on box, INSPECT first): the fork's `moshi/server.py` (the main conversational server — the LM loader / `parse_args`).

- [ ] **Step 1: INSPECT ON BOX — find server.py's arg parser + LM load + service deps**

```bash
grep -n "add_argument\|get_moshi_lm\|CheckpointInfo\|lora\|STT\|conditioner\|reference\|os.environ" "$FORKPKG/server.py" | head -60
```
Record: where args are parsed, where the LM is built, and **what external services/env vars it reads at startup** (STT_URL/API key, reference-encoder URL, retrieval LLM URL) — this determines the Step-3 branch.

- [ ] **Step 2: Add the `--lora-weight` flag and thread it through**

In the fork's `server.py`, add to `parse_args()`:
```python
parser.add_argument("--lora-weight", type=str, default=None,
                    help="LoRA safetensors to fuse over the moshika-rag base weights")
```
and where `CheckpointInfo` is built / `get_moshi_lm` is called, pass it through (match the real signature from Task 3 Step 1):
```python
checkpoint_info.lora_weights = Path(args.lora_weight) if args.lora_weight else None
# and/or: get_moshi_lm(..., lora_weights=checkpoint_info.lora_weights, fuse_lora=True)
```
Back up first: `cp "$FORKPKG/server.py" "$FORKPKG/server.py.orig"`.

- [ ] **Step 3: Boot the server minimal (neutral conditioning, no gemma/STT) — BRANCH on its startup deps**

Attempt the lightest boot with the adapter:
```bash
cd ~/moshi-finetune
CUDA_VISIBLE_DEVICES=0 setsid nohup ~/fork-venv/bin/python -m moshi.server \
  --config <moshika-rag config.json path from Task 2> \
  --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors \
  --lora-weight runs/moshika_dialogue/checkpoints/checkpoint_000700/consolidated/lora.safetensors \
  > ~/serve_ragbase.log 2>&1 &
echo "PID $!"; sleep 25; tail -30 ~/serve_ragbase.log
```
(Use the exact flag names discovered in Step 1 — the moshi-rag server's flags may differ; `server_conditioner.py` exposes `--config/--moshi-weight/--conditioner`, the main `server.py` may differ.)

Branch on the result:
- **Boots + listens** (log shows it accepting connections) → this is the Phase-1 rig; go to Task 6.
- **Hard-fails needing a service** (STT key / reference-encoder URL / retrieval LLM) → either set those env vars to harmless stubs (empty reference, no-op STT) enough to boot, OR if that's not quick, **fall back**: converse via a load-and-converse harness built on the Task-3 probe + Mimi decode, and **defer real-server boot to Phase 2**. Record which path taken.

**Checkpoint: something serves and accepts a WebSocket connection with the adapter fused.**

---

### Task 6: Converse & judge — the Phase-1 verdict

**Files:** none.

- [ ] **Step 1: Tunnel from the mac and open the UI**

```bash
# on the mac:
ssh -L 8998:localhost:8998 wb-gpu-training
# browser -> http://localhost:8998
```

- [ ] **Step 2: Judge both criteria and record**

Talk to her and answer:
1. **Voice** — recognizably Danielle on the moshika-rag base (as good as on plain `moshi.server`)?
2. **Turn-taking** — does she still yield / let you speak (survived the overlay onto moshika-rag)?
3. **Coherence under neutral conditioning** — is basic conversation sane with no retrieval fed? (input for Phase 2, not a pass/fail here).

- [ ] **Step 3: Write the Phase-1 outcome**

Record in `NEXT_STEPS.md` (and memory): env used, meta-patch applied, MATCH/MISMATCH verdict, whether the real server booted standalone or needed stubs, and the voice/turn-taking judgment. **Phase 1 is done when: the fork env loads moshika-rag + 700 with no meta/key errors, the `--lora-weight` plumbing is in `server.py`, and live conversation preserves voice + turn-taking.**

---

## Notes for the executor

- The fork source is **not on the mac**; every `INSPECT ON BOX` step reveals the real API/lines before editing — never guess line numbers.
- Keep `.orig` backups of any fork file you edit (`loaders.py`, `server.py`) so the fork env stays reproducible.
- If Task 4 forces a fork retrain (branch b), that's a bounded detour (data + config already exist) — re-enter at Task 3, don't restart the plan.
- Retrieval, persona, gemma-27b, STT, and the 3-service stack are **Phase 2** — resist pulling them in to make Task 5 "boot fully"; neutral conditioning is sufficient for the Phase-1 verdict.

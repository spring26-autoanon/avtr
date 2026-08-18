# Track A voice-clone checkpoints — tester instructions

You're auditioning three LoRA checkpoints of a Moshi voice clone ("Danielle") for
conversational coherence. Each checkpoint is a folder with `lora.safetensors` +
`config.json`. No retrieval stack involved — this is plain Moshi + adapter.

## What you need
- A CUDA GPU with ~20+ GB VRAM (the model is ~7B bf16; an A100/L4/3090+ works)
- Python 3.10+ and ~30 GB disk (model weights download from HuggingFace on first run)

## Setup (once, ~10 min)
```bash
python3 -m venv ~/moshi-venv && source ~/moshi-venv/bin/activate
pip install "moshi==0.2.13"
```
No HuggingFace login needed (the base model `kyutai/moshika-pytorch-bf16` is public).

## Run a checkpoint
```bash
source ~/moshi-venv/bin/activate
python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  --lora-weight /path/to/checkpoint_001000/consolidated/lora.safetensors \
  --config-path /path/to/checkpoint_001000/consolidated/config.json \
  --port 8999
```
Wait ~2 min for "listening", then open http://localhost:8999 in a **fresh private
browser window** (stale service workers cause a white screen / fake "at capacity"
page — new private window fixes it). Allow the mic. If remote, tunnel first:
`ssh -L 8999:localhost:8999 <host> -N`.

Swap checkpoints by killing the server (Ctrl-C or `pkill -f moshi.server`) and
relaunching with the other checkpoint's two paths.

## The test (per checkpoint, ~5 min): 1000 first, then 800, then 600
Speak naturally, short questions, letting her finish:
1. "Hi, how are you?" — does she answer THIS question?
2. "What did you do this weekend?" — coherent, on-topic reply?
3. "Do you have any pets?" — DON'T mention pets before this; a pet answer only
   counts if you asked.
4. "What's your favorite food?"
5. Chat freely for 2–3 minutes.

Score each checkpoint (1–5 each):
- **Coherence**: answers track your actual questions; no drifting into unprompted
  monologue topics.
- **Voice**: sounds like the reference speaker (young female, natural fillers —
  "um", "like", laughs are GOOD signs, not glitches).
- **Stability**: no long silences (>10 s), survives past 4–5 minutes.

## What to send back
For each checkpoint: the three scores + 2–3 quotable exchanges (what you asked,
what she said). If the server prints a wall of "LM buffer empty" lines while she
goes silent, note it — that's a stall, the key failure we're hunting.

Known behavior: she may ramble when you leave long silences (keep the
conversational ball moving); mild audio crackle at session start is normal.

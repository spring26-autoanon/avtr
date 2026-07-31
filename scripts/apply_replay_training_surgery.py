#!/usr/bin/env python3
"""Path A Stage 2/3 — enable reference-conditioning during TRAINING (the surgery).

Two changes, both required to train moshika-rag on retrieval (replay) data:

1. Write `config.replaytrain.json` — the moshika-rag config with `reference_with_time`
   removed from `conditioners` (so the ARC encoder isn't built in the trainer → no
   `autocast('meta')` crash) but KEPT in `fuser.streaming_sum` (so the fuser still
   applies a *precomputed* reference tensor supplied via `condition_tensors`).

2. Patch the fork's `moshi/models/lm.py` `forward_text` streaming_sum block. The stock
   code applies streaming_sum only in generation (`S==1`, one reference row per step).
   We add a TRAINING branch: for full sequences (`S>1`), add the reference offset over
   the first `condition_len` timesteps. Verified 2026-07-31: with this patch the
   training forward consumes an injected `reference_with_time` tensor (logits respond,
   nan-mean diff ~1.35 vs zeros).

Run under the FORK env (moshi 0.2.13). Idempotent. Keeps `lm.py.orig` backup.

    ~/fork-venv/bin/python scripts/apply_replay_training_surgery.py

Serve/generation is unaffected — the original `S==1` path is preserved.

STILL TODO for a full replay-training run (engineering, not research):
- data plumbing: carry a precomputed reference tensor + ⟨ret⟩ frame position per
  example through the interleaver, inject into `condition_tensors` in train.py's loop;
- position the reference at the ⟨ret⟩ token (not the prefix);
- re-add `first_speaker` prepend for real training;
- generate replay data (QA -> dialogue -> her-voice TTS via Gradium -> precompute the
  reference tensor via the :8001 encoder -> package) and retrain on the voice+replay mix.
"""
import glob
import json
import os

import moshi

FORKPKG = os.path.dirname(moshi.__file__)


def write_replay_config() -> str:
    cfgs = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--kyutai--moshika-rag-pytorch-bf16/snapshots/*/config.json"))
    if not cfgs:
        raise SystemExit("moshika-rag config.json not cached; pull it first (HF login).")
    c = json.load(open(cfgs[0]))
    c.get("conditioners", {}).pop("reference_with_time", None)  # drop ARC module, KEEP in fuser
    out = os.path.expanduser("~/moshi-finetune/checkpoints/moshika-rag/config.replaytrain.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(c, open(out, "w"))
    print(f"wrote {out}: conditioners={list(c.get('conditioners',{}).keys())} "
          f"fuser.streaming_sum={c['fuser']['streaming_sum']}")
    return out


def patch_forward_text() -> None:
    lm = os.path.join(FORKPKG, "models", "lm.py")
    s = open(lm).read()
    if "TRAINING (full sequence): apply the reference offset" in s:
        print("lm.py already patched.")
        return
    old = (
        "            if condition_len > 0:\n"
        "                assert condition_len == 1\n"
        "                assert S == 1\n"
        "                assert streaming_sum_condition.shape[0] == input_.shape[0]\n"
        "                input_ = input_ + streaming_sum_condition"
    )
    new = (
        "            if condition_len > 0:\n"
        "                assert streaming_sum_condition.shape[0] == input_.shape[0]\n"
        "                if S == 1:\n"
        "                    assert condition_len == 1\n"
        "                    input_ = input_ + streaming_sum_condition\n"
        "                else:\n"
        "                    # TRAINING (full sequence): apply the reference offset over the\n"
        "                    # first `condition_len` timesteps of the sequence.\n"
        "                    L = min(condition_len, S)\n"
        "                    pad = torch.zeros_like(input_)\n"
        "                    pad[:, :L] = streaming_sum_condition[:, :L].to(input_)\n"
        "                    input_ = input_ + pad"
    )
    if old not in s:
        raise SystemExit("streaming_sum block not found (fork version drift?) — patch by hand.")
    if not os.path.exists(lm + ".orig"):
        open(lm + ".orig", "w").write(s)
    open(lm, "w").write(s.replace(old, new))
    print(f"patched {lm} (backup at {lm}.orig)")


if __name__ == "__main__":
    write_replay_config()
    patch_forward_text()

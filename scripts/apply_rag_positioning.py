#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 5: rag-token emission + reference positioning.

Two code changes so replay training teaches BOTH halves of retrieval:

1. finetune/data/interleaver.py `_tokenize`: a `"<RAG>"` marker word maps to the raw
   rag_token_id (4) instead of being sentencepiece-encoded. The post-processor inserts a
   `"<RAG>"` alignment at each retrieval example's answer start, so token 4 lands in the text
   stream and the model learns to EMIT ⟨ret⟩.

2. train.py reference injection: position the precomputed references at EVERY rag_token frame
   in the text row (codes[:,0,:] == 4), pairing the i-th marker with the i-th reference and
   clamping each span at the next marker — matching serve (a reference conditions the answer
   following its own trigger). The tensor maths lives in
   finetune/data/reference_injection.py so it is unit-testable without moshi/CUDA; the lm.py
   streaming_sum surgery then adds it at the right frames.

   Stage 3 note: an earlier version used only the FIRST marker (`hit[0]`). Every retrieval
   turn after the first in a clip was then trained with no conditioning while the loss still
   demanded a grounded answer — i.e. it taught confabulation. Do not revert that.

Idempotent; keeps .orig backups. Run wherever the repo lives (Mac to commit, box to apply).

    python scripts/apply_rag_positioning.py
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERLEAVER = os.path.join(REPO, "finetune", "data", "interleaver.py")
TRAIN = os.path.join(REPO, "train.py")
RAG_TOKEN_ID = 4


def _backup(path, text):
    if not os.path.exists(path + ".orig2"):
        open(path + ".orig2", "w").write(text)


def patch_interleaver():
    s = open(INTERLEAVER).read()
    if '"<RAG>"' in s:
        print("interleaver.py already has RAG marker.")
        return
    old = (
        "        for word, ts, speaker in alignments:\n"
        "            toks = tokenize(self.tokenizer, word.strip(), bos=False)\n"
        "            out.append((toks, ts, speaker))\n"
    )
    new = (
        "        for word, ts, speaker in alignments:\n"
        "            if word.strip() == \"<RAG>\":\n"
        f"                toks = [{RAG_TOKEN_ID}]  # rag_token_id -> emit the ⟨ret⟩ retrieval trigger\n"
        "            else:\n"
        "                toks = tokenize(self.tokenizer, word.strip(), bos=False)\n"
        "            out.append((toks, ts, speaker))\n"
    )
    if old not in s:
        raise SystemExit("interleaver _tokenize loop not found (drift?) — patch by hand.")
    _backup(INTERLEAVER, s)
    open(INTERLEAVER, "w").write(s.replace(old, new))
    print(f"patched {INTERLEAVER}")


def patch_train():
    t = open(TRAIN).read()
    if "build_reference_condition" in t:
        print("train.py already positions references at every rag frame.")
        return

    comment_start = "            # Inject precomputed reference_with_time"
    if_start = '            if getattr(batch, "reference_tensors", None) is not None:'
    end_anchor = '                condition_tensors["reference_with_time"] = ConditionType(ref_cond, ref_mask)'

    i = t.find(comment_start)
    if i == -1:
        i = t.find(if_start)
    j = t.find(end_anchor)
    if i == -1 or j == -1:
        raise SystemExit("train.py reference-injection block not found (drift?) — patch by hand.")
    j_end = j + len(end_anchor)

    new_block = (
        '            if getattr(batch, "reference_tensors", None) is not None:\n'
        "                # Inject precomputed reference_with_time (replay) at EVERY rag_token (⟨ret⟩)\n"
        "                # frame, so multi-retrieval clips condition each answer (matches serve).\n"
        "                from moshi.conditioners.base import ConditionType\n"
        "\n"
        "                from finetune.data.reference_injection import build_reference_condition\n"
        "\n"
        "                refs = batch.reference_tensors\n"
        "                present = [r for r in refs if r is not None and len(r)]\n"
        "                if present:\n"
        "                    first = present[0]\n"
        "                    dim = (first[0] if isinstance(first, list) else first).shape[-1]\n"
        "                    ref_cond, ref_mask = build_reference_condition(\n"
        "                        codes[:, 0, :], refs, dim, next(model.parameters()).dtype\n"
        "                    )\n"
        "                    if condition_tensors is None:\n"
        "                        condition_tensors = {}\n"
        '                    condition_tensors["reference_with_time"] = ConditionType(ref_cond, ref_mask)'
    )
    _backup(TRAIN, t)
    open(TRAIN, "w").write(t[:i] + new_block + t[j_end:])
    print(f"patched {TRAIN}")


if __name__ == "__main__":
    patch_interleaver()
    patch_train()
    print("rag-token emission + reference positioning applied.")

#!/usr/bin/env python3
"""Path A Stage 2/3 — Task 5: rag-token emission + reference positioning.

Two code changes so replay training teaches BOTH halves of retrieval:

1. finetune/data/interleaver.py `_tokenize`: a `"<RAG>"` marker word maps to the raw
   rag_token_id (4) instead of being sentencepiece-encoded. The post-processor inserts a
   `"<RAG>"` alignment at each retrieval example's answer start, so token 4 lands in the text
   stream and the model learns to EMIT ⟨ret⟩.

2. train.py reference injection: position the precomputed reference at the rag_token frame in
   the text row (codes[:,0,:] == 4) and apply it over the following T_ref frames — matching
   serve (reference conditions the answer, from the trigger), instead of the earlier
   prefix-only placement. Full-length ref_cond, zero except at [rag:rag+T_ref]; the lm.py
   streaming_sum surgery then adds it at the right frames.

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
    if "RAG_TOKEN_ID" in t:
        print("train.py already positions reference at rag frame.")
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
        "                # Inject precomputed reference_with_time (replay), positioned at the\n"
        "                # rag_token (⟨ret⟩) frame so it conditions her answer (matches serve),\n"
        "                # not the prefix. Full-length ref_cond, zero except at [rag:rag+T_ref].\n"
        "                from moshi.conditioners.base import ConditionType\n"
        "\n"
        f"                RAG_TOKEN_ID = {RAG_TOKEN_ID}\n"
        "                refs = batch.reference_tensors\n"
        "                present = [r for r in refs if r is not None]\n"
        "                dim = present[0].shape[-1]\n"
        "                dt = next(model.parameters()).dtype\n"
        "                Bsz, _, S = codes.shape\n"
        "                text_row = codes[:, 0, :]\n"
        "                ref_cond = torch.zeros(Bsz, S, dim, device=codes.device, dtype=dt)\n"
        "                ref_mask = torch.zeros(Bsz, S, device=codes.device)\n"
        "                for bi, r in enumerate(refs):\n"
        "                    if r is None:\n"
        "                        continue\n"
        "                    hit = (text_row[bi] == RAG_TOKEN_ID).nonzero(as_tuple=False)\n"
        "                    start = int(hit[0]) if len(hit) else 0\n"
        "                    L = min(r.shape[0], S - start)\n"
        "                    ref_cond[bi, start:start + L] = r[:L].to(device=codes.device, dtype=dt)\n"
        "                    ref_mask[bi, start:start + L] = 1\n"
        "                if condition_tensors is None:\n"
        "                    condition_tensors = {}\n"
        '                condition_tensors["reference_with_time"] = ConditionType(ref_cond, ref_mask)'
    )
    _backup(TRAIN, t)
    open(TRAIN, "w").write(t[:i] + new_block + t[j_end:])
    print(f"patched {TRAIN}")


if __name__ == "__main__":
    patch_interleaver()
    patch_train()
    print("rag-token emission + reference positioning applied.")

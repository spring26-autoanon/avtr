"""Build the reference_with_time condition tensor for a training batch.

At serve time the model emits ⟨ret⟩, the retrieval LLM writes a passage, the :8001 ARC
encoder embeds it, and the result conditions the answer that follows. Training mirrors that:
each precomputed reference is placed AT its own ⟨ret⟩ frame and applied over the following
T_ref frames.

A clip can hold several retrieval turns, so the i-th ⟨ret⟩ frame is paired with the i-th
reference and each span is clamped at the next ⟨ret⟩ frame — otherwise a long passage would
bleed over the following turn, and (in the earlier first-hit-only version) every retrieval
turn after the first would be trained with no conditioning at all, which teaches the model to
answer factual questions from its own head.

Deliberately free of any `moshi` import so it can be unit-tested off the A100.
"""
import torch

RAG_TOKEN_ID = 4


def build_reference_condition(text_row, refs, dim, dtype, rag_token_id=RAG_TOKEN_ID):
    """(ref_cond [B, S, dim], ref_mask [B, S]) from per-example lists of [T_ref, dim] tensors.

    `refs[b]` is a list of tensors, or None. `text_row` is codes[:, 0, :].
    An example whose references outnumber its markers (or vice versa) uses only the
    positions that pair up. An example with references but no marker falls back to the
    sequence prefix, matching the pre-Stage-3 behaviour for legacy clips.
    """
    device = text_row.device
    bsz, seq = text_row.shape
    ref_cond = torch.zeros(bsz, seq, dim, device=device, dtype=dtype)
    ref_mask = torch.zeros(bsz, seq, device=device)

    for bi in range(bsz):
        tensors = refs[bi] if refs is not None else None
        if tensors is None:
            continue
        if torch.is_tensor(tensors):        # tolerate a bare [T_ref, dim] tensor
            tensors = [tensors]
        if not tensors:
            continue

        hits = (text_row[bi] == rag_token_id).nonzero(as_tuple=False).flatten().tolist()
        if not hits:
            hits = [0]

        n = min(len(hits), len(tensors))
        for i in range(n):
            start = hits[i]
            limit = hits[i + 1] if i + 1 < len(hits) else seq
            span = min(tensors[i].shape[0], limit - start, seq - start)
            if span <= 0:
                continue
            ref_cond[bi, start:start + span] = tensors[i][:span].to(device=device, dtype=dtype)
            ref_mask[bi, start:start + span] = 1
    return ref_cond, ref_mask

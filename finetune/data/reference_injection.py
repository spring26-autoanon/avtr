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
import random as _random

import torch

RAG_TOKEN_ID = 4
FRAME_RATE = 12.5
# Her measured median lead (reference-word method) — the fallback until §4.2 lead labelling
# writes a real d_lead per turn into the manifest.
DEFAULT_LEAD_SEC = 1.2


def sample_delay_sec(d_lead, rng=_random):
    """moshi-rag paper Eq. 3 — how long after ⟨ret⟩ the reference becomes available.

        d' = U(0, d_lead)          if d_lead < 2.0s or p < 0.2
             U(1.0, d_lead - 1.0)  otherwise,          p ~ U(0,1)

    The second branch keeps a second of runway at each end of the lead; the p < 0.2 escape
    hatch keeps some near-zero delays in the distribution so the model still handles a fast
    retrieval. p is always drawn, so a seeded rng gives the same stream on either branch.
    """
    if d_lead <= 0.0:
        return 0.0
    p = rng.random()
    if d_lead < 2.0 or p < 0.2:
        return rng.uniform(0.0, d_lead)
    return rng.uniform(1.0, d_lead - 1.0)


def build_reference_condition(
    text_row,
    refs,
    dim,
    dtype,
    rag_token_id=RAG_TOKEN_ID,
    leads=None,
    default_lead=DEFAULT_LEAD_SEC,
    sample_delay=False,
    dropout=0.0,
    rng=_random,
    frame_rate=FRAME_RATE,
):
    """(ref_cond [B, S, dim], ref_mask [B, S]) from per-example lists of [T_ref, dim] tensors.

    `refs[b]` is a list of tensors, or None. `text_row` is codes[:, 0, :].
    An example whose references outnumber its markers (or vice versa) uses only the
    positions that pair up. An example with references but no marker falls back to the
    sequence prefix, matching the pre-Stage-3 behaviour for legacy clips.

    `sample_delay` places each reference at `hit + d'` (Eq. 3) instead of at the hit, so
    training matches serve time, where the document takes 1.7–3.4 s to arrive. `leads[b][i]`
    is the i-th turn's d_lead in seconds; `default_lead` covers turns the manifest hasn't
    labelled yet. `dropout` skips injection for a turn entirely (paper: 0.2) — the ⟨ret⟩
    marker itself is untouched, so the model still learns to fire it.

    Both are OFF by default: Stage 3 behaviour is the default path.
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

        ex_leads = leads[bi] if leads is not None and bi < len(leads) else None

        n = min(len(hits), len(tensors))
        for i in range(n):
            if dropout > 0.0 and rng.random() < dropout:
                continue                    # reference dropped; the ⟨ret⟩ marker stays

            start = hits[i]
            if sample_delay:
                lead = default_lead
                if ex_leads is not None and i < len(ex_leads) and ex_leads[i] is not None:
                    lead = ex_leads[i]
                start += int(round(sample_delay_sec(lead, rng) * frame_rate))

            limit = hits[i + 1] if i + 1 < len(hits) else seq
            span = min(tensors[i].shape[0], limit - start, seq - start)
            if span <= 0:
                continue                    # arrived after the answer ended — a real case
            ref_cond[bi, start:start + span] = tensors[i][:span].to(device=device, dtype=dtype)
            ref_mask[bi, start:start + span] = 1
    return ref_cond, ref_mask

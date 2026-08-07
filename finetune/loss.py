import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
    rag_token_id: int | None = None,
    rag_token_weight: float = 1.0,
    rag_loss_mask: torch.Tensor | None = None,
):
    """
    rag_loss_mask: optional bool tensor [B]. Where True, the rag_token logit is removed from
        the text softmax so it receives NO gradient — used for examples with no reference,
        i.e. windows where we never verified whether retrieval was warranted.

        Without this, plain cross-entropy pushes P(⟨ret⟩) down at every position of every
        marker-free window. With ~135 markers against thousands of such windows, that
        negative signal dominates and un-teaches the base model's trigger.

        Costs nothing elsewhere: it touches one logit index in the text head. The audio
        codebooks (voice) and every other text token (her phrasing and filler) are untouched.
    """
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight
        # Upweight the rag_token (⟨ret⟩) so the model learns to EMIT it reliably instead of
        # confabulating; it's ~1 token/example so it needs a strong multiplier to matter.
        if rag_token_id is not None and rag_token_weight != 1.0:
            weights[target == rag_token_id] *= rag_token_weight

    # Blank the rag logit on unlabelled examples, BEFORE the softmax, so no gradient pushes
    # it down. Skip any position whose target IS the rag token: a marked clip whose reference
    # failed to precompute would otherwise take cross_entropy over a -inf target -> NaN.
    blank = None
    if mode == "text" and rag_token_id is not None and rag_loss_mask is not None:
        per_example = rag_loss_mask.view(-1, *([1] * (target.dim() - 1))).expand_as(target)
        blank = (per_example & (target != rag_token_id)).view(-1)

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    if blank is not None:
        logits = logits.masked_fill(
            blank.unsqueeze(-1)
            & (
                torch.arange(logits.size(-1), device=logits.device) == rag_token_id
            ).unsqueeze(0),
            torch.finfo(logits.dtype).min,
        )
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights)

    return mb_loss

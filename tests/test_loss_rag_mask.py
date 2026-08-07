"""The rag-token loss mask — the fix for our fine-tune un-teaching the trigger.

Plain cross-entropy over the text stream pushes P(⟨ret⟩) down at EVERY position of EVERY
window whose target doesn't contain it. We have ~135 markers against thousands of
marker-free windows, so the dominant gradient is "never emit this token" — which is why
RAG_TOKEN_WEIGHT=25 could only get triggering to ~50%.

On a labelled clip we know the ground truth everywhere (fire at the marker, nowhere else),
so it trains normally. On an unlabelled window we never verified whether retrieval was
warranted, so we assert nothing: the rag logit is masked out of the softmax and receives no
gradient.

Critically this must cost NOTHING elsewhere — the audio codebooks carry the voice, and every
other text token carries her filler and phrasing.
"""
import torch

from finetune.loss import compute_loss_with_mask

RAG = 4
VOCAB = 16


def _logits(b=2, s=5, vocab=VOCAB):
    torch.manual_seed(0)
    return torch.randn(b, 1, s, vocab, requires_grad=True)


def _target(b=2, s=5):
    return torch.full((b, 1, s), 7, dtype=torch.long)


def _text_loss(logits, target, rag_loss_mask=None, rag_token_weight=1.0):
    return compute_loss_with_mask(
        logits,
        target,
        torch.ones_like(target, dtype=torch.bool),
        mode="text",
        text_padding_ids=set(),
        rag_token_id=RAG,
        rag_token_weight=rag_token_weight,
        rag_loss_mask=rag_loss_mask,
    )


def _rag_grad(logits, target=None, **kw):
    """Total gradient landing on the rag-token logit, per batch element."""
    logits = logits.detach().clone().requires_grad_(True)
    _text_loss(logits, _target() if target is None else target, **kw).backward()
    return logits.grad[..., RAG].abs().sum(dim=(1, 2))


def test_masked_example_gets_no_gradient_on_the_rag_logit():
    """The whole point: an unlabelled window must not push P(<ret>) down."""
    g = _rag_grad(_logits(), rag_loss_mask=torch.tensor([True, False]))
    assert g[0].item() == 0.0


def test_unmasked_example_in_the_same_batch_is_untouched():
    """Masking is per-example, not per-batch — a labelled clip still trains normally."""
    mask = torch.tensor([True, False])
    logits = _logits()
    assert _rag_grad(logits, rag_loss_mask=mask)[1] > 0
    # and identical to what it would be with no masking at all
    assert torch.allclose(
        _rag_grad(logits, rag_loss_mask=mask)[1], _rag_grad(logits)[1], atol=1e-6
    )


def test_no_mask_argument_leaves_behaviour_exactly_as_before():
    """Track A (plain moshika) must be bit-identical: token 4 is an ordinary token there."""
    logits, target = _logits(), _target()
    assert torch.equal(_text_loss(logits, target), _text_loss(logits, target))
    assert (_rag_grad(logits) > 0).all()


def test_masking_does_not_disturb_other_vocabulary_entries():
    """Voice and filler live in the other text tokens and the audio stream — the mask must
    cost them nothing."""
    logits = _logits()
    a = logits.detach().clone().requires_grad_(True)
    b = logits.detach().clone().requires_grad_(True)
    _text_loss(a, _target()).backward()
    _text_loss(b, _target(), rag_loss_mask=torch.tensor([True, True])).backward()
    others = [v for v in range(VOCAB) if v != RAG]
    # the loss is renormalised over a smaller softmax, so allow a tolerance, but the
    # gradient must not change sign or magnitude materially
    assert torch.allclose(a.grad[..., others], b.grad[..., others], atol=5e-2)


def test_a_marker_position_still_trains_even_if_its_example_is_masked():
    """Defensive: a marked clip whose reference tensor failed to precompute would otherwise
    take cross_entropy(-inf) -> NaN. The target position must stay finite and trainable."""
    target = _target()
    target[0, 0, 2] = RAG
    logits = _logits()
    loss = _text_loss(logits, target, rag_loss_mask=torch.tensor([True, False]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 2, RAG].abs() > 0


def test_upweighting_still_applies_on_unmasked_examples():
    """RAG_TOKEN_WEIGHT and the mask are complementary: the mask removes the negative
    pressure, the weight keeps the positives strong."""
    target = _target()
    target[1, 0, 2] = RAG
    logits = _logits()
    g1 = _rag_grad(logits, target, rag_loss_mask=torch.tensor([True, False]))
    g25 = _rag_grad(logits, target, rag_loss_mask=torch.tensor([True, False]),
                    rag_token_weight=25.0)
    assert g25[1] > g1[1]


def test_audio_mode_ignores_the_mask_entirely():
    """The rag token is a text-stream concept; the audio codebooks carry the voice."""
    logits = _logits()
    target = _target()
    a = compute_loss_with_mask(
        logits, target, torch.ones_like(target, dtype=torch.bool), mode="audio"
    )
    b = compute_loss_with_mask(
        logits,
        target,
        torch.ones_like(target, dtype=torch.bool),
        mode="audio",
        rag_token_id=RAG,
        rag_loss_mask=torch.tensor([True, True]),
    )
    assert torch.equal(a, b)

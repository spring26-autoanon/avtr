"""Delay sampling (moshi-rag paper Eq. 3) and reference dropout.

Why this exists: we inject the reference AT the ⟨ret⟩ frame. The paper injects at
`i_⟨ret⟩ + d'`, where d' is sampled per turn and bounded by the lead. Training with a
zero delay teaches the model the document is available the instant it asks — at serve time
it takes 1.7–3.4 s. Observed consequence: she committed to "Argentina" 1.8 s BEFORE the
correct document arrived, then defended the wrong answer.

    d' = U(0, d_lead)          if d_lead < 2.0s  or  p < 0.2
         U(1.0, d_lead - 1.0)  otherwise,           p ~ U(0,1)

Both delay and dropout default OFF so Stage 3 behaviour and the existing 12 injection tests
are unchanged; train.py opts in per run.
"""
import random

import torch

from finetune.data.reference_injection import build_reference_condition, sample_delay_sec

RAG = 4
DIM = 3
FRAME_RATE = 12.5


def _row(marker_positions, seq=40):
    r = torch.full((1, seq), 7, dtype=torch.long)
    for p in marker_positions:
        r[0, p] = RAG
    return r


def _ref(n=10):
    return [[torch.ones(n, DIM)]]


def _span(mask):
    """(first, last) frame indices where the reference is applied, or None."""
    on = mask[0].nonzero().flatten().tolist()
    return (on[0], on[-1]) if on else None


# --- Eq. 3 ------------------------------------------------------------------


def test_short_lead_samples_the_whole_window():
    """d_lead < 2s -> U(0, d_lead). Her median lead is 1.2s, so this is the common branch."""
    rng = random.Random(0)
    got = [sample_delay_sec(1.2, rng) for _ in range(500)]
    assert all(0.0 <= d <= 1.2 for d in got)
    assert max(got) > 1.0 and min(got) < 0.2


def test_long_lead_usually_samples_the_inset_window():
    """d_lead >= 2s -> U(1.0, d_lead - 1.0), keeping a second of runway at each end."""
    rng = random.Random(0)
    got = [sample_delay_sec(4.0, rng) for _ in range(2000)]
    inset = [d for d in got if 1.0 <= d <= 3.0]
    assert len(inset) / len(got) > 0.75


def test_long_lead_falls_back_to_the_full_window_about_a_fifth_of_the_time():
    """The p < 0.2 escape hatch keeps some near-zero delays in the distribution."""
    rng = random.Random(0)
    got = [sample_delay_sec(4.0, rng) for _ in range(4000)]
    outside = [d for d in got if d < 1.0 or d > 3.0]
    assert 0.10 < len(outside) / len(got) < 0.30


def test_zero_or_negative_lead_yields_no_delay():
    rng = random.Random(0)
    assert sample_delay_sec(0.0, rng) == 0.0
    assert sample_delay_sec(-1.0, rng) == 0.0


def test_sampling_is_reproducible_under_a_seed():
    a = [sample_delay_sec(3.0, random.Random(7)) for _ in range(5)]
    b = [sample_delay_sec(3.0, random.Random(7)) for _ in range(5)]
    assert a == b


# --- injection --------------------------------------------------------------


def test_delay_off_by_default_keeps_stage3_behaviour():
    """The existing 12 injection tests must stay valid."""
    _, mask = build_reference_condition(_row([10]), _ref(), DIM, torch.float32)
    assert _span(mask) == (10, 19)


def test_delay_shifts_the_span_forward_from_the_marker():
    # 4s of lead is up to 50 frames at 12.5 Hz, so the clip must be long enough to hold it —
    # otherwise this exercises suppression instead of the shift.
    _, mask = build_reference_condition(
        _row([10], seq=100), _ref(), DIM, torch.float32,
        leads=[[4.0]], sample_delay=True, rng=random.Random(1),
    )
    first, _ = _span(mask)
    assert first > 10, "reference must arrive after the marker, as it does at serve time"
    assert first <= 10 + int(round(4.0 * FRAME_RATE))


def test_delay_is_bounded_by_the_lead_not_the_clip():
    """A 1.2s lead can never push the reference 3s downstream."""
    rng = random.Random(2)
    for _ in range(200):
        _, mask = build_reference_condition(
            _row([5]), _ref(), DIM, torch.float32,
            leads=[[1.2]], sample_delay=True, rng=rng,
        )
        first, _ = _span(mask)
        assert 5 <= first <= 5 + int(round(1.2 * FRAME_RATE))


def test_a_delay_past_the_next_marker_suppresses_injection():
    """'Retrieval arrived too late' is a valid training case, not an error."""
    _, mask = build_reference_condition(
        _row([10, 12]), [[torch.ones(10, DIM), torch.ones(10, DIM)]], DIM, torch.float32,
        leads=[[8.0, 0.0]], sample_delay=True, rng=random.Random(3),
    )
    on = mask[0, 10:12].sum().item()
    assert on <= 2


def test_missing_lead_falls_back_to_the_measured_median():
    """Lead labelling (§4.2) isn't built yet — the delay must still work without it."""
    _, mask = build_reference_condition(
        _row([10]), _ref(), DIM, torch.float32,
        sample_delay=True, rng=random.Random(4), default_lead=1.2,
    )
    first, _ = _span(mask)
    assert 10 <= first <= 10 + int(round(1.2 * FRAME_RATE))


def test_each_turn_gets_its_own_delay():
    """Two markers must not share one draw — they are independent training instances."""
    firsts = set()
    rng = random.Random(5)
    for _ in range(40):
        _, mask = build_reference_condition(
            _row([2, 22]), [[torch.ones(4, DIM), torch.ones(4, DIM)]], DIM, torch.float32,
            leads=[[1.2, 1.2]], sample_delay=True, rng=rng,
        )
        on = mask[0].nonzero().flatten().tolist()
        firsts.add(tuple(on[:1]))
    assert len(firsts) > 1


# --- dropout ----------------------------------------------------------------


def test_dropout_off_by_default():
    _, mask = build_reference_condition(_row([10]), _ref(), DIM, torch.float32)
    assert mask.sum() > 0


def test_dropout_fires_at_roughly_the_configured_rate():
    rng = random.Random(6)
    dropped = 0
    for _ in range(500):
        _, mask = build_reference_condition(
            _row([10]), _ref(), DIM, torch.float32, dropout=0.2, rng=rng,
        )
        dropped += mask.sum().item() == 0
    assert 0.13 < dropped / 500 < 0.28


def test_dropout_leaves_the_marker_trainable():
    """Dropping the reference must not drop the ⟨ret⟩ from the text stream — the model still
    needs to learn to fire it. Injection is what's suppressed, nothing else."""
    cond, mask = build_reference_condition(
        _row([10]), _ref(), DIM, torch.float32, dropout=1.0, rng=random.Random(7),
    )
    assert mask.sum() == 0
    assert cond.sum() == 0
    assert cond.shape == (1, 40, DIM)

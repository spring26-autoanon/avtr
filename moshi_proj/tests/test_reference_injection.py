import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data.reference_injection import build_reference_condition

RAG = 4
DIM = 3


def _text_row(rows):
    return torch.tensor(rows, dtype=torch.long)


def test_single_reference_lands_at_the_rag_frame():
    text_row = _text_row([[0, 0, RAG, 0, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.float32)
    assert cond.shape == (1, 6, DIM) and mask.shape == (1, 6)
    assert mask.tolist() == [[0, 0, 1, 1, 0, 0]]
    assert torch.equal(cond[0, 2], torch.ones(DIM))
    assert torch.equal(cond[0, 0], torch.zeros(DIM))


def test_second_reference_lands_at_the_second_rag_frame():
    text_row = _text_row([[0, RAG, 0, 0, RAG, 0, 0, 0]])
    a = torch.full((2, DIM), 1.0)
    b = torch.full((2, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b]], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 1, 0, 1, 1, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))
    assert torch.equal(cond[0, 4], torch.full((DIM,), 2.0))


def test_long_reference_is_clamped_at_the_next_rag_frame():
    text_row = _text_row([[RAG, 0, RAG, 0, 0, 0]])
    a = torch.full((10, DIM), 1.0)      # would run past the second marker
    b = torch.full((2, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b]], DIM, torch.float32)
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))
    assert torch.equal(cond[0, 2], torch.full((DIM,), 2.0))   # not overwritten by a


def test_reference_is_truncated_at_sequence_end():
    text_row = _text_row([[0, 0, 0, RAG]])
    ref = torch.ones(5, DIM)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.float32)
    assert mask.tolist() == [[0, 0, 0, 1]]
    assert torch.equal(cond[0, 3], torch.ones(DIM))


def test_example_with_no_references_stays_zero():
    text_row = _text_row([[0, RAG, 0, 0], [0, 0, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[ref], None], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 1, 0], [0, 0, 0, 0]]
    assert torch.equal(cond[1], torch.zeros(4, DIM))


def test_more_markers_than_references_uses_the_pairs_that_match():
    text_row = _text_row([[RAG, 0, RAG, 0]])
    a = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[a]], DIM, torch.float32)
    # first marker conditioned, second left alone rather than reusing a
    assert mask.tolist() == [[1, 1, 0, 0]]


def test_more_references_than_markers_uses_the_pairs_that_match():
    text_row = _text_row([[0, RAG, 0, 0]])
    a = torch.full((1, DIM), 1.0)
    b = torch.full((1, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b]], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))


def test_example_with_no_marker_falls_back_to_the_prefix():
    """Legacy synthetic clips whose <RAG> was stripped still get conditioned at frame 0."""
    text_row = _text_row([[0, 0, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.float32)
    assert mask.tolist() == [[1, 1, 0, 0]]


def test_bare_tensor_is_tolerated_like_a_one_element_list():
    """Belt and braces: a legacy caller passing [T,D] directly must not crash."""
    text_row = _text_row([[0, RAG, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [ref], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 1, 0]]


def test_refs_none_produces_all_zeros():
    text_row = _text_row([[0, RAG, 0, 0]])
    cond, mask = build_reference_condition(text_row, None, DIM, torch.float32)
    assert torch.equal(mask, torch.zeros(1, 4))
    assert torch.equal(cond, torch.zeros(1, 4, DIM))


def test_dtype_and_shape_follow_the_model():
    text_row = _text_row([[RAG, 0]])
    ref = torch.ones(1, DIM, dtype=torch.float32)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.bfloat16)
    assert cond.dtype == torch.bfloat16
    assert cond.shape == (1, 2, DIM)


def test_batch_of_mixed_examples():
    """The real case: a retrieval clip and a plain dialogue window in one batch."""
    text_row = _text_row([[0, RAG, 0, RAG], [0, 0, 0, 0]])
    a = torch.full((1, DIM), 1.0)
    b = torch.full((1, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b], None], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 0, 1], [0, 0, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))
    assert torch.equal(cond[0, 3], torch.full((DIM,), 2.0))

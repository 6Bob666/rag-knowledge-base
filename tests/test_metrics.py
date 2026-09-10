import pytest

from services.metrics import (
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


def test_recall_at_k_returns_zero_when_correct_chunk_is_outside_top_k():
    assert calculate_recall_at_k(["c", "d", "a"], ["a"], 2) == 0


def test_recall_at_k_returns_one_when_correct_chunk_is_inside_top_k():
    assert calculate_recall_at_k(["c", "d", "a"], ["a"], 3) == 1


def test_recall_at_k_returns_zero_for_empty_results():
    assert calculate_recall_at_k([], ["a"], 3) == 0


@pytest.mark.parametrize(
    "returned_ids,relevant_ids,k",
    [([], ["a"], 0), (["a"], [], 3)],
)
def test_recall_at_k_rejects_invalid_inputs(returned_ids, relevant_ids, k):
    with pytest.raises(ValueError):
        calculate_recall_at_k(returned_ids, relevant_ids, k)


def test_reciprocal_rank_uses_the_first_relevant_chunk():
    assert calculate_reciprocal_rank(["c", "a", "b"], ["a", "b"], 3) == 0.5


def test_reciprocal_rank_is_zero_when_no_relevant_chunk_is_found():
    assert calculate_reciprocal_rank(["c", "d"], ["a"], 2) == 0.0


def test_mrr_is_the_average_of_reciprocal_ranks():
    assert calculate_mrr([1.0, 0.5, 0.0]) == 0.5


def test_mrr_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_mrr([])


def test_precision_at_k_counts_relevant_results_in_top_k():
    assert calculate_precision_at_k(["a", "c", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)


def test_precision_at_k_returns_zero_for_empty_results():
    assert calculate_precision_at_k([], ["a"], 3) == 0.0

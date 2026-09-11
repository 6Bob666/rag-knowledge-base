"""索引压测脚本的可测部分：语料构造与召回计算。

真正的压测（建索引、打延迟）不进 CI——它要跑分钟级并且吃内存，
CI 只保证"造数据"和"算指标"这两块逻辑是对的。
"""

import numpy as np
import pytest

from benchmark_index import build_corpus, exact_topk
from services.metrics import recall_at_k_from_ids


def test_build_corpus_returns_normalised_vectors():
    corpus, queries, exact_ids = build_corpus(
        size=300, dim=64, queries=5, intrinsic_dim=8
    )

    assert corpus.shape == (300, 64)
    assert queries.shape == (5, 64)
    assert len(exact_ids) == 5
    assert np.allclose(np.linalg.norm(corpus, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-5)


def test_build_corpus_is_deterministic_for_same_seed():
    first = build_corpus(size=200, dim=32, queries=3, intrinsic_dim=8, seed=42)
    second = build_corpus(size=200, dim=32, queries=3, intrinsic_dim=8, seed=42)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_build_corpus_differs_between_seeds():
    first = build_corpus(size=200, dim=32, queries=3, intrinsic_dim=8, seed=1)
    second = build_corpus(size=200, dim=32, queries=3, intrinsic_dim=8, seed=2)

    assert not np.array_equal(first[0], second[0])


def test_intrinsic_dim_controls_effective_rank():
    """intrinsic_dim 是这份压测的核心变量：它决定数据到底有多难。"""
    low_rank, _, _ = build_corpus(
        size=400, dim=64, queries=2, intrinsic_dim=4, seed=3
    )
    full_rank, _, _ = build_corpus(
        size=400, dim=64, queries=2, intrinsic_dim=64, seed=3
    )

    # 奇异值里"显著大于 0"的个数应当接近设定的内在维度。
    low_singular = np.linalg.svd(low_rank, compute_uv=False)
    full_singular = np.linalg.svd(full_rank, compute_uv=False)
    low_energy = np.sum(low_singular**2)
    low_effective = int(np.sum(np.cumsum(low_singular**2) < 0.999 * low_energy))
    full_effective = int(np.sum(full_singular > 1e-3))

    assert low_effective <= 8
    assert full_effective > 32


def test_build_corpus_rejects_invalid_intrinsic_dim():
    with pytest.raises(ValueError):
        build_corpus(size=10, dim=16, queries=1, intrinsic_dim=0)
    with pytest.raises(ValueError):
        build_corpus(size=10, dim=16, queries=1, intrinsic_dim=17)


def test_build_corpus_rejects_invalid_size():
    with pytest.raises(ValueError):
        build_corpus(size=0, dim=16, queries=1)
    with pytest.raises(ValueError):
        build_corpus(size=10, dim=16, queries=0)


def test_exact_topk_orders_by_similarity():
    # 第一个向量与查询同向，第二个正交，第三个接近第一个。
    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = np.array([[1.0, 0.0]], dtype=np.float32)

    top = exact_topk(query, corpus, k=3)

    assert list(top[0]) == [0, 2, 1]


def test_exact_topk_rejects_non_positive_k():
    corpus = np.eye(3, dtype=np.float32)

    with pytest.raises(ValueError):
        exact_topk(corpus, corpus, k=0)


def test_recall_at_k_from_ids_counts_overlap():
    returned = ["a", "b", "c"]
    exact = ["c", "d", "e"]

    assert recall_at_k_from_ids(returned, exact, 1) == 0.0
    assert recall_at_k_from_ids(returned, exact, 2) == 0.0
    assert recall_at_k_from_ids(returned, exact, 3) == pytest.approx(1 / 3)
    assert recall_at_k_from_ids(returned, exact, 5) == pytest.approx(1 / 3)


def test_recall_at_k_from_ids_is_one_for_identical_order():
    ids = ["a", "b", "c", "d"]

    assert recall_at_k_from_ids(ids, ids, 4) == 1.0

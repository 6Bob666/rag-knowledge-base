from services.reranker import ReRanker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, show_progress_bar=False):
        return self.scores[:len(pairs)]


def test_reranker_sorts_by_score():
    reranker = ReRanker(model=FakeCrossEncoder([0.2, 0.95, 0.6]), score_threshold=0.0)
    docs = ["无关内容", "最相关内容", "一般相关内容"]
    assert reranker.rerank("问题", docs, top_k=2) == ["最相关内容", "一般相关内容"]


def test_reranker_filters_low_scores():
    reranker = ReRanker(model=FakeCrossEncoder([0.1, 0.8, -0.2]), score_threshold=0.5)
    assert reranker.rerank("问题", ["低分", "合格", "拒答"], top_k=3) == ["合格"]


def test_reranker_respects_top_k():
    reranker = ReRanker(model=FakeCrossEncoder([0.9, 0.8, 0.7]), score_threshold=0.0)
    assert len(reranker.rerank("问题", ["a", "b", "c"], top_k=1)) == 1


def test_reranker_batches_multiple_queries_and_preserves_groups():
    reranker = ReRanker(
        model=FakeCrossEncoder([0.2, 0.95, 0.6, 0.8]),
        score_threshold=0.0,
    )
    groups = reranker.rerank_records_batch(
        ["问题一", "问题二"],
        [
            [{"id": "a", "text": "文档 a"}, {"id": "b", "text": "文档 b"}],
            [{"id": "c", "text": "文档 c"}, {"id": "d", "text": "文档 d"}],
        ],
        top_k=1,
    )
    assert [[item["id"] for item in group] for group in groups] == [["b"], ["d"]]

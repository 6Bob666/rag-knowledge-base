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

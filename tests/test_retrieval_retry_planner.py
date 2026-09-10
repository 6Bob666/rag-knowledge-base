from services.retrieval_retry_planner import plan_attempt_queries


def test_plan_returns_rewritten_then_original_when_different():
    queries = plan_attempt_queries(
        question="它有哪些应用？",
        rewritten_query="CNN 有哪些应用？",
    )
    assert queries == ["CNN 有哪些应用？", "它有哪些应用？"]


def test_plan_returns_single_query_when_rewritten_equals_original():
    queries = plan_attempt_queries(
        question="CNN 有哪些应用？",
        rewritten_query="CNN 有哪些应用？",
    )
    assert queries == ["CNN 有哪些应用？"]


def test_plan_uses_original_when_rewritten_missing():
    queries = plan_attempt_queries(question="CNN 是什么？")
    assert queries == ["CNN 是什么？"]

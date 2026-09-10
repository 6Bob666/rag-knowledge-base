from analyze_rag_logs import (
    aggregate_agent_metrics,
    aggregate_rag_logs,
    parse_rag_log_line,
)


def test_parse_rag_log_line_returns_fields():
    line = (
        "2026-09-06 20:00:00 | INFO | rag | rag_answer "
        "request_id=abc123 | conversation_id=demo-001 | status=success | "
        "candidate_count=6 | context_count=3 | rewrite_ms=12.3 | "
        "retrieval_ms=45.6 | rerank_ms=78.9 | llm_ms=1500.2 | "
        "total_ms=1800.4"
    )

    parsed = parse_rag_log_line(line)

    assert parsed is not None
    assert parsed["request_id"] == "abc123"
    assert parsed["conversation_id"] == "demo-001"
    assert parsed["status"] == "success"
    assert parsed["candidate_count"] == 6
    assert parsed["context_count"] == 3
    assert parsed["rewrite_ms"] == 12.3
    assert parsed["total_ms"] == 1800.4


def test_parse_rag_log_line_ignores_unrelated_lines():
    assert parse_rag_log_line("INFO | rag | 开始启动 RAG 应用") is None
    assert parse_rag_log_line("DEBUG | rag | request_id=abc123") is None


def test_parse_rag_log_line_returns_agent_fields():
    line = (
        "INFO | rag | rag_answer request_id=abc123 | conversation_id=- | "
        "status=success | candidate_count=2 | context_count=2 | "
        "route_action=retrieve | decision=success | attempt=2 | "
        "max_attempts=2 | search_count=2 | retrieval_ms=45.6 | "
        "rerank_ms=78.9 | total_ms=200.0"
    )

    parsed = parse_rag_log_line(line)

    assert parsed["route_action"] == "retrieve"
    assert parsed["decision"] == "success"
    assert parsed["attempt"] == 2
    assert parsed["max_attempts"] == 2
    assert parsed["search_count"] == 2


def test_aggregate_rag_logs_counts_and_averages_by_status():
    success_line = (
        "INFO | rag | rag_answer request_id={rid} | conversation_id=- | "
        "status=success | candidate_count=1 | context_count=1 | "
        "rewrite_ms=10.0 | retrieval_ms=20.0 | rerank_ms=30.0 | "
        "llm_ms=100.0 | total_ms=200.0"
    )
    error_line = (
        "INFO | rag | rag_stream request_id={rid} | conversation_id=- | "
        "status=llm_error | candidate_count=1 | context_count=1 | "
        "rewrite_ms=10.0 | retrieval_ms=20.0 | rerank_ms=30.0 | "
        "llm_ms=50.0 | total_ms=120.0"
    )
    lines = [
        success_line.format(rid="a"),
        success_line.format(rid="b").replace("llm_ms=100.0", "llm_ms=300.0").replace(
            "total_ms=200.0", "total_ms=400.0"
        ),
        error_line.format(rid="c"),
    ]

    summary = aggregate_rag_logs(lines)

    assert summary["success"]["count"] == 2
    assert summary["success"]["avg_llm_ms"] == 200.0
    assert summary["success"]["avg_total_ms"] == 300.0
    assert summary["success"]["max_total_ms"] == 400.0
    assert summary["llm_error"]["count"] == 1
    assert summary["llm_error"]["avg_total_ms"] == 120.0


def test_aggregate_agent_metrics_counts_route_actions_and_attempts():
    def agent_line(rid, route_action, status, attempt, search_count):
        return (
            f"INFO | rag | rag_answer request_id={rid} | conversation_id=- | "
            f"status={status} | candidate_count=1 | context_count=1 | "
            f"route_action={route_action} | decision={status} | "
            f"attempt={attempt} | max_attempts=2 | search_count={search_count} | "
            "retrieval_ms=10.0 | total_ms=100.0"
        )

    lines = [
        agent_line("a", "retrieve", "success", 1, 1),
        agent_line("b", "retrieve", "success", 2, 2),
        agent_line("c", "direct_answer", "direct_answer", 0, 0),
        agent_line("d", "retrieve", "no_documents", 2, 2),
    ]

    metrics = aggregate_agent_metrics(lines)

    assert metrics["total_requests"] == 4
    assert metrics["route_actions"] == {
        "retrieve": 3,
        "direct_answer": 1,
    }
    assert metrics["status"]["success"]["count"] == 2
    assert metrics["status"]["success"]["avg_attempt"] == 1.5
    assert metrics["status"]["success"]["avg_search_count"] == 1.5
    assert metrics["status"]["no_documents"]["count"] == 1

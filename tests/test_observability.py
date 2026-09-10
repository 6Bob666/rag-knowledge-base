from routers.chat import build_rag_trace, new_request_id


def test_new_request_id_is_unique_and_short():
    first = new_request_id()
    second = new_request_id()
    assert first != second
    assert len(first) == 12
    assert len(second) == 12


def test_build_rag_trace_includes_timing_fields_without_question_or_answer():
    trace = build_rag_trace(
        request_id="req-abc-123",
        conversation_id="conv-001",
        status="success",
        candidate_count=6,
        context_count=3,
        timings={
            "rewrite_ms": 12.3,
            "retrieval_ms": 45.6,
            "rerank_ms": 78.9,
            "llm_ms": 1500.2,
            "total_ms": 1800.4,
        },
    )

    assert "request_id=req-abc-123" in trace
    assert "conversation_id=conv-001" in trace
    assert "status=success" in trace
    assert "candidate_count=6" in trace
    assert "context_count=3" in trace
    assert "rewrite_ms=12.3" in trace
    assert "llm_ms=1500.2" in trace
    assert "total_ms=1800.4" in trace
    assert "问题" not in trace


def test_build_rag_trace_includes_agent_extra_fields():
    trace = build_rag_trace(
        request_id="req-abc-123",
        conversation_id=None,
        status="success",
        candidate_count=2,
        context_count=2,
        timings={"total_ms": 123.4},
        extra_fields={
            "route_action": "retrieve",
            "decision": "success",
            "attempt": 2,
            "max_attempts": 2,
            "search_count": 2,
        },
    )

    assert "route_action=retrieve" in trace
    assert "decision=success" in trace
    assert "attempt=2" in trace
    assert "max_attempts=2" in trace
    assert "search_count=2" in trace

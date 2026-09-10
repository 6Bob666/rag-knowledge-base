import pytest
from pydantic import ValidationError

from services.agent_router import route_question
from services.agent_state import AgentState, RouteDecision


def test_route_decision_accepts_allowed_actions():
    for action in ("retrieve", "direct_answer", "refuse"):
        decision = RouteDecision(action=action)
        assert decision.action == action


def test_route_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        RouteDecision(action="search_all")


def test_greeting_routes_to_direct_answer():
    decision = route_question("你好")
    assert decision.action == "direct_answer"
    assert decision.confidence >= 0.5


def test_self_introduction_routes_to_direct_answer():
    decision = route_question("介绍一下你自己")
    assert decision.action == "direct_answer"


def test_factual_question_routes_to_retrieve():
    decision = route_question("CNN 的全称是什么？")
    assert decision.action == "retrieve"


def test_question_without_obvious_marker_defaults_to_retrieve():
    decision = route_question("请告诉我这个系统怎么用")
    assert decision.action == "retrieve"


def test_empty_question_routes_to_refuse():
    decision = route_question("   ")
    assert decision.action == "refuse"


def test_agent_state_limits_retry_attempts():
    state = AgentState(
        request_id="req-1",
        question="问题",
        max_attempts=2,
    )
    assert state.can_retry() is True
    state.attempt = 1
    assert state.can_retry() is True
    state.attempt = 2
    assert state.can_retry() is False


def test_agent_state_snapshot_contains_summary_only():
    state = AgentState(
        request_id="req-1",
        question="问题",
        max_attempts=2,
    )
    state.attempt = 1
    state.route_action = "retrieve"
    state.search_queries.append("改写后的问题")
    state.candidates = [{"text": "很长很长的文档内容"}]

    snapshot = state.snapshot()
    assert snapshot["attempt"] == 1
    assert snapshot["max_attempts"] == 2
    assert snapshot["search_queries"] == ["改写后的问题"]
    assert snapshot["candidate_count"] == 1
    assert snapshot["context_count"] == 0
    # 摘要不应把完整文档文本带出来，避免日志/评测泄露数据。
    assert "很长很长的文档内容" not in str(snapshot)

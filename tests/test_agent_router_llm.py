import pytest

from config import settings
from services.agent_router import (
    parse_route_json,
    route_question,
    route_question_with_llm,
)


def test_parse_route_json_extracts_json_from_mixed_text():
    text = (
        "好的，我的判断如下：\n"
        '{"action": "retrieve", "query": "CNN 有哪些典型应用", '
        '"reason": "需要知识库事实", "confidence": 0.92}'
    )

    decision = parse_route_json(text)

    assert decision.action == "retrieve"
    assert decision.query == "CNN 有哪些典型应用"
    assert decision.confidence == 0.92


def test_parse_route_json_treats_empty_query_as_none():
    decision = parse_route_json(
        '{"action": "retrieve", "query": "", "confidence": 0.8}'
    )
    assert decision.query is None


def test_parse_route_json_rejects_missing_action():
    with pytest.raises(ValueError):
        parse_route_json('{"reason": "缺少 action"}')


def test_parse_route_json_rejects_unknown_action():
    with pytest.raises(ValueError):
        parse_route_json('{"action": "search_all"}')


def test_parse_route_json_rejects_non_json_text():
    with pytest.raises(ValueError):
        parse_route_json("这不是 JSON")


def test_llm_router_uses_valid_llm_decision():
    def fake_llm(question, history):
        return (
            '{"action": "direct_answer", "reason": "LLM 判断为闲聊", '
            '"confidence": 0.99}'
        )

    # 规则会把“CNN 是什么”判为检索，因此能证明这里用的是 LLM 的决策。
    decision = route_question_with_llm(
        "CNN 是什么？",
        history=[],
        llm_call=fake_llm,
    )

    assert decision.action == "direct_answer"
    assert decision.reason == "LLM 判断为闲聊"


def test_llm_router_falls_back_to_rules_when_output_is_invalid():
    def fake_llm(question, history):
        return "抱歉，我无法给出 JSON。"

    decision = route_question_with_llm(
        "CNN 是什么？",
        history=[],
        llm_call=fake_llm,
    )

    assert decision.action == route_question("CNN 是什么？").action
    assert decision.action == "retrieve"


def test_llm_router_skips_llm_and_refuses_empty_question():
    def fake_llm(question, history):
        raise AssertionError("空问题不应调用 LLM")

    decision = route_question_with_llm(
        "   ",
        history=[],
        llm_call=fake_llm,
    )

    assert decision.action == "refuse"


def test_get_agent_router_respects_settings_mode(monkeypatch):
    from routers.chat import get_agent_router

    monkeypatch.setattr(settings, "agent_router_mode", "rules")
    assert get_agent_router() is route_question

    monkeypatch.setattr(settings, "agent_router_mode", "llm")
    assert get_agent_router() is route_question_with_llm

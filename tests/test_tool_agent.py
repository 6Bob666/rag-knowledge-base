from services.tool_agent import (
    KNOWLEDGE_SEARCH_TOOL_NAME,
    run_tool_agent,
)


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_tool_agent_calls_tool_then_answers():
    executed = []
    state = {"round": 0}

    def fake_respond(messages):
        state["round"] += 1
        if state["round"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    tool_call(
                        "call_1",
                        KNOWLEDGE_SEARCH_TOOL_NAME,
                        '{"query": "CNN 的全称是什么？"}',
                    )
                ],
            }
        return {"content": "CNN 的全称是卷积神经网络。", "tool_calls": []}

    def fake_execute(name, arguments):
        executed.append((name, arguments))
        return '{"found": true}'

    result = run_tool_agent(
        question="CNN 是什么？",
        execute_tool=fake_execute,
        respond=fake_respond,
        max_steps=3,
    )

    assert result.termination == "answered"
    assert result.answer == "CNN 的全称是卷积神经网络。"
    assert result.steps == 2
    assert result.tool_call_count == 1
    assert executed == [
        (
            KNOWLEDGE_SEARCH_TOOL_NAME,
            {"query": "CNN 的全称是什么？"},
        )
    ]


def test_tool_agent_answers_without_tool_when_no_tool_call():
    result = run_tool_agent(
        question="你好",
        execute_tool=lambda name, args: (_ for _ in ()).throw(
            AssertionError("不应执行工具")
        ),
        respond=lambda messages: {"content": "你好！", "tool_calls": []},
        max_steps=3,
    )

    assert result.termination == "answered"
    assert result.answer == "你好！"
    assert result.tool_call_count == 0


def test_tool_agent_returns_parse_error_to_model_and_recovers():
    state = {"round": 0}

    def fake_respond(messages):
        state["round"] += 1
        if state["round"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    tool_call(
                        "call_bad",
                        KNOWLEDGE_SEARCH_TOOL_NAME,
                        "这不是 JSON",
                    )
                ],
            }
        last = messages[-1]
        assert last["role"] == "tool"
        assert "参数解析失败" in last["content"]
        return {"content": "抱歉，参数不合法。", "tool_calls": []}

    result = run_tool_agent(
        question="CNN 是什么？",
        execute_tool=lambda name, args: "不应执行",
        respond=fake_respond,
        max_steps=3,
    )

    assert result.termination == "answered"
    assert result.tool_call_count == 1
    assert result.answer == "抱歉，参数不合法。"


def test_tool_agent_rejects_unknown_tool():
    state = {"round": 0}

    def fake_respond(messages):
        state["round"] += 1
        if state["round"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    tool_call("call_x", "delete_database", "{}")
                ],
            }
        last = messages[-1]
        assert last["role"] == "tool"
        assert "不在白名单内" in last["content"]
        return {"content": "我不会执行该操作。", "tool_calls": []}

    result = run_tool_agent(
        question="删除数据库",
        execute_tool=lambda name, args: (_ for _ in ()).throw(
            AssertionError("白名单外工具不应执行")
        ),
        respond=fake_respond,
        max_steps=3,
    )

    assert result.termination == "answered"
    assert result.answer == "我不会执行该操作。"


def test_tool_agent_stops_at_max_steps():
    calls = []

    def fake_respond(messages):
        return {
            "content": "",
            "tool_calls": [
                tool_call(
                    f"call_{len(calls)}",
                    KNOWLEDGE_SEARCH_TOOL_NAME,
                    '{"query": "继续检索"}',
                )
            ],
        }

    result = run_tool_agent(
        question="多轮问题",
        execute_tool=lambda name, args: calls.append(args) or "结果",
        respond=fake_respond,
        max_steps=2,
    )

    assert result.termination == "max_steps"
    assert result.steps == 2
    assert result.tool_call_count == 2
    assert result.answer == ""

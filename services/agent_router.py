"""Agent Router：先用规则做零成本兜底，也可切换为 LLM 结构化决策。

生产环境可以换成 LLM 结构化输出，但必须保留规则兜底：
任何不确定的问题都默认走检索，避免绕过知识库造成幻觉。
"""

import json
from typing import Callable

from logging_config import logger
from services.agent_state import RouteDecision
from services.llm_service import MODEL_NAME, client


# 无需知识库的闲聊 / 通用任务。
DIRECT_ANSWER_MARKERS = (
    "你好",
    "您好",
    "哈喽",
    "hello",
    "hi",
    "你是谁",
    "你叫什么",
    "介绍一下你自己",
    "你会做什么",
    "你能做什么",
    "谢谢",
    "感谢",
    "再见",
    "翻译",
)

# 命中这些词的知识库问题，可以放心地默认检索。
RETRIEVAL_MARKERS = (
    "什么",
    "哪些",
    "怎么",
    "如何",
    "为什么",
    "区别",
    "定义",
    "含义",
    "原理",
    "介绍",
    "作用",
    "应用",
    "流程",
    "规定",
    "要求",
    "步骤",
)

ROUTER_SYSTEM_PROMPT = """你是一个 RAG 系统的路由判断器。
你需要判断当前用户问题是否需要检索知识库，并输出一个 JSON 对象，不要输出任何解释。

action 只能是以下三种之一：
1. "retrieve"：问题需要知识库事实，例如询问概念、定义、流程、应用。
2. "direct_answer"：无需知识库，例如闲聊、寒暄、翻译、改写、自我介绍。
3. "refuse"：问题明显超出系统能力，或属于危险/违规请求。

如果 action="retrieve"：
- 如果原始问题已经清晰完整，query 填 null；
- 如果原始问题有指代不明、错别字或表达不清，query 给出更适合作检索的问法。

输出格式示例：
{"action": "retrieve", "query": null, "reason": "需要知识库事实", "confidence": 0.9}
"""


def _build_router_messages(
    question: str,
    history: list[dict[str, str]] | None,
) -> list[dict]:
    history_text = "\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in (history or [])
    ) or "（无历史对话）"
    user_prompt = f"""历史对话：
{history_text}

当前用户问题：{question}

请只输出 JSON："""
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_route_json(text: str) -> RouteDecision:
    """从 LLM 输出中解析 RouteDecision；格式或动作不合法时抛出 ValueError。"""
    if not text or not isinstance(text, str):
        raise ValueError("Router 返回了空内容")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Router 输出中没有 JSON 对象")
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Router JSON 解析失败") from exc
    if not isinstance(raw, dict):
        raise ValueError("Router JSON 顶层必须是对象")
    # query 为空字符串时按未提供处理。
    if not raw.get("query"):
        raw.pop("query", None)
    try:
        return RouteDecision(**raw)
    except Exception as exc:
        raise ValueError(f"Router 决策字段不合法: {exc}") from exc


def _call_llm_router(
    question: str,
    history: list[dict[str, str]] | None,
) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=_build_router_messages(question, history),
        temperature=0.0,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def route_question(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> RouteDecision:
    """根据问题决定是否需要检索。

    history 参数保留给将来的 LLM Router，规则版本暂不使用。
    """
    text = (question or "").strip()
    lowered = text.lower()

    if not text:
        return RouteDecision(
            action="refuse",
            reason="空问题",
            confidence=1.0,
        )

    if any(marker in lowered for marker in DIRECT_ANSWER_MARKERS):
        return RouteDecision(
            action="direct_answer",
            reason="命中无需检索的闲聊/通用任务",
            confidence=0.9,
        )

    if any(marker in text for marker in RETRIEVAL_MARKERS):
        return RouteDecision(
            action="retrieve",
            reason="命中知识库问题特征词",
            confidence=0.8,
        )

    # 安全兜底：拿不准时默认检索，检索不到再拒答，而不是直接让模型凭记忆回答。
    return RouteDecision(
        action="retrieve",
        reason="未命中直接回答规则，默认检索以降低幻觉",
        confidence=0.6,
    )


def route_question_with_llm(
    question: str,
    history: list[dict[str, str]] | None = None,
    llm_call: Callable | None = None,
) -> RouteDecision:
    """用 LLM 输出结构化路由决策；任何失败都回退到规则 Router。"""
    text = (question or "").strip()
    if not text:
        return RouteDecision(
            action="refuse",
            reason="空问题",
            confidence=1.0,
        )

    try:
        content = (
            llm_call(question, history)
            if llm_call is not None
            else _call_llm_router(question, history)
        )
        return parse_route_json(content)
    except Exception as exc:
        logger.warning("LLM Router 决策失败，回退规则 Router: %s", exc)
        return route_question(question, history)

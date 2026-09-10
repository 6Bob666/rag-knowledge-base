"""Agentic RAG 请求的一次状态与结构化路由决策。

这个模块只定义数据结构，不调用模型，方便单元测试和后续扩展。
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator


ALLOWED_ROUTE_ACTIONS = ("retrieve", "direct_answer", "refuse")


class RouteDecision(BaseModel):
    """Router 的结构化决策。

    action:
        retrieve      需要检索知识库
        direct_answer 不需要检索，直接生成（闲聊、翻译等）
        refuse        明确拒绝回答
    query:
        Router 如果希望用另一种问法检索，可以给出；None 表示继续走查询改写。
    """

    action: str = Field(..., description="retrieve / direct_answer / refuse")
    query: str | None = Field(
        default=None,
        description="Router 建议的检索问法；None 表示沿用后续查询改写",
    )
    reason: str = Field(default="", description="记录决策原因，便于评测与日志")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("action")
    @classmethod
    def _action_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_ROUTE_ACTIONS:
            raise ValueError(f"未知路由动作: {value!r}")
        return value


@dataclass
class AgentState:
    """记录一次 RAG 请求的完整状态。

    状态机简化表达为：
    route -> retrieve -> check -> generate / refuse
    任何分支都必须受 attempt/max_attempts 限制。
    """

    request_id: str
    question: str
    conversation_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    # 有界循环：attempt 从 1 开始，达到 max_attempts 后不再允许新的检索。
    max_attempts: int = 2
    attempt: int = 0

    # Router 决策快照
    route_action: str = "retrieve"
    route_reason: str = ""
    route_confidence: float = 0.0

    # 检索过程
    search_queries: list[str] = field(default_factory=list)
    current_query: str | None = None
    candidates: list[dict] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)

    # 当前所处状态与最终结果
    decision: str = "start"
    answer: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    failure_status: str | None = None
    error: str | None = None

    def can_retry(self) -> bool:
        """是否还有剩余的检索次数。"""
        return self.attempt < self.max_attempts

    def snapshot(self) -> dict:
        """输出状态摘要，供日志和测试使用，不包含完整文档文本。"""
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "route_action": self.route_action,
            "route_reason": self.route_reason,
            "route_confidence": self.route_confidence,
            "search_queries": list(self.search_queries),
            "candidate_count": len(self.candidates),
            "context_count": len(self.contexts),
            "decision": self.decision,
            "failure_status": self.failure_status,
        }

"""Token 用量与成本核算。

一次 RAG 请求可能调用多次大模型（查询改写、生成、Router、Agent 循环），
成本要按"整条请求"来算，所以这里提供三样东西：

    TokenUsage     单次调用的 token 数
    UsageTracker   一次请求内多次调用的累加器
    上下文变量      让 llm_service 不用改签名就能记到当前请求上

价格从配置读取，只做量级估算：不同模型、不同缓存命中价格差别很大，
看板上的成本用于发现"哪个接口在偷偷烧钱"，不适合当账单。
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from config import settings


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_cny": estimate_cost_cny(self),
        }


def estimate_cost_cny(
    usage: TokenUsage,
    *,
    price_input_per_million: float | None = None,
    price_output_per_million: float | None = None,
) -> float:
    """按百万 token 单价估算成本（人民币，保留 6 位小数）。"""
    price_in = (
        settings.llm_price_input_per_million
        if price_input_per_million is None
        else price_input_per_million
    )
    price_out = (
        settings.llm_price_output_per_million
        if price_output_per_million is None
        else price_output_per_million
    )
    cost = (
        usage.prompt_tokens * price_in + usage.completion_tokens * price_out
    ) / 1_000_000
    return round(cost, 6)


def usage_from_response(response: Any) -> TokenUsage | None:
    """从 SDK 响应里取出 token 用量；拿不到就返回 None。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and completion is None:
        return None
    return TokenUsage(
        prompt_tokens=int(prompt or 0),
        completion_tokens=int(completion or 0),
    )


class UsageTracker:
    """一次请求内的用量累加器，可安全地被多个线程写入。"""

    def __init__(self) -> None:
        self._usage = TokenUsage()
        self._calls = 0

    def add(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self._usage = self._usage + usage
        self._calls += 1

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    @property
    def llm_calls(self) -> int:
        return self._calls

    def to_dict(self) -> dict:
        return {"llm_calls": self._calls, **self._usage.to_dict()}


_current_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "rag_usage_tracker", default=None
)


def start_usage_tracking() -> UsageTracker:
    """为当前请求开启用量统计，返回累加器。"""
    tracker = UsageTracker()
    _current_tracker.set(tracker)
    return tracker


def use_tracker(tracker: UsageTracker | None) -> None:
    """在另一个执行上下文（例如流式生成器）里继续累加同一个请求。"""
    _current_tracker.set(tracker)


def record_usage(usage: TokenUsage | None) -> None:
    """记录一次模型调用的用量：累加到当前请求，并更新全局指标。"""
    if usage is None:
        return
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.add(usage)
    from services.metrics_registry import metrics_registry

    metrics_registry.record_llm_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


def record_response_usage(response: Any) -> None:
    """SDK 响应 → 用量统计，调用点一行就够。"""
    record_usage(usage_from_response(response))

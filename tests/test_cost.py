"""token 用量与成本核算的单元测试。"""

import pytest

from services.cost import (
    TokenUsage,
    UsageTracker,
    estimate_cost_cny,
    record_response_usage,
    record_usage,
    start_usage_tracking,
    usage_from_response,
    use_tracker,
)
from services.metrics_registry import metrics_registry


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, usage):
        self.usage = usage


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics_registry.reset()
    yield
    metrics_registry.reset()


def test_token_usage_adds_up():
    total = TokenUsage(10, 5) + TokenUsage(2, 3)

    assert total.prompt_tokens == 12
    assert total.completion_tokens == 8
    assert total.total_tokens == 20


def test_estimate_cost_uses_separate_input_and_output_prices():
    # 输入 2 元/百万，输出 8 元/百万：100 万输入 + 100 万输出 = 10 元
    cost = estimate_cost_cny(
        TokenUsage(1_000_000, 1_000_000),
        price_input_per_million=2.0,
        price_output_per_million=8.0,
    )

    assert cost == 10.0


def test_estimate_cost_is_tiny_for_single_request():
    cost = estimate_cost_cny(
        TokenUsage(1500, 300),
        price_input_per_million=2.0,
        price_output_per_million=8.0,
    )

    assert cost == pytest.approx(0.0054, abs=1e-9)


def test_usage_from_response_reads_openai_shape():
    usage = usage_from_response(FakeResponse(FakeUsage(120, 30)))

    assert usage == TokenUsage(120, 30)


def test_usage_from_response_returns_none_when_absent():
    assert usage_from_response(FakeResponse(None)) is None
    assert usage_from_response(object()) is None


def test_usage_tracker_accumulates_calls_and_tokens():
    tracker = UsageTracker()
    tracker.add(TokenUsage(10, 2))
    tracker.add(TokenUsage(20, 4))
    tracker.add(None)

    assert tracker.llm_calls == 2
    assert tracker.usage == TokenUsage(30, 6)
    assert tracker.to_dict()["total_tokens"] == 36


def test_record_usage_updates_current_request_and_global_metrics():
    tracker = start_usage_tracking()

    record_usage(TokenUsage(100, 20))
    record_response_usage(FakeResponse(FakeUsage(50, 10)))

    assert tracker.usage == TokenUsage(150, 30)
    snapshot = metrics_registry.snapshot()
    assert snapshot["prompt_tokens"] == 150
    assert snapshot["completion_tokens"] == 30
    assert snapshot["llm_calls"] == 2
    assert snapshot["cost_cny"] > 0


def test_record_usage_without_active_tracker_only_updates_metrics():
    use_tracker(None)

    record_usage(TokenUsage(10, 5))

    snapshot = metrics_registry.snapshot()
    assert snapshot["prompt_tokens"] == 10


def test_use_tracker_lets_another_context_keep_accumulating():
    """流式生成器在别的上下文里跑，要能接回同一个请求的累加器。"""
    tracker = start_usage_tracking()
    record_usage(TokenUsage(10, 1))

    use_tracker(None)
    record_usage(TokenUsage(999, 999))
    assert tracker.usage == TokenUsage(10, 1)

    use_tracker(tracker)
    record_usage(TokenUsage(5, 5))
    assert tracker.usage == TokenUsage(15, 6)

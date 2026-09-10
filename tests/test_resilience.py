import pytest

from services.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    FixedWindowRateLimiter,
    RateLimitExceeded,
    retry_call,
)


def test_fixed_window_rate_limiter_rejects_after_limit():
    limiter = FixedWindowRateLimiter()
    assert limiter.check("user-1", 2, 60) == 1
    assert limiter.check("user-1", 2, 60) == 0
    with pytest.raises(RateLimitExceeded) as error:
        limiter.check("user-1", 2, 60)
    assert error.value.retry_after >= 1
    assert limiter.check("user-2", 2, 60) == 1


def test_retry_call_uses_exponential_backoff_and_stops():
    calls = []
    sleeps = []

    def failing_call():
        calls.append(1)
        raise ConnectionError("temporary")

    with pytest.raises(ConnectionError):
        retry_call(
            failing_call,
            max_attempts=3,
            backoff_seconds=0.1,
            sleep_func=sleeps.append,
        )
    assert len(calls) == 3
    assert sleeps == [0.1, 0.2]


def test_retry_call_returns_after_transient_failure():
    state = {"count": 0}

    def flaky_call():
        state["count"] += 1
        if state["count"] == 1:
            raise ConnectionError("temporary")
        return "ok"

    assert retry_call(
        flaky_call,
        max_attempts=2,
        sleep_func=lambda _: None,
    ) == "ok"
    assert state["count"] == 2


def test_retry_call_does_not_retry_non_retryable_error():
    calls = []

    def bad_request():
        calls.append(1)
        raise ValueError("invalid request")

    with pytest.raises(ValueError):
        retry_call(
            bad_request,
            max_attempts=3,
            retry_if=lambda exc: isinstance(exc, ConnectionError),
            sleep_func=lambda _: None,
        )
    assert len(calls) == 1


def test_circuit_breaker_opens_and_recovers():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0.01)

    with pytest.raises(ConnectionError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(ConnectionError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not run")

    import time

    time.sleep(0.02)
    assert breaker.call(lambda: "recovered") == "recovered"
    assert breaker.state == "closed"

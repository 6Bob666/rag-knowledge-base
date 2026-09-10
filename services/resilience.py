"""外部依赖调用的通用稳定性组件。"""

from threading import Lock
from time import monotonic, sleep
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimitExceeded(RuntimeError):
    def __init__(self, retry_after: int):
        super().__init__("请求频率超过限制")
        self.retry_after = retry_after


class FixedWindowRateLimiter:
    """固定窗口限流器；适合单进程，分布式部署应换成 Redis Lua 实现。"""

    def __init__(self):
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> int:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit 和 window_seconds 必须大于 0")
        now = monotonic()
        with self._lock:
            started_at, count = self._windows.get(key, (now, 0))
            if now - started_at >= window_seconds:
                started_at, count = now, 0
            if count >= limit:
                retry_after = max(1, int(window_seconds - (now - started_at)))
                raise RateLimitExceeded(retry_after)
            self._windows[key] = (started_at, count + 1)
            return limit - count - 1

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()


def retry_call(
    func: Callable[[], T],
    *,
    max_attempts: int = 2,
    backoff_seconds: float = 0.2,
    retry_if: Callable[[Exception], bool] | None = None,
    sleep_func: Callable[[float], None] = sleep,
) -> T:
    """有限次重试；max_attempts 表示总调用次数，不是额外重试次数。"""
    if max_attempts <= 0:
        raise ValueError("max_attempts 必须大于 0")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds 不能小于 0")
    retry_if = retry_if or (lambda _: True)
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_attempts or not retry_if(exc):
                raise
            sleep_func(backoff_seconds * (2**attempt))
    raise last_error  # pragma: no cover


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """简单熔断器：closed -> open -> half_open -> closed。"""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30):
        if failure_threshold <= 0 or recovery_seconds <= 0:
            raise ValueError("熔断参数必须大于 0")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.state = "closed"
        self.failure_count = 0
        self.opened_at = 0.0
        self._lock = Lock()

    def before_call(self) -> None:
        with self._lock:
            if self.state != "open":
                return
            if monotonic() - self.opened_at >= self.recovery_seconds:
                self.state = "half_open"
                return
            raise CircuitOpenError("外部依赖熔断中，请稍后重试")

    def on_success(self) -> None:
        with self._lock:
            self.state = "closed"
            self.failure_count = 0

    def on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.opened_at = monotonic()

    def call(self, func: Callable[[], T]) -> T:
        self.before_call()
        try:
            value = func()
        except Exception:
            self.on_failure()
            raise
        self.on_success()
        return value

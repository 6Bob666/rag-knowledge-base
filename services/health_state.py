"""应用存活与就绪状态。

状态对象不负责加载模型，只记录 lifespan 已完成的初始化结果，
这样健康检查可以被单独测试，也不会在每次探针请求时重复加载模型。
"""

from threading import Lock


class HealthState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._status = "starting"
        self._checks: dict[str, dict[str, str | bool]] = {}

    def mark_starting(self) -> None:
        with self._lock:
            self._status = "starting"
            self._checks = {}

    def mark_dependency_ready(self, name: str) -> None:
        with self._lock:
            self._checks[name] = {"ok": True, "detail": "ready"}

    def mark_failed(self, name: str, detail: str) -> None:
        with self._lock:
            self._checks[name] = {"ok": False, "detail": detail}
            self._status = "failed"

    def mark_ready(self) -> None:
        with self._lock:
            self._status = "ready"

    def snapshot(self) -> dict:
        with self._lock:
            checks = {name: dict(value) for name, value in self._checks.items()}
            return {
                "status": self._status,
                "checks": checks,
            }

    def is_ready(self) -> bool:
        with self._lock:
            return self._status == "ready"


health_state = HealthState()

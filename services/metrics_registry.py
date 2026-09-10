"""轻量 Prometheus 兼容指标注册器。

生产环境可以替换为 prometheus-client；当前实现零外部依赖，
用于把请求量、错误率和延迟指标链路跑通。
"""

from collections import defaultdict, deque
from threading import Lock
from time import monotonic


class MetricsRegistry:
    def __init__(self, max_samples_per_key: int = 1000):
        if max_samples_per_key <= 0:
            raise ValueError("max_samples_per_key 必须大于 0")
        self.max_samples_per_key = max_samples_per_key
        self._counts: defaultdict[tuple[str, str, int], int] = defaultdict(int)
        self._durations: defaultdict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples_per_key)
        )
        self._lock = Lock()

    def record_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        key = (method.upper(), path, int(status_code))
        duration_key = (method.upper(), path)
        with self._lock:
            self._counts[key] += 1
            self._durations[duration_key].append(float(duration_ms))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "request_counts": dict(self._counts),
                "durations": {
                    key: list(values) for key, values in self._durations.items()
                },
            }

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
        return ordered[index]

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP rag_http_requests_total Total HTTP requests.",
            "# TYPE rag_http_requests_total counter",
        ]
        for (method, path, status), count in sorted(
            snapshot["request_counts"].items()
        ):
            lines.append(
                f'rag_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        lines.extend(
            [
                "# HELP rag_http_request_duration_ms HTTP request duration in milliseconds.",
                "# TYPE rag_http_request_duration_ms summary",
            ]
        )
        for (method, path), values in sorted(snapshot["durations"].items()):
            total = sum(values)
            p95 = self._p95(values)
            labels = f'method="{method}",path="{path}"'
            lines.append(
                f"rag_http_request_duration_ms_count{{{labels}}} {len(values)}"
            )
            lines.append(
                f"rag_http_request_duration_ms_sum{{{labels}}} {total:.3f}"
            )
            lines.append(
                f"rag_http_request_duration_ms_p95{{{labels}}} {p95:.3f}"
            )
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()


def record_request_metric(method: str, path: str, status_code: int, started_at: float) -> None:
    metrics_registry.record_request(
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=(monotonic() - started_at) * 1000,
    )

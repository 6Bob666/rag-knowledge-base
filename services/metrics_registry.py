"""轻量 Prometheus 兼容指标注册器。

生产环境可以替换为 prometheus-client；当前实现零外部依赖，
用于把请求量、错误率和延迟指标链路跑通。
"""

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from services.cost import estimate_cost_cny, TokenUsage


class MetricsRegistry:
    def __init__(self, max_samples_per_key: int = 1000):
        if max_samples_per_key <= 0:
            raise ValueError("max_samples_per_key 必须大于 0")
        self.max_samples_per_key = max_samples_per_key
        self._counts: defaultdict[tuple[str, str, int], int] = defaultdict(int)
        self._durations: defaultdict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples_per_key)
        )
        # Planner / Verifier 结论分布：用来回答"线上到底有多少请求走了重规划"。
        self._planner_verdicts: defaultdict[str, int] = defaultdict(int)
        self._planner_actions: defaultdict[str, int] = defaultdict(int)
        self._planner_replans: defaultdict[str, int] = defaultdict(int)
        self._planner_degraded = 0
        # 成本：token 累加在最外层，和请求量一样属于只增不减的计数器。
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_cny = 0.0
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
                "planner_verdicts": dict(self._planner_verdicts),
                "planner_actions": dict(self._planner_actions),
                "planner_replans": dict(self._planner_replans),
                "planner_degraded": self._planner_degraded,
                "llm_calls": self._llm_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cost_cny": self._cost_cny,
            }

    def record_planner_outcome(
        self,
        *,
        verdicts: list[str],
        actions: list[str],
        degraded: bool = False,
    ) -> None:
        """记录一次请求的规划轨迹：终局结论、动作序列、是否降级。"""
        if not verdicts:
            return
        with self._lock:
            self._planner_verdicts[verdicts[-1]] += 1
            self._planner_actions[actions[-1] if actions else "unknown"] += 1
            # 每轮检索都会产生一条规划动作，动作多于一条就说明重规划过。
            replan_count = len(actions) - 1
            if replan_count > 0:
                self._planner_replans[verdicts[0]] += 1
            if degraded:
                self._planner_degraded += 1

    def record_llm_usage(
        self, *, prompt_tokens: int, completion_tokens: int
    ) -> None:
        usage = TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )
        with self._lock:
            self._llm_calls += 1
            self._prompt_tokens += usage.prompt_tokens
            self._completion_tokens += usage.completion_tokens
            self._cost_cny += estimate_cost_cny(usage)

    def reset(self) -> None:
        """清空所有指标，测试和压测之间复用同一个进程时使用。"""
        with self._lock:
            self._counts.clear()
            self._durations.clear()
            self._planner_verdicts.clear()
            self._planner_actions.clear()
            self._planner_replans.clear()
            self._planner_degraded = 0
            self._llm_calls = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._cost_cny = 0.0

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

        lines.extend(
            [
                "# HELP rag_planner_verdict_total Requests by final verifier verdict.",
                "# TYPE rag_planner_verdict_total counter",
            ]
        )
        for verdict, count in sorted(snapshot["planner_verdicts"].items()):
            lines.append(
                f'rag_planner_verdict_total{{verdict="{verdict}"}} {count}'
            )

        lines.extend(
            [
                "# HELP rag_planner_action_total Requests by final planner action.",
                "# TYPE rag_planner_action_total counter",
            ]
        )
        for action, count in sorted(snapshot["planner_actions"].items()):
            lines.append(
                f'rag_planner_action_total{{action="{action}"}} {count}'
            )

        lines.extend(
            [
                "# HELP rag_planner_replan_total Requests that ran a second retrieval.",
                "# TYPE rag_planner_replan_total counter",
            ]
        )
        for verdict, count in sorted(snapshot["planner_replans"].items()):
            lines.append(
                f'rag_planner_replan_total{{first_verdict="{verdict}"}} {count}'
            )

        lines.extend(
            [
                "# HELP rag_planner_degraded_total Requests answered with best-effort context.",
                "# TYPE rag_planner_degraded_total counter",
                f"rag_planner_degraded_total {snapshot['planner_degraded']}",
                "# HELP rag_llm_tokens_total LLM tokens consumed.",
                "# TYPE rag_llm_tokens_total counter",
                f'rag_llm_tokens_total{{type="prompt"}} {snapshot["prompt_tokens"]}',
                f'rag_llm_tokens_total{{type="completion"}} {snapshot["completion_tokens"]}',
                "# HELP rag_llm_calls_total LLM API calls.",
                "# TYPE rag_llm_calls_total counter",
                f"rag_llm_calls_total {snapshot['llm_calls']}",
                "# HELP rag_llm_estimated_cost_cny_total Estimated LLM cost in CNY.",
                "# TYPE rag_llm_estimated_cost_cny_total counter",
                f"rag_llm_estimated_cost_cny_total {snapshot['cost_cny']:.6f}",
            ]
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

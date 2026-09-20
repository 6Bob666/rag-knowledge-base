"""评测阶段耗时的纯函数汇总，便于和检索指标一起输出。"""

from statistics import mean

from services.metrics import percentile


def summarize_timing_values(values: list[float]) -> dict:
    """返回一组耗时的平均值、P50/P95/P99 和最大值。"""
    if not values:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "avg_ms": round(mean(values), 1),
        "p50_ms": round(percentile(values, 0.50), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "p99_ms": round(percentile(values, 0.99), 1),
        "max_ms": round(max(values), 1),
    }


def summarize_timings(
    records: list[dict],
    keys: tuple[str, ...] = ("retrieval_ms", "rerank_ms", "total_ms"),
) -> dict:
    """按阶段名汇总多条评测记录的耗时。"""
    return {
        key: summarize_timing_values(
            [record[key] for record in records if key in record]
        )
        for key in keys
    }

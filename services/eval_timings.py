"""评测阶段耗时的纯函数汇总，便于和检索指标一起输出。"""

from statistics import mean


def summarize_timing_values(values: list[float]) -> dict:
    """返回一组耗时的 count / avg_ms / max_ms。"""
    if not values:
        return {"count": 0, "avg_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "avg_ms": round(mean(values), 1),
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

"""从日志文件中聚合 RAG 请求的各阶段耗时。

日志文件位置默认为 logs/app.log，也可以通过命令行参数指定：

    python analyze_rag_logs.py path/to/app.log
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


LINE_PATTERN = re.compile(
    r"rag_(?:answer|stream) "
    r"request_id=(?P<request_id>[^ |]+) \| "
    r"conversation_id=(?P<conversation_id>[^ |]+) \| "
    r"status=(?P<status>[^ |]+) \| "
    r"candidate_count=(?P<candidate_count>\d+) \| "
    r"context_count=(?P<context_count>\d+) \| "
    r"(?P<fields>.*)"
)

MS_FIELD_PATTERN = re.compile(r"(?P<name>[a-z_]+_ms)=(?P<value>[0-9.]+)")

TIMING_FIELDS = (
    "rewrite_ms",
    "retrieval_ms",
    "rerank_ms",
    "llm_ms",
    "total_ms",
)

# Agentic RAG 决策类字段：旧版普通 RAG 日志没有它们。
AGENT_FIELD_PATTERN = re.compile(
    r"(route_action|decision|attempt|max_attempts|search_count)=(?P<value>[^ |]+)"
)
AGENT_INT_FIELDS = ("attempt", "max_attempts", "search_count")


def parse_rag_log_line(line: str) -> dict | None:
    """解析一条 RAG 请求日志，非 RAG 日志返回 None。"""
    match = LINE_PATTERN.search(line)
    if not match:
        return None

    data = match.groupdict()
    result = {
        "request_id": data["request_id"],
        "conversation_id": data["conversation_id"],
        "status": data["status"],
        "candidate_count": int(data["candidate_count"]),
        "context_count": int(data["context_count"]),
    }
    for field in MS_FIELD_PATTERN.finditer(data["fields"]):
        result[field.group("name")] = float(field.group("value"))
    for name, value in AGENT_FIELD_PATTERN.findall(data["fields"]):
        result[name] = int(value) if name in AGENT_INT_FIELDS else value
    return result


def aggregate_rag_logs(lines) -> dict:
    """按 status 聚合各阶段耗时，返回 count、avg、max。"""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for line in lines:
        parsed = parse_rag_log_line(line)
        if parsed is None:
            continue
        status = parsed["status"]
        for field in TIMING_FIELDS:
            if field in parsed:
                grouped[status][field].append(parsed[field])

    summary = {}
    for status, fields in grouped.items():
        record = {"count": len(fields["total_ms"])}
        for field in TIMING_FIELDS:
            values = fields.get(field, [])
            if values:
                record[f"avg_{field}"] = round(sum(values) / len(values), 1)
                record[f"max_{field}"] = round(max(values), 1)
            else:
                record[f"avg_{field}"] = 0.0
                record[f"max_{field}"] = 0.0
        summary[status] = record

    return summary


def format_summary(summary: dict) -> str:
    """把聚合结果输出成适合阅读的文本。"""
    lines = ["status | count | avg_rewrite_ms | avg_retrieval_ms | avg_rerank_ms | avg_llm_ms | avg_total_ms | max_total_ms"]
    for status in sorted(summary):
        record = summary[status]
        lines.append(
            " | ".join(
                [
                    status,
                    str(record["count"]),
                    f"{record['avg_rewrite_ms']:.1f}",
                    f"{record['avg_retrieval_ms']:.1f}",
                    f"{record['avg_rerank_ms']:.1f}",
                    f"{record['avg_llm_ms']:.1f}",
                    f"{record['avg_total_ms']:.1f}",
                    f"{record['max_total_ms']:.1f}",
                ]
            )
        )
    return "\n".join(lines)


def aggregate_agent_metrics(lines) -> dict:
    """统计 Agentic RAG 日志：路由决策分布、每次请求的检索轮次。"""
    total_requests = 0
    route_counts: dict[str, int] = defaultdict(int)
    status_records: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"attempts": [], "search_counts": []}
    )

    for line in lines:
        parsed = parse_rag_log_line(line)
        if parsed is None or "route_action" not in parsed:
            continue
        total_requests += 1
        route_counts[parsed["route_action"]] += 1
        record = status_records[parsed["status"]]
        record["attempts"].append(parsed.get("attempt", 0))
        record["search_counts"].append(parsed.get("search_count", 0))

    status_summary = {}
    for status, record in sorted(status_records.items()):
        count = len(record["attempts"])
        status_summary[status] = {
            "count": count,
            "avg_attempt": round(sum(record["attempts"]) / count, 2)
            if count
            else 0.0,
            "avg_search_count": round(sum(record["search_counts"]) / count, 2)
            if count
            else 0.0,
        }

    return {
        "total_requests": total_requests,
        "route_actions": dict(route_counts),
        "status": status_summary,
    }


def format_agent_metrics(metrics: dict) -> str:
    """把 Agent 指标输出成可读文本。"""
    lines = [f"total_requests={metrics['total_requests']}"]
    lines.append("route_action | count")
    for action, count in sorted(metrics["route_actions"].items()):
        lines.append(f"{action} | {count}")
    if not metrics["status"]:
        return "\n".join(lines)
    lines.append("")
    lines.append("status | count | avg_attempt | avg_search_count")
    for status, record in metrics["status"].items():
        lines.append(
            f"{status} | {record['count']} | {record['avg_attempt']} | "
            f"{record['avg_search_count']}"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = Path(__file__).resolve().parent / "logs" / "app.log"

    if not log_path.exists():
        print(f"未找到日志文件: {log_path}")
        return

    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary = aggregate_rag_logs(lines)
    if not summary:
        print("没有解析到 RAG 请求日志。")
        return
    print(format_summary(summary))
    agent_metrics = aggregate_agent_metrics(lines)
    if agent_metrics["total_requests"]:
        print()
        print(format_agent_metrics(agent_metrics))


if __name__ == "__main__":
    main()

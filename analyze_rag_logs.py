"""从日志文件中聚合 RAG 请求的耗时、规划轨迹和成本。

日志文件位置默认为 logs/app.log，也可以通过命令行参数指定：

    python analyze_rag_logs.py path/to/app.log
    python analyze_rag_logs.py path/to/app.log --json dashboard.json

输出三块内容，正好对应线上最需要盯的三件事：

    耗时   各阶段 avg / P50 / P95 / max，按 status 分组
    规划   Verifier 结论分布、重规划触发率、降级生成率
    成本   每次请求的 token 与估算费用
"""

import argparse
import json
import re
from collections import Counter, defaultdict
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

# 日志里的字段统一是 key=value，用一条通用规则解析，新增字段不用改正则。
FIELD_PATTERN = re.compile(r"(?P<name>[a-z_][a-z0-9_]*)=(?P<value>[^ |]+)")

PERCENTILES = (("p50", 0.50), ("p95", 0.95))

TIMING_FIELDS = (
    "rewrite_ms",
    "retrieval_ms",
    "rerank_ms",
    "llm_ms",
    "total_ms",
)


def _coerce(value: str):
    """把日志字符串还原成数字/布尔；无法识别就保持字符串。"""
    if value == "True":
        return True
    if value == "False":
        return False
    # verify_coverage=None 表示这一轮没做覆盖度检查，不能当成字符串参与计算。
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


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
    for field in FIELD_PATTERN.finditer(data["fields"]):
        result[field.group("name")] = _coerce(field.group("value"))
    return result


def percentile(values: list[float], ratio: float) -> float:
    """最近秩百分位；样本少时不会插值出一个没人经历过的耗时。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(len(ordered) * ratio) - 1))
    return ordered[index]


def aggregate_rag_logs(lines) -> dict:
    """按 status 聚合各阶段耗时，返回 count、avg、P50、P95、max。"""
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
                for name, ratio in PERCENTILES:
                    record[f"{name}_{field}"] = round(
                        percentile(values, ratio), 1
                    )
                record[f"max_{field}"] = round(max(values), 1)
            else:
                record[f"avg_{field}"] = 0.0
                for name, _ in PERCENTILES:
                    record[f"{name}_{field}"] = 0.0
                record[f"max_{field}"] = 0.0
        summary[status] = record

    return summary


def format_summary(summary: dict) -> str:
    """把聚合结果输出成适合阅读的文本。"""
    lines = [
        "status | count | avg_total_ms | p50_total_ms | p95_total_ms | "
        "max_total_ms | avg_retrieval_ms | avg_rerank_ms | avg_llm_ms"
    ]
    for status in sorted(summary):
        record = summary[status]
        lines.append(
            " | ".join(
                [
                    status,
                    str(record["count"]),
                    f"{record['avg_total_ms']:.1f}",
                    f"{record['p50_total_ms']:.1f}",
                    f"{record['p95_total_ms']:.1f}",
                    f"{record['max_total_ms']:.1f}",
                    f"{record['avg_retrieval_ms']:.1f}",
                    f"{record['avg_rerank_ms']:.1f}",
                    f"{record['avg_llm_ms']:.1f}",
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


def aggregate_planner_metrics(lines) -> dict:
    """统计 Verifier 结论、Planner 动作、重规划与降级情况。

    这些字段只有接入了 Planner / Verifier 的日志才有，旧日志会被跳过。
    """
    total = 0
    verdicts: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    first_verdicts: Counter[str] = Counter()
    trails: Counter[str] = Counter()
    replans = 0
    degraded = 0

    for line in lines:
        parsed = parse_rag_log_line(line)
        if parsed is None or "verify_verdict" not in parsed:
            continue
        total += 1
        verdicts[parsed["verify_verdict"]] += 1
        actions[parsed.get("plan_action", "unknown")] += 1
        if parsed.get("degraded"):
            degraded += 1

        trail = parsed.get("verify_trail")
        if isinstance(trail, str):
            trails[trail] += 1
            first_verdicts[trail.split(">")[0]] += 1
        # attempt 是"这一轮请求真正检索了几次"，次数大于 1 就说明发生过重规划。
        if int(parsed.get("attempt", 1) or 1) > 1:
            replans += 1

    def rate(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    return {
        "total_requests": total,
        "verdicts": dict(verdicts),
        "actions": dict(actions),
        "replan_count": replans,
        "replan_rate": rate(replans),
        "degraded_count": degraded,
        "degraded_rate": rate(degraded),
        "retry_causes": dict(first_verdicts),
        "top_trails": trails.most_common(5),
    }


def format_planner_metrics(metrics: dict) -> str:
    """把 Planner / Verifier 指标输出成可读文本。"""
    if not metrics["total_requests"]:
        return ""
    lines = [f"planner_requests={metrics['total_requests']}"]
    lines.append("verdict | count")
    for verdict, count in sorted(metrics["verdicts"].items()):
        lines.append(f"{verdict} | {count}")
    lines.append("")
    lines.append(
        f"replan_rate={metrics['replan_rate']} ({metrics['replan_count']}) | "
        f"degraded_rate={metrics['degraded_rate']} "
        f"({metrics['degraded_count']})"
    )
    if metrics["retry_causes"]:
        lines.append(
            "重规划触发原因: "
            + ", ".join(
                f"{cause}={count}"
                for cause, count in sorted(metrics["retry_causes"].items())
            )
        )
    return "\n".join(lines)


def aggregate_cost_metrics(lines) -> dict:
    """统计 token 与估算成本；没有用量字段的日志会被跳过。"""
    requests = 0
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    per_status: dict[str, float] = defaultdict(float)
    status_counts: dict[str, int] = defaultdict(int)

    for line in lines:
        parsed = parse_rag_log_line(line)
        if parsed is None or "cost_cny" not in parsed:
            continue
        requests += 1
        llm_calls += int(parsed.get("llm_calls", 0) or 0)
        prompt_tokens += int(parsed.get("prompt_tokens", 0) or 0)
        completion_tokens += int(parsed.get("completion_tokens", 0) or 0)
        cost += float(parsed.get("cost_cny", 0.0) or 0.0)
        status = parsed["status"]
        per_status[status] += float(parsed.get("cost_cny", 0.0) or 0.0)
        status_counts[status] += 1

    return {
        "requests_with_usage": requests,
        "llm_calls": llm_calls,
        "avg_llm_calls": round(llm_calls / requests, 2) if requests else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_cny": round(cost, 6),
        "avg_cost_cny": round(cost / requests, 6) if requests else 0.0,
        "cost_by_status": {
            status: {
                "count": status_counts[status],
                "cost_cny": round(value, 6),
                "avg_cost_cny": round(value / status_counts[status], 6),
            }
            for status, value in sorted(per_status.items())
        },
    }


def format_cost_metrics(metrics: dict) -> str:
    """把成本指标输出成可读文本。"""
    if not metrics["requests_with_usage"]:
        return ""
    lines = [
        f"统计请求数={metrics['requests_with_usage']} | "
        f"模型调用次数={metrics['llm_calls']} | "
        f"平均调用次数={metrics['avg_llm_calls']}",
        f"prompt_tokens={metrics['prompt_tokens']} | "
        f"completion_tokens={metrics['completion_tokens']}",
        f"总估算成本={metrics['total_cost_cny']:.6f} 元 | "
        f"单次请求平均={metrics['avg_cost_cny']:.6f} 元",
    ]
    if metrics["cost_by_status"]:
        lines.append("")
        lines.append("status | count | cost_cny | avg_cost_cny")
        for status, record in metrics["cost_by_status"].items():
            lines.append(
                f"{status} | {record['count']} | {record['cost_cny']:.6f} | "
                f"{record['avg_cost_cny']:.6f}"
            )
    return "\n".join(lines)


def build_dashboard(lines) -> dict:
    """汇总成一份可直接喂给看板的 JSON 报告。"""
    return {
        "timing_by_status": aggregate_rag_logs(lines),
        "agent": aggregate_agent_metrics(lines),
        "planner": aggregate_planner_metrics(lines),
        "cost": aggregate_cost_metrics(lines),
    }


def main():
    parser = argparse.ArgumentParser(description="聚合 RAG 日志的耗时、规划与成本")
    parser.add_argument(
        "log_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "logs" / "app.log"),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default="",
        help="额外输出一份 JSON 看板报告",
    )
    args = parser.parse_args()
    log_path = Path(args.log_path)

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
    planner_metrics = aggregate_planner_metrics(lines)
    if planner_metrics["total_requests"]:
        print()
        print(format_planner_metrics(planner_metrics))
    cost_metrics = aggregate_cost_metrics(lines)
    if cost_metrics["requests_with_usage"]:
        print()
        print(format_cost_metrics(cost_metrics))
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(build_dashboard(lines), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n看板报告已保存到 {args.json_path}")


if __name__ == "__main__":
    main()

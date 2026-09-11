"""日志聚合看板的解析与统计测试（纯文本输入，不加载模型）。"""

from analyze_rag_logs import (
    aggregate_cost_metrics,
    aggregate_planner_metrics,
    build_dashboard,
    format_cost_metrics,
    format_planner_metrics,
    parse_rag_log_line,
    percentile,
)


def planner_line(
    rid,
    status="success",
    *,
    verdict="sufficient",
    action="generate",
    trail=None,
    plan_trail=None,
    degraded=None,
    cost=0.001,
    prompt=100,
    completion=20,
    llm_calls=2,
):
    fields = [
        f"route_action=retrieve",
        f"decision={status}",
        f"attempt=2" if trail else "attempt=1",
        "max_attempts=2",
        "search_count=2" if trail else "search_count=1",
        f"verify_verdict={verdict}",
        "verify_coverage=0.9",
        f"plan_action={action}",
        f"llm_calls={llm_calls}",
        f"prompt_tokens={prompt}",
        f"completion_tokens={completion}",
        f"cost_cny={cost}",
        "retrieval_ms=10.0",
        "total_ms=100.0",
    ]
    if trail:
        fields.append(f"verify_trail={trail}")
    if plan_trail:
        fields.append(f"plan_trail={plan_trail}")
    if degraded:
        fields.append("degraded=True")
    joined = " | ".join(fields)
    return (
        f"INFO | rag | rag_answer request_id={rid} | conversation_id=- | "
        f"status={status} | candidate_count=3 | context_count=2 | {joined}"
    )


def test_parse_reads_planner_and_cost_fields():
    parsed = parse_rag_log_line(
        planner_line(
            "a",
            verdict="low_coverage",
            trail="low_coverage>sufficient",
            plan_trail="retrieve>generate",
        )
    )

    assert parsed["verify_verdict"] == "low_coverage"
    assert parsed["verify_coverage"] == 0.9
    assert parsed["verify_trail"] == "low_coverage>sufficient"
    assert parsed["plan_trail"] == "retrieve>generate"
    assert parsed["llm_calls"] == 2
    assert parsed["prompt_tokens"] == 100
    assert parsed["cost_cny"] == 0.001


def test_parse_treats_missing_coverage_as_none():
    line = planner_line("a").replace("verify_coverage=0.9", "verify_coverage=None")

    assert parse_rag_log_line(line)["verify_coverage"] is None


def test_parse_marks_degraded_as_boolean():
    parsed = parse_rag_log_line(planner_line("a", degraded=True))

    assert parsed["degraded"] is True


def test_pencentile_uses_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.95) == 100.0
    assert percentile([], 0.95) == 0.0


def test_aggregate_planner_metrics_counts_replans_and_degraded():
    lines = [
        planner_line("a"),
        planner_line(
            "b",
            trail="low_coverage>sufficient",
            plan_trail="retrieve>generate",
        ),
        planner_line(
            "c",
            status="no_context",
            verdict="empty",
            action="refuse",
        ),
        planner_line(
            "d",
            verdict="low_coverage",
            action="generate",
            degraded=True,
        ),
    ]

    metrics = aggregate_planner_metrics(lines)

    assert metrics["total_requests"] == 4
    assert metrics["verdicts"]["sufficient"] == 2
    assert metrics["verdicts"]["empty"] == 1
    assert metrics["replan_count"] == 1
    assert metrics["replan_rate"] == 0.25
    assert metrics["degraded_count"] == 1
    assert metrics["degraded_rate"] == 0.25
    # 重规划是由哪一类失败触发的
    assert metrics["retry_causes"] == {"low_coverage": 1}


def test_aggregate_planner_metrics_ignores_logs_without_verdict():
    plain = (
        "INFO | rag | rag_answer request_id=x | conversation_id=- | "
        "status=success | candidate_count=1 | context_count=1 | total_ms=10.0"
    )

    metrics = aggregate_planner_metrics([plain])

    assert metrics["total_requests"] == 0
    assert format_planner_metrics(metrics) == ""


def test_aggregate_cost_metrics_sums_tokens_and_cost():
    lines = [
        planner_line("a", cost=0.001, prompt=100, completion=20, llm_calls=2),
        planner_line("b", cost=0.003, prompt=200, completion=40, llm_calls=3),
    ]

    metrics = aggregate_cost_metrics(lines)

    assert metrics["requests_with_usage"] == 2
    assert metrics["llm_calls"] == 5
    assert metrics["avg_llm_calls"] == 2.5
    assert metrics["prompt_tokens"] == 300
    assert metrics["total_cost_cny"] == 0.004
    assert metrics["avg_cost_cny"] == 0.002
    assert metrics["cost_by_status"]["success"]["count"] == 2


def test_format_cost_metrics_reports_per_request_average():
    metrics = aggregate_cost_metrics([planner_line("a", cost=0.002)])

    text = format_cost_metrics(metrics)

    assert "单次请求平均=0.002000 元" in text


def test_build_dashboard_contains_all_four_blocks():
    lines = [
        planner_line("a"),
        planner_line("b", trail="empty>sufficient", plan_trail="retrieve>generate"),
    ]

    dashboard = build_dashboard(lines)

    assert set(dashboard) == {"timing_by_status", "agent", "planner", "cost"}
    assert dashboard["planner"]["total_requests"] == 2
    assert dashboard["cost"]["requests_with_usage"] == 2
    assert dashboard["agent"]["route_actions"] == {"retrieve": 2}

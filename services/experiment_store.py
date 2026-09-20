"""实验报告的保存、加载和对比。"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiment_results"


def make_experiment_id(name: str | None = None) -> str:
    """根据名称生成稳定的实验 ID；没有名称时使用时间戳。"""
    if name:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
        if cleaned:
            return cleaned
    return (
        "experiment-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )


def save_experiment(
    report: dict,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    experiment_id: str | None = None,
) -> Path:
    """保存实验报告，返回保存后的文件路径。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    experiment_id = experiment_id or make_experiment_id(report.get("experiment_id"))
    report = dict(report)
    report["experiment_id"] = experiment_id

    target = output_path / f"{experiment_id}.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_experiments(output_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    """加载目录下的全部实验报告，按文件名排序。"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    reports = []
    for file in sorted(output_path.glob("*.json")):
        try:
            report = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # 详情文件通常是 list[dict]，不能参与实验汇总对比。
        if not isinstance(report, dict):
            continue
        if not isinstance(report.get("summary"), dict):
            continue
        report.setdefault("experiment_id", file.stem)
        reports.append(report)
    return reports


def get_strategy_summary(report: dict, strategy: str = "reranked") -> dict:
    """从报告里取某个策略的聚合指标；缺省时返回空字典。"""
    summary = report.get("summary") or {}
    return summary.get(strategy) or {}


def format_comparison(
    reports: list[dict],
    strategy: str = "reranked",
) -> str:
    """把多份实验报告格式化成可读的对比表。"""
    header = (
        "experiment_id | top_k | threshold | recall_at_k | precision_at_k | "
        "mrr | rejection_rate | avg_total_ms | p95_total_ms | max_total_ms"
    )
    lines = [header]
    for report in reports:
        metadata = report.get("metadata") or {}
        summary = get_strategy_summary(report, strategy)
        if not summary:
            continue
        values = [
            report.get("experiment_id", "-"),
            str(metadata.get("top_k", "-")),
            str(metadata.get("reranker_score_threshold", "-")),
            str(summary.get("recall_at_k", "-")),
            str(summary.get("precision_at_k", "-")),
            str(summary.get("mrr", "-")),
            str(summary.get("rejection_rate", "-")),
            str(summary.get("avg_total_ms", "-")),
            str(
                summary.get(
                    "p95_total_ms",
                    (report.get("timing_summary") or {})
                    .get("total_ms", {})
                    .get("p95_ms", "-"),
                )
            ),
            str(summary.get("max_total_ms", "-")),
        ]
        lines.append(" | ".join(values))
    return "\n".join(lines)

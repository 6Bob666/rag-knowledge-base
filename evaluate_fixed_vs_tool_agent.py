"""对比固定 RAG 接口与 Function Calling Tool Agent。

两种方案使用同一份评测集、同一个知识库和同一个 top_k：

    python evaluate_fixed_vs_tool_agent.py --top-k 3

脚本通过 HTTP 调用正在运行的 FastAPI 服务，适合做系统级对照实验。
报告会保存到 experiment_results/fixed-vs-tool-agent.json。
"""

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import requests

from evaluate_retrieval import load_dataset
from services.experiment_store import save_experiment
from services.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


def _chunk_ids(payload: dict, top_k: int) -> list[str]:
    """从 AnswerResponse 中提取最终返回的 chunk_id。"""
    ids = []
    for context in payload.get("contexts") or []:
        chunk_id = context.get("chunk_id")
        if chunk_id is not None:
            ids.append(str(chunk_id))
    return ids[:top_k]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 1)


def _is_negative(case: dict) -> bool:
    return not case.get("relevant_chunk_ids")


def _case_metrics(returned_ids: list[str], case: dict, top_k: int) -> dict:
    relevant_ids = [str(item) for item in case.get("relevant_chunk_ids", [])]
    if not relevant_ids:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "rr": 0.0,
            "returned_chunk_ids": returned_ids,
            "returned_count": len(returned_ids),
            "is_negative": True,
        }
    return {
        "recall_at_k": calculate_recall_at_k(returned_ids, relevant_ids, top_k),
        "precision_at_k": calculate_precision_at_k(
            returned_ids, relevant_ids, top_k
        ),
        "rr": calculate_reciprocal_rank(returned_ids, relevant_ids, top_k),
        "returned_chunk_ids": returned_ids,
        "returned_count": len(returned_ids),
        "is_negative": False,
    }


def _call_endpoint(
    session: requests.Session,
    url: str,
    question: str,
    top_k: int,
    timeout: float,
) -> dict:
    start = time.perf_counter()
    try:
        response = session.post(
            url,
            json={"question": question, "top_k": top_k},
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return {
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "payload": payload if isinstance(payload, dict) else {},
            "error": None if response.ok else payload.get("detail", response.text),
            "headers": {
                "tool_calls": int(response.headers.get("X-Agent-Tool-Calls", "0")),
                "steps": int(response.headers.get("X-Agent-Steps", "0")),
                "termination": response.headers.get("X-Agent-Termination", ""),
            },
        }
    except requests.RequestException as exc:
        return {
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "payload": {},
            "error": str(exc),
            "headers": {"tool_calls": 0, "steps": 0, "termination": "error"},
        }


def run_strategy(
    *,
    strategy: str,
    url: str,
    cases: list[dict],
    top_k: int,
    timeout: float,
) -> dict:
    details = []
    positive = []
    negative = []
    elapsed_values = []
    session = requests.Session()

    for index, case in enumerate(cases, start=1):
        result = _call_endpoint(session, url, case["question"], top_k, timeout)
        returned_ids = _chunk_ids(result["payload"], top_k)
        metrics = _case_metrics(returned_ids, case, top_k)
        item = {
            "index": index,
            "question": case["question"],
            "status_code": result["status_code"],
            "elapsed_ms": result["elapsed_ms"],
            "answer_present": bool((result["payload"].get("answer") or "").strip()),
            "error": result["error"],
            "metrics": metrics,
        }
        if strategy == "tool_agent":
            item.update(result["headers"])
        details.append(item)
        elapsed_values.append(result["elapsed_ms"])
        (negative if _is_negative(case) else positive).append(metrics)
        print(
            f"[{strategy}] {index:02d}/{len(cases)} "
            f"status={result['status_code']} time={result['elapsed_ms']}ms "
            f"contexts={len(returned_ids)}"
        )

    def avg(items: list[dict], key: str) -> float:
        return round(mean(item[key] for item in items), 3) if items else 0.0

    def empty_rate(items: list[dict]) -> float:
        return round(
            sum(item["returned_count"] == 0 for item in items) / len(items),
            3,
        ) if items else 0.0

    summary = {
        "total_cases": len(cases),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_recall_at_k": avg(positive, "recall_at_k"),
        "positive_precision_at_k": avg(positive, "precision_at_k"),
        "positive_mrr": avg(positive, "rr"),
        "negative_empty_context_rate": empty_rate(negative),
        "http_success_rate": round(
            sum(item["status_code"] == 200 for item in details) / len(details),
            3,
        ) if details else 0.0,
        "avg_total_ms": round(mean(elapsed_values), 1) if elapsed_values else 0.0,
        "p95_total_ms": _percentile(elapsed_values, 0.95),
        "error_count": sum(item["error"] is not None for item in details),
    }
    if strategy == "tool_agent":
        summary.update(
            {
                "avg_tool_calls": round(
                    mean(item.get("tool_calls", 0) for item in details), 3
                ),
                "avg_steps": round(
                    mean(item.get("steps", 0) for item in details), 3
                ),
                "termination_counts": {
                    value: sum(item.get("termination") == value for item in details)
                    for value in sorted(
                        {item.get("termination", "") for item in details}
                    )
                },
            }
        )
    return {"summary": summary, "details": details}


def build_metadata(dataset_path: str, top_k: int, timeout: float, base_url: str) -> dict:
    dataset_file = Path(dataset_path).resolve()
    digest = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    return {
        "dataset": str(dataset_file),
        "dataset_sha256": digest,
        "knowledge_base": "local Chroma + SQLite",
        "top_k": top_k,
        "timeout_seconds": timeout,
        "base_url": base_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "同一评测集下，对比固定 RAG 与 Function Calling Tool Agent；指标来自接口最终返回 contexts。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_dataset.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="experiment_results")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    base_url = args.base_url.rstrip("/")
    metadata = build_metadata(args.dataset, args.top_k, args.timeout, base_url)
    report = {
        "experiment_id": "fixed-vs-tool-agent",
        "metadata": metadata,
        "strategies": {
            "fixed_rag": run_strategy(
                strategy="fixed_rag",
                url=f"{base_url}/chat/ask",
                cases=cases,
                top_k=args.top_k,
                timeout=args.timeout,
            ),
            "tool_agent": run_strategy(
                strategy="tool_agent",
                url=f"{base_url}/chat/agent/tool",
                cases=cases,
                top_k=args.top_k,
                timeout=args.timeout,
            ),
        },
    }
    target = save_experiment(
        report,
        output_dir=args.output_dir,
        experiment_id="fixed-vs-tool-agent",
    )
    print("\n===== 固定 RAG vs Tool Agent =====")
    for name, value in report["strategies"].items():
        summary = value["summary"]
        print(
            f"{name}: Recall@{args.top_k}={summary['positive_recall_at_k']:.3f}, "
            f"Precision@{args.top_k}={summary['positive_precision_at_k']:.3f}, "
            f"MRR={summary['positive_mrr']:.3f}, "
            f"负样本空上下文率={summary['negative_empty_context_rate']:.3f}, "
            f"平均耗时={summary['avg_total_ms']:.1f}ms, "
            f"P95={summary['p95_total_ms']:.1f}ms"
        )
        if name == "tool_agent":
            print(
                f"  平均工具调用={summary['avg_tool_calls']:.3f}, "
                f"平均步骤={summary['avg_steps']:.3f}"
            )
    print(f"报告已保存到: {target}")


if __name__ == "__main__":
    main()

"""使用人工标注 chunk ID，对比混合检索和 Reranker 的效果。

运行前请准备 evaluation_dataset.json，格式可参考
evaluation_dataset.example.json。
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from config import settings
from services.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from services.eval_timings import summarize_timings
from services.reranker import ReRanker
from services.vector_store import VectorStore


def load_dataset(path: str) -> list[dict]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"评测集不存在: {dataset_path}。"
            "请复制 evaluation_dataset.example.json 并补充人工标注的 chunk_id。"
        )

    with dataset_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须是非空 JSON 数组")
    for index, case in enumerate(cases, start=1):
        if not case.get("question"):
            raise ValueError(f"第 {index} 道题缺少 question")
        if "relevant_chunk_ids" not in case:
            raise ValueError(f"第 {index} 道题缺少 relevant_chunk_ids 字段")
    return cases


def evaluate_strategy(
    returned_records: list[dict],
    case: dict,
    top_k: int,
    timings: dict | None = None,
) -> dict:
    returned_ids = [record["id"] for record in returned_records]
    relevant_ids = case["relevant_chunk_ids"]
    if not relevant_ids:
        # 负样本：知识库中没有标准相关文档，应尽量返回空结果。
        result = {
            "recall_at_k": 0,
            "precision_at_k": 0,
            "rr": 0,
            "returned_chunk_ids": returned_ids[:top_k],
            "is_negative": True,
            "returned_count": len(returned_ids[:top_k]),
        }
    else:
        result = {
            "recall_at_k": calculate_recall_at_k(returned_ids, relevant_ids, top_k),
            "precision_at_k": calculate_precision_at_k(returned_ids, relevant_ids, top_k),
            "rr": calculate_reciprocal_rank(returned_ids, relevant_ids, top_k),
            "returned_chunk_ids": returned_ids[:top_k],
            "is_negative": False,
            "returned_count": len(returned_ids[:top_k]),
        }
    if timings is not None:
        result["timings"] = timings
    return result


def build_metadata(dataset_path: str, top_k: int, threshold: float) -> dict:
    """生成本次评测的配置快照，保证结果可以复现和追溯。"""
    return {
        "dataset": str(Path(dataset_path).resolve()),
        "top_k": top_k,
        "reranker_score_threshold": threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model_path": settings.embedding_model_path,
        "reranker_model_path": settings.reranker_model_path,
        "collection_name": "kb_docs",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(
    dataset_path: str = "evaluation_dataset.json",
    top_k: int = 3,
    threshold: float = 0.0,
):
    cases = load_dataset(dataset_path)
    store = VectorStore()
    reranker = ReRanker(score_threshold=threshold)
    baseline_results = []
    reranked_results = []
    baseline_negative = []
    reranked_negative = []
    reranked_timings = []
    details = []

    for case in cases:
        question = case["question"]
        start = time.perf_counter()
        baseline = store.hybrid_search_records(question, top_k=top_k)
        baseline_total_ms = round((time.perf_counter() - start) * 1000, 1)

        start = time.perf_counter()
        candidates = store.hybrid_search_records(question, top_k=top_k * 3)
        candidate_retrieval_ms = round((time.perf_counter() - start) * 1000, 1)

        start = time.perf_counter()
        reranked = reranker.rerank_records(question, candidates, top_k=top_k)
        rerank_ms = round((time.perf_counter() - start) * 1000, 1)
        timings = {
            "retrieval_ms": candidate_retrieval_ms,
            "rerank_ms": rerank_ms,
            "total_ms": round(candidate_retrieval_ms + rerank_ms, 1),
        }

        baseline_metrics = evaluate_strategy(
            baseline,
            case,
            top_k,
            {"search_ms": baseline_total_ms, "total_ms": baseline_total_ms},
        )
        reranked_metrics = evaluate_strategy(reranked, case, top_k, timings)
        reranked_timings.append(timings)
        if case["relevant_chunk_ids"]:
            baseline_results.append(baseline_metrics)
            reranked_results.append(reranked_metrics)
        else:
            baseline_negative.append(baseline_metrics)
            reranked_negative.append(reranked_metrics)
        details.append({
            "question": question,
            "baseline": baseline_metrics,
            "reranked": reranked_metrics,
        })

    def average(results: list[dict], key: str) -> float:
        return mean(item[key] for item in results)

    def rejection_rate(results: list[dict]) -> float:
        if not results:
            return 0.0
        rejected = sum(1 for item in results if item["returned_count"] == 0)
        return rejected / len(results)

    def strategy_summary(
        results: list[dict],
        negative_results: list[dict],
    ) -> dict:
        return {
            "count": len(results),
            "recall_at_k": (
                average(results, "recall_at_k") if results else 0.0
            ),
            "precision_at_k": (
                average(results, "precision_at_k") if results else 0.0
            ),
            "mrr": average(results, "rr") if results else 0.0,
            "rejection_rate": rejection_rate(negative_results),
        }

    print(f"K={top_k}, Reranker threshold={threshold}")
    print(f"正样本={len(baseline_results)} 道，负样本={len(baseline_negative)} 道")
    print("strategy                    Recall@K   Precision@K       MRR   拒答率")
    print(
        f"hybrid                      "
        f"{average(baseline_results, 'recall_at_k'):8.3f}   "
        f"{average(baseline_results, 'precision_at_k'):12.3f}   "
        f"{average(baseline_results, 'rr'):8.3f}   "
        f"{rejection_rate(baseline_negative):7.3f}"
    )
    print(
        f"hybrid + reranker           "
        f"{average(reranked_results, 'recall_at_k'):8.3f}   "
        f"{average(reranked_results, 'precision_at_k'):12.3f}   "
        f"{average(reranked_results, 'rr'):8.3f}   "
        f"{rejection_rate(reranked_negative):7.3f}"
    )

    timing_summary = summarize_timings(reranked_timings)
    print("耗时统计（hybrid + reranker 流程，单位 ms）：")
    print(
        "阶段         平均       最大\n"
        f"retrieval   {timing_summary['retrieval_ms']['avg_ms']:8.1f}   "
        f"{timing_summary['retrieval_ms']['max_ms']:8.1f}\n"
        f"rerank      {timing_summary['rerank_ms']['avg_ms']:8.1f}   "
        f"{timing_summary['rerank_ms']['max_ms']:8.1f}\n"
        f"total       {timing_summary['total_ms']['avg_ms']:8.1f}   "
        f"{timing_summary['total_ms']['max_ms']:8.1f}"
    )

    Path("retrieval_eval_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("详细结果已保存到 retrieval_eval_details.json")

    report = {
        "summary": {
            "baseline": strategy_summary(baseline_results, baseline_negative),
            "reranked": strategy_summary(
                reranked_results,
                reranked_negative,
            ),
        },
        "metadata": build_metadata(dataset_path, top_k, threshold),
        "timing_summary": timing_summary,
        "details": details,
    }
    report["summary"]["reranked"]["avg_total_ms"] = (
        timing_summary["total_ms"]["avg_ms"] if reranked_timings else 0.0
    )
    report["summary"]["reranked"]["max_total_ms"] = (
        timing_summary["total_ms"]["max_ms"] if reranked_timings else 0.0
    )
    Path("retrieval_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("评测报告（含配置快照）已保存到 retrieval_eval_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_dataset.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    run(args.dataset, args.top_k, args.threshold)

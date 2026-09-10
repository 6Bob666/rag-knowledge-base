"""困难/多轮检索评测：对比“单次检索”与“最多一次补救检索”。

运行示例：
    python evaluate_multi_turn.py \
        --dataset evaluation_dataset_hard.example.json \
        --top-k 3 --threshold 0.5
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from config import settings
from evaluate_retrieval import evaluate_strategy, load_dataset
from services.retrieval_retry_planner import plan_attempt_queries
from services.reranker import ReRanker
from services.vector_store import VectorStore


def retrieve_once(store, reranker, query: str, top_k: int) -> dict:
    """执行一轮混合检索 + Reranker，返回候选、上下文和耗时。"""
    start = time.perf_counter()
    candidates = store.hybrid_search_records(query, top_k=top_k * 3)
    contexts = reranker.rerank_records(query, candidates, top_k=top_k)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    return {
        "query": query,
        "candidates": candidates,
        "contexts": contexts,
        "elapsed_ms": elapsed_ms,
    }


def choose_final_attempt(attempts: list[dict]) -> dict:
    """优先返回第一份有上下文的结果；都没有则返回最后一次结果。"""
    for attempt in attempts:
        if attempt["contexts"]:
            return attempt
    return attempts[-1]


def evaluate_case(
    store,
    reranker,
    case: dict,
    top_k: int,
) -> dict:
    question = case["question"]
    rewritten_query = case.get("rewritten_query")
    queries = plan_attempt_queries(question, rewritten_query)

    # 不开启重试：只执行第一条查询
    first_attempt = retrieve_once(store, reranker, queries[0], top_k)

    attempts = [first_attempt]
    if len(queries) > 1 and not first_attempt["contexts"]:
        second_attempt = retrieve_once(store, reranker, queries[1], top_k)
        attempts.append(second_attempt)

    retry_attempt = choose_final_attempt(attempts)

    single_metrics = evaluate_strategy(
        first_attempt["contexts"],
        case,
        top_k,
    )
    retry_metrics = evaluate_strategy(
        retry_attempt["contexts"],
        case,
        top_k,
    )

    return {
        "id": case.get("id", ""),
        "question": question,
        "rewritten_query": rewritten_query,
        "queries_used": [attempt["query"] for attempt in attempts],
        "retry_triggered": len(attempts) > 1,
        "attempt_count": len(attempts),
        "single": {
            **single_metrics,
            "elapsed_ms": first_attempt["elapsed_ms"],
        },
        "retry": {
            **retry_metrics,
            "elapsed_ms": sum(
                attempt["elapsed_ms"] for attempt in attempts
            ),
        },
    }


def run(
    dataset_path: str = "evaluation_dataset_hard.example.json",
    top_k: int = 3,
    threshold: float = 0.5,
    limit: int | None = None,
):
    cases = load_dataset(dataset_path)
    if limit is not None:
        cases = cases[:limit]

    store = VectorStore()
    reranker = ReRanker(score_threshold=threshold)
    details = [
        evaluate_case(store, reranker, case, top_k)
        for case in cases
    ]

    def positive_records(strategy: str) -> list[dict]:
        return [
            item[strategy]
            for item in details
            if not item[strategy]["is_negative"]
        ]

    def negative_records(strategy: str) -> list[dict]:
        return [
            item[strategy]
            for item in details
            if item[strategy]["is_negative"]
        ]

    def average(strategy: str, key: str) -> float:
        records = positive_records(strategy)
        return mean(item[key] for item in records) if records else 0.0

    def rejection_rate(strategy: str) -> float:
        records = negative_records(strategy)
        if not records:
            return 0.0
        rejected = sum(1 for item in records if item["returned_count"] == 0)
        return rejected / len(records)

    print(f"数据集={dataset_path}，正样本={len(positive_records('single'))} 道，"
          f"负样本={len(negative_records('single'))} 道，top_k={top_k}，threshold={threshold}")
    print("strategy   Recall@K   Precision@K       MRR   拒答率  avg_ms")
    for strategy in ("single", "retry"):
        print(
            f"{strategy:10}"
            f"{average(strategy, 'recall_at_k'):10.3f}   "
            f"{average(strategy, 'precision_at_k'):12.3f}   "
            f"{average(strategy, 'rr'):8.3f}   "
            f"{rejection_rate(strategy):7.3f}   "
            f"{mean(item[strategy]['elapsed_ms'] for item in details):7.1f}"
        )

    trigger_count = sum(1 for item in details if item["retry_triggered"])
    print(f"触发重试的题数={trigger_count}/{len(details)}")

    report = {
        "metadata": {
            "dataset": str(Path(dataset_path).resolve()),
            "top_k": top_k,
            "reranker_score_threshold": threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "settings_chunk_size": settings.chunk_size,
        },
        "summary": {
            "single": {
                "recall_at_k": average("single", "recall_at_k"),
                "precision_at_k": average("single", "precision_at_k"),
                "mrr": average("single", "rr"),
                "rejection_rate": rejection_rate("single"),
            },
            "retry": {
                "recall_at_k": average("retry", "recall_at_k"),
                "precision_at_k": average("retry", "precision_at_k"),
                "mrr": average("retry", "rr"),
                "rejection_rate": rejection_rate("retry"),
            },
        },
        "retry_triggered_rate": (
            trigger_count / len(details) if details else 0.0
        ),
        "details": details,
    }
    Path("retrieval_retry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("报告已保存到 retrieval_retry_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_dataset_hard.example.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.dataset, args.top_k, args.threshold, args.limit)

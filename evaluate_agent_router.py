"""Agent Router 对照实验：rules vs LLM。

同一个评测集分别跑两种 Router 模式，并用混合检索 + Reranker
衡量决策后的检索效果。相比 evaluate_retrieval.py，这里多统计了：

    - Router 动作分布（retrieve / direct_answer / refuse）
    - 知识库外问题是否被错误分流到直接回答（幻觉风险）
    - 平均检索次数与路由耗时

用法：
    python evaluate_agent_router.py --top-k 3 --threshold 0.5
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from evaluate_retrieval import evaluate_strategy, load_dataset
from services.agent_router import route_question, route_question_with_llm
from services.experiment_store import save_experiment
from services.reranker import ReRanker
from services.vector_store import VectorStore


def run_mode(
    *,
    mode: str,
    cases: list[dict],
    store: VectorStore,
    reranker: ReRanker,
    top_k: int,
) -> dict:
    """跑一种 Router 模式，返回报告与明细。"""
    details = []
    positive = []
    negative = []
    timings = []
    action_counts = {"retrieve": 0, "direct_answer": 0, "refuse": 0}

    for case in cases:
        question = case["question"]
        route_start = time.perf_counter()
        if mode == "llm":
            decision = route_question_with_llm(question, history=[])
        else:
            decision = route_question(question, history=[])
        route_ms = round((time.perf_counter() - route_start) * 1000, 1)
        action_counts[decision.action] += 1

        is_negative = not case["relevant_chunk_ids"]
        if decision.action != "retrieve":
            # 不检索：正样本视为未召回；负样本视为“没有走检索”而非安全拒答。
            metrics = {
                "recall_at_k": 0,
                "precision_at_k": 0,
                "rr": 0,
                "returned_chunk_ids": [],
                "is_negative": is_negative,
                "returned_count": 0,
                "skipped_by_router": True,
            }
            item_timings = {
                "route_ms": route_ms,
                "retrieval_ms": 0.0,
                "rerank_ms": 0.0,
                "total_ms": route_ms,
            }
        else:
            query = decision.query or question
            retrieval_start = time.perf_counter()
            candidates = store.hybrid_search_records(query, top_k=top_k * 3)
            retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 1)

            rerank_start = time.perf_counter()
            contexts = reranker.rerank_records(query, candidates, top_k=top_k)
            rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 1)
            metrics = evaluate_strategy(contexts, case, top_k)
            item_timings = {
                "route_ms": route_ms,
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "total_ms": round(route_ms + retrieval_ms + rerank_ms, 1),
            }

        timings.append(item_timings)
        target = negative if is_negative else positive
        target.append(metrics)
        details.append(
            {
                "question": question,
                "router_action": decision.action,
                "router_reason": decision.reason,
                "router_confidence": decision.confidence,
                "query_used": decision.query or question,
                "expected_retrieve": True,
                "metrics": metrics,
                "timings": item_timings,
            }
        )

    def average(items, key):
        return mean(item[key] for item in items) if items else 0.0

    def ratio(items, predicate):
        if not items:
            return 0.0
        return sum(1 for item in items if predicate(item)) / len(items)

    out_of_kb = [item for item in details if item["metrics"]["is_negative"]]
    summary = {
        "total_cases": len(cases),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "router_action_counts": action_counts,
        "expected_retrieve_rate": ratio(
            details,
            lambda item: item["router_action"] == "retrieve",
        ),
        "positive_recall_at_k": average(positive, "recall_at_k"),
        "positive_precision_at_k": average(positive, "precision_at_k"),
        "positive_mrr": average(positive, "rr"),
        "negative_empty_return_rate": ratio(
            negative,
            lambda item: item["returned_count"] == 0,
        ),
        "out_of_kb_direct_answer_rate": ratio(
            out_of_kb,
            lambda item: item["router_action"] != "retrieve",
        ),
        "avg_route_ms": round(average(timings, "route_ms"), 1),
        "avg_retrieval_ms": round(average(timings, "retrieval_ms"), 1),
        "avg_rerank_ms": round(average(timings, "rerank_ms"), 1),
        "avg_total_ms": round(average(timings, "total_ms"), 1),
    }
    return {
        "mode": mode,
        "summary": summary,
        "details": details,
    }


def build_metadata(dataset_path: str, top_k: int, threshold: float) -> dict:
    from config import settings

    return {
        "dataset": str(Path(dataset_path).resolve()),
        "top_k": top_k,
        "reranker_score_threshold": threshold,
        "reranker_model_path": settings.reranker_model_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "检索阶段统一使用 Router 提供的 query 或原问题，"
        "不调用查询改写器，以隔离 Router 对检索的影响",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_dataset.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=["rules", "llm"],
        action="append",
        default=[],
        help="可重复传入；不传时同时跑 rules 和 llm",
    )
    args = parser.parse_args()
    modes = args.mode or ["rules", "llm"]

    cases = load_dataset(args.dataset)
    store = VectorStore()
    reranker = ReRanker(score_threshold=args.threshold)

    reports = []
    for mode in modes:
        print(f"\n===== Router 模式: {mode} =====")
        report = run_mode(
            mode=mode,
            cases=cases,
            store=store,
            reranker=reranker,
            top_k=args.top_k,
        )
        report["metadata"] = build_metadata(args.dataset, args.top_k, args.threshold)
        target = save_experiment(
            report,
            experiment_id=f"agent-router-{mode}",
        )
        s = report["summary"]
        print(
            f"router_action={s['router_action_counts']}\n"
            f"Recall@{args.top_k}={s['positive_recall_at_k']:.3f}\n"
            f"Precision@{args.top_k}={s['positive_precision_at_k']:.3f}\n"
            f"MRR={s['positive_mrr']:.3f}\n"
            f"负样本空返回率={s['negative_empty_return_rate']:.3f}\n"
            f"平均 route={s['avg_route_ms']}ms, "
            f"平均 total={s['avg_total_ms']}ms"
        )
        print(f"报告已保存到: {target}")
        reports.append(report)

    print("\n===== rules vs llm 对照 =====")
    print(
        "mode | Recall@K | Precision@K | MRR | 负样本空返回率 | "
        "错误直接回答率(库外) | avg_route_ms | avg_total_ms"
    )
    for report in reports:
        s = report["summary"]
        print(
            f"{report['mode']} | {s['positive_recall_at_k']:.3f} | "
            f"{s['positive_precision_at_k']:.3f} | {s['positive_mrr']:.3f} | "
            f"{s['negative_empty_return_rate']:.3f} | "
            f"{s['out_of_kb_direct_answer_rate']:.3f} | "
            f"{s['avg_route_ms']:.1f} | {s['avg_total_ms']:.1f}"
        )

    summary_by_mode = {
        report["mode"]: report["summary"]
        for report in reports
    }
    for other_mode in ("rules", "llm"):
        if other_mode in summary_by_mode:
            continue
        other_path = Path("experiment_results") / f"agent-router-{other_mode}.json"
        if other_path.exists():
            summary_by_mode[other_mode] = json.loads(
                other_path.read_text(encoding="utf-8")
            )["summary"]

    comparison = {
        "summary": summary_by_mode,
        "metadata": build_metadata(args.dataset, args.top_k, args.threshold),
    }
    comparison_path = Path("experiment_results") / "agent-router-comparison.json"
    comparison_path.parent.mkdir(exist_ok=True)
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n对照报告已保存到: {comparison_path}")


if __name__ == "__main__":
    main()

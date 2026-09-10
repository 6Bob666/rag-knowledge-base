"""校准检索验证器：规则验证器的判断和人工标注一致吗？

评测的是 Verifier 这一层，而不是最终答案。对每道题：

    真实检索 → Reranker → Verifier 判断"够不够" 
    人工标注 → 返回的 chunk 里到底有没有相关文档

两个结论一对比，就能算出验证器是"该拦的拦住了"还是"把好结果也拦了"：

    判 sufficient 且真的命中   → 正确放行（TP）
    判 sufficient 但没命中     → 危险误判，会把跑题文档喂给模型（FP）
    判 insufficient 但命中了   → 代价是白跑一次检索（FN）
    判 insufficient 且没命中   → 正确拦截（TN）

同时对比"不做覆盖度检查"的退化验证器，用来看这一层到底带来了什么。

用法：
    python evaluate_verifier.py --top-k 3 --threshold 0.5
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from config import settings
from evaluate_retrieval import load_dataset
from services.plan_verify import (
    AlwaysSufficientVerifier,
    RuleBasedVerifier,
)
from services.reranker import ReRanker
from services.vector_store import VectorStore


def evaluate_case(
    case: dict,
    *,
    store: VectorStore,
    reranker: ReRanker,
    verifiers: dict,
    top_k: int,
) -> dict:
    question = case["question"]
    candidates = store.hybrid_search_records(question, top_k=top_k * 3)
    contexts = reranker.rerank_records(question, candidates, top_k=top_k)

    returned_ids = {str(item["id"]) for item in contexts}
    relevant_ids = {str(item) for item in case["relevant_chunk_ids"]}
    # 负样本（知识库里本来没有答案）不参与"是否命中"的统计。
    is_negative = not relevant_ids
    hit = bool(returned_ids & relevant_ids)

    verdicts = {}
    for name, verifier in verifiers.items():
        result = verifier.verify(
            question=question,
            candidates=candidates,
            contexts=contexts,
        )
        verdicts[name] = result.to_dict()

    return {
        "question": question,
        "is_negative": is_negative,
        "hit": hit,
        "candidate_count": len(candidates),
        "context_count": len(contexts),
        "returned_chunk_ids": sorted(returned_ids),
        "relevant_chunk_ids": sorted(relevant_ids),
        "verdicts": verdicts,
    }


def summarize(details: list[dict], verifier_name: str) -> dict:
    """统计验证器判断与人工标注的一致情况。"""
    positive = [item for item in details if not item["is_negative"]]
    negative = [item for item in details if item["is_negative"]]

    def count(items, *, sufficient, hit):
        return sum(
            1
            for item in items
            if item["verdicts"][verifier_name]["sufficient"] is sufficient
            and item["hit"] is hit
        )

    true_pass = count(positive, sufficient=True, hit=True)
    false_pass = count(positive, sufficient=True, hit=False)
    false_block = count(positive, sufficient=False, hit=True)
    true_block = count(positive, sufficient=False, hit=False)

    # 负样本上"判够用"就是误放行：知识库里根本没有答案。
    negative_pass = sum(
        1 for item in negative if item["verdicts"][verifier_name]["sufficient"]
    )

    total = len(positive)
    agreement = (true_pass + true_block) / total if total else 0.0
    coverages = [
        item["verdicts"][verifier_name]["coverage"]
        for item in positive
        if item["verdicts"][verifier_name]["coverage"] is not None
    ]
    replans = sum(
        1
        for item in details
        if item["verdicts"][verifier_name]["sufficient"] is False
    )

    return {
        "verifier": verifier_name,
        "positive_count": total,
        "negative_count": len(negative),
        "true_pass": true_pass,
        "false_pass": false_pass,
        "false_block": false_block,
        "true_block": true_block,
        "agreement_with_labels": round(agreement, 4),
        "false_pass_rate": round(false_pass / total, 4) if total else 0.0,
        "false_block_rate": round(false_block / total, 4) if total else 0.0,
        "negative_pass": negative_pass,
        "replan_trigger_rate": round(
            replans / len(details), 4
        )
        if details
        else 0.0,
        "avg_coverage": round(mean(coverages), 4) if coverages else None,
    }


def build_metadata(dataset_path: str, top_k: int, threshold: float) -> dict:
    return {
        "dataset": str(Path(dataset_path).resolve()),
        "top_k": top_k,
        "candidate_k": top_k * 3,
        "reranker_score_threshold": threshold,
        "verifier_min_coverage": settings.verifier_min_coverage,
        "knowledge_base_version": settings.knowledge_base_version,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(
    dataset_path: str = "evaluation_dataset.json",
    top_k: int = 3,
    threshold: float = 0.5,
):
    cases = load_dataset(dataset_path)
    store = VectorStore()
    reranker = ReRanker(score_threshold=threshold)
    verifiers = {
        "rules_coverage": RuleBasedVerifier(
            min_coverage=settings.verifier_min_coverage
        ),
        "no_coverage_check": AlwaysSufficientVerifier(),
    }

    details = [
        evaluate_case(
            case,
            store=store,
            reranker=reranker,
            verifiers=verifiers,
            top_k=top_k,
        )
        for case in cases
    ]

    summaries = {
        name: summarize(details, name) for name in verifiers
    }

    print(f"K={top_k}, Reranker threshold={threshold}")
    print(
        f"共 {len(details)} 道题"
        f"（正样本 {summaries['rules_coverage']['positive_count']}，"
        f"负样本 {summaries['rules_coverage']['negative_count']}）"
    )
    print()
    print(
        "verifier              一致率  误放行  误拦截  负样本误放行  触发重规划  平均覆盖度"
    )
    for name, summary in summaries.items():
        coverage = summary["avg_coverage"]
        print(
            f"{name:20s}"
            f"{summary['agreement_with_labels']:6.3f}  "
            f"{summary['false_pass']:5d}   "
            f"{summary['false_block']:5d}   "
            f"{summary['negative_pass']:10d}   "
            f"{summary['replan_trigger_rate']:9.3f}   "
            f"{'n/a' if coverage is None else f'{coverage:8.3f}'}"
        )

    report = {
        "summary": summaries,
        "metadata": build_metadata(dataset_path, top_k, threshold),
        "details": details,
    }
    # 阈值不同结论完全不同，所以文件名里带上阈值，两份报告可以并存对比。
    report_path = Path(f"verifier_calibration_report_threshold{threshold}.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n校准报告已保存到 {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_dataset.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.dataset, args.top_k, args.threshold)

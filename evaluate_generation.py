"""端到端生成层评测：Faithfulness、Context Recall、Answer Relevance。"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from config import settings
from services.llm_service import ask_llm, client
from services.eval_timings import summarize_timings
from services.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from services.reranker import ReRanker
from services.vector_store import VectorStore


def load_positive_cases(path: str) -> list[dict]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测集不存在: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    return [
        case
        for case in cases
        if case.get("relevant_chunk_ids") and case.get("answer")
    ]


def split_sentences(text: str) -> list[str]:
    for punctuation in ["。", "！", "？", "!", "?", "\n"]:
        text = text.replace(punctuation, "。")
    return [part.strip() for part in text.split("。") if len(part.strip()) >= 4]


JUDGE_SYSTEM_PROMPT = (
    "你是严格的评估助手。判断给出的信息是否成立。"
    "只输出 JSON，格式为 {\"supported\": true} 或 {\"supported\": false}，"
    "不要输出任何其他文字。"
)


def parse_yes_no(text: str) -> bool | None:
    """尽量稳健地把 LLM 输出解析成布尔值。"""
    cleaned = text.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "supported" in data:
            return bool(data["supported"])
    except (json.JSONDecodeError, TypeError):
        pass

    lowered = cleaned.lower()
    if lowered.startswith("是") or lowered.startswith("true") or lowered.startswith("yes"):
        return True
    if lowered.startswith("否") or lowered.startswith("false") or lowered.startswith("no"):
        return False
    return None


def llm_judge_yes_no(prompt: str, samples: int = 1) -> bool:
    """让 LLM 判断一次；samples>1 时多次采样取多数。"""
    votes = []
    for _ in range(samples):
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        answer = response.choices[0].message.content.strip()
        result = parse_yes_no(answer)
        votes.append(result if result is not None else False)
    return sum(votes) > samples / 2


def evaluate_one(
    store: VectorStore,
    reranker: ReRanker,
    case: dict,
    top_k: int,
    judge_samples: int = 1,
) -> dict:
    total_start = time.perf_counter()
    question = case["question"]
    ground_truth = case["answer"]

    retrieval_start = time.perf_counter()
    candidates = store.hybrid_search_records(question, top_k=top_k * 3)
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 1)

    rerank_start = time.perf_counter()
    contexts = reranker.rerank_records(question, candidates, top_k=top_k)
    rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 1)
    context_text = "\n".join(item["text"] for item in contexts)

    answer_start = time.perf_counter()
    answer = ask_llm(question, context_text)
    answer_llm_ms = round((time.perf_counter() - answer_start) * 1000, 1)

    answer_sentences = split_sentences(answer)
    judge_start = time.perf_counter()
    faithful_supported = sum(
        llm_judge_yes_no(
            f"参考资料：\n{context_text}\n\n"
            f"这句话是否能由参考资料支持？\n句子：{sentence}",
            samples=judge_samples,
        )
        for sentence in answer_sentences
    )
    faithfulness = (
        faithful_supported / len(answer_sentences) if answer_sentences else 0.0
    )

    ground_truth_sentences = split_sentences(ground_truth)
    recalled = sum(
        llm_judge_yes_no(
            f"参考资料：\n{context_text}\n\n"
            f"这条信息是否在参考资料中被提及？\n信息：{sentence}",
            samples=judge_samples,
        )
        for sentence in ground_truth_sentences
    )
    context_recall = (
        recalled / len(ground_truth_sentences) if ground_truth_sentences else 0.0
    )

    relevant = llm_judge_yes_no(
        f"问题：{question}\n\n回答：{answer}\n\n"
        "这个回答是否直接回答了问题？",
        samples=judge_samples,
    )
    judge_ms = round((time.perf_counter() - judge_start) * 1000, 1)

    returned_ids = [item["id"] for item in contexts]
    relevant_ids = case["relevant_chunk_ids"]
    recall_at_k = calculate_recall_at_k(returned_ids, relevant_ids, top_k)
    precision_at_k = calculate_precision_at_k(returned_ids, relevant_ids, top_k)
    rr = calculate_reciprocal_rank(returned_ids, relevant_ids, top_k)
    total_ms = round((time.perf_counter() - total_start) * 1000, 1)
    judge_calls = (
        len(answer_sentences) + len(ground_truth_sentences) + 1
    ) * judge_samples

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "rr": rr,
        "faithfulness": faithfulness,
        "context_recall": context_recall,
        "answer_relevance": 1.0 if relevant else 0.0,
        "answer_sentences": len(answer_sentences),
        "faithful_supported": faithful_supported,
        "returned_chunk_ids": [item["id"] for item in contexts],
        "timings": {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "answer_llm_ms": answer_llm_ms,
            "judge_ms": judge_ms,
            "total_ms": total_ms,
        },
        "judge_calls": judge_calls,
    }


def run(
    dataset_path: str = "evaluation_dataset.json",
    top_k: int = 3,
    threshold: float = 0.5,
    limit: int | None = None,
    judge_samples: int = 1,
):
    cases = load_positive_cases(dataset_path)
    if not cases:
        raise ValueError("没有可用于生成层评测的正样本")
    if limit is not None:
        cases = cases[:limit]

    store = VectorStore()
    reranker = ReRanker(score_threshold=threshold)
    results = [
        evaluate_one(store, reranker, case, top_k, judge_samples)
        for case in cases
    ]

    def average(key: str) -> float:
        return mean(item[key] for item in results)

    print(f"正样本={len(results)} 道，top_k={top_k}，threshold={threshold}")
    print("检索层：Recall@K   Precision@K       MRR")
    print(
        f"{average('recall_at_k'):12.3f}    "
        f"{average('precision_at_k'):12.3f}    "
        f"{average('rr'):8.3f}"
    )
    print("生成层：Faithfulness   ContextRecall   AnswerRelevance")
    print(
        f"{average('faithfulness'):15.3f}    "
        f"{average('context_recall'):13.3f}    "
        f"{average('answer_relevance'):14.3f}"
    )

    timing_records = [item["timings"] for item in results]
    timing_summary = summarize_timings(
        timing_records,
        keys=(
            "retrieval_ms",
            "rerank_ms",
            "answer_llm_ms",
            "judge_ms",
            "total_ms",
        ),
    )
    print("耗时统计（单位 ms）：")
    for stage, values in timing_summary.items():
        print(
            f"{stage:15} 平均={values['avg_ms']:8.1f} "
            f"最大={values['max_ms']:8.1f}"
        )

    report = {
        "metadata": {
            "dataset": str(Path(dataset_path).resolve()),
            "top_k": top_k,
            "reranker_score_threshold": threshold,
            "llm_model_name": settings.llm_model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "timing_summary": timing_summary,
        "results": results,
    }
    Path("generation_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("详细报告已保存到 generation_eval_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_dataset.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge-samples", type=int, default=1)
    args = parser.parse_args()
    run(args.dataset, args.top_k, args.threshold, args.limit, args.judge_samples)

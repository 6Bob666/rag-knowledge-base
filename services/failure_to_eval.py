"""把已人工标注的失败样本转换成评测集草稿。"""

from pathlib import Path

from services.failure_labeler import LABELS


EXPORTABLE_LABELS = {"retrieval", "knowledge_gap"}


def convert_labels_to_cases(records: list[dict]) -> list[dict]:
    """保留与 RAG 检索质量有关的标签，服务异常不转成检索题。"""
    cases = []
    for index, record in enumerate(records, start=1):
        label = record.get("label", "")
        if label not in EXPORTABLE_LABELS:
            continue

        question = record.get("question", "").strip()
        if not question:
            continue

        source_status = record.get("status", "unknown")
        label_text = LABELS.get(label, label)
        cases.append(
            {
                "id": f"failure-{index:03d}",
                "question": question,
                "source_status": source_status,
                "failure_label": label,
                "failure_label_text": label_text,
                "rewritten_query": "",
                "ground_truth_answer": "",
                "relevant_chunk_ids": [],
                "why_hard": (
                    f"来自真实失败记录：{source_status}；"
                    f"人工初判：{label_text}"
                ),
            }
        )
    return cases


def write_draft(cases: list[dict], output: str | Path) -> None:
    """把草稿写成 JSON 文件。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    import json

    target.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

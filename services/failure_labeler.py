"""失败样本的读取、标签写入与已标注判断。"""

import json
from datetime import datetime, timezone
from pathlib import Path


LABELS = {
    "retrieval": "检索/表达问题",
    "knowledge_gap": "知识库缺失/能力边界",
    "service_error": "服务异常",
}


def load_journal(path: str | Path) -> list[dict]:
    """读取 failure_journal.jsonl。"""
    target = Path(path)
    if not target.exists():
        return []
    entries = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def load_labels(path: str | Path) -> dict[str, str]:
    """读取已标注文件，返回 request_id -> label。"""
    target = Path(path)
    if not target.exists():
        return {}
    labels = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("request_id"):
            labels[entry["request_id"]] = entry.get("label", "")
    return labels


def append_label(
    entry: dict,
    label: str,
    path: str | Path,
) -> None:
    """把一条人工标注追加到 labels 文件。"""
    if label not in LABELS:
        raise ValueError(f"未知标签: {label}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "request_id": entry.get("request_id", ""),
        "status": entry.get("status", ""),
        "question": entry.get("question", ""),
        "label": label,
        "label_text": LABELS[label],
    }
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

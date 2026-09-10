"""人工标注失败样本。

运行：
    python label_failures.py

会逐条显示 failure_journal.jsonl 中尚未标注的记录，
输入 1 / 2 / 3 或 s（跳过）进行标注。
"""

import argparse
from pathlib import Path

from config import settings
from logging_config import logger
from services.failure_labeler import (
    LABELS,
    append_label,
    load_journal,
    load_labels,
)


CHOICES = {
    "1": "retrieval",
    "2": "knowledge_gap",
    "3": "service_error",
}


def ask_label(entry: dict) -> str | None:
    print("\n" + "-" * 60)
    print(f"request_id={entry.get('request_id', '-')} | status={entry.get('status', '-')}")
    print(f"问题：{entry.get('question', '')}")
    print("标签：1=检索/表达问题  2=知识库缺失/能力边界  3=服务异常  s=跳过")
    choice = input("请输入：").strip().lower()
    if choice in CHOICES:
        return CHOICES[choice]
    if choice in ("s", "skip", ""):
        return None
    print("无效输入，已跳过本条")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--journal",
        default=settings.failure_journal_path,
        help="失败样本文件路径",
    )
    parser.add_argument(
        "--labels",
        default=str(Path(settings.failure_journal_path).parent / "failure_labels.jsonl"),
        help="标签输出文件路径",
    )
    args = parser.parse_args()

    entries = load_journal(args.journal)
    if not entries:
        print(f"没有可标注的失败样本: {args.journal}")
        return

    already = load_labels(args.labels)
    pending = [entry for entry in entries if entry.get("request_id") not in already]
    if not pending:
        print("所有样本都已标注，无需继续")
        return

    print(f"共 {len(entries)} 条失败样本，其中 {len(pending)} 条待标注")
    for entry in pending:
        label = ask_label(entry)
        if label is None:
            continue
        try:
            append_label(entry, label, args.labels)
            print(f"已保存: {entry.get('request_id')} -> {LABELS[label]}")
        except ValueError as exc:
            logger.warning("保存失败: %s", exc)

    print(f"\n标注结果已保存到 {args.labels}")


if __name__ == "__main__":
    main()

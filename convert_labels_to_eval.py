"""把已标注失败样本转换成评测集草稿。

运行：
    python convert_labels_to_eval.py \
        --labels logs/failure_labels.jsonl \
        --output evaluation_dataset_hard.draft.json
"""

import argparse
from pathlib import Path

from services.failure_labeler import load_labels, LABELS
from services.failure_to_eval import convert_labels_to_cases, write_draft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        default="logs/failure_labels.jsonl",
        help="人工标注文件路径",
    )
    parser.add_argument(
        "--output",
        default="evaluation_dataset_hard.draft.json",
        help="评测集草稿输出路径",
    )
    args = parser.parse_args()

    label_map = load_labels(args.labels)
    if not label_map:
        print(f"{args.labels} 中没有已标注样本")
        return

    # load_labels 只保留 request_id->label，转换需要原始记录。
    # 这里从标注文件读取完整行记录。
    import json

    records = []
    path = Path(args.labels)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    cases = convert_labels_to_cases(records)
    write_draft(cases, args.output)

    counts: dict[str, int] = {}
    for record in records:
        label = record.get("label", "")
        if label in LABELS:
            counts[label] = counts.get(label, 0) + 1
    print("标签统计：")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    print(f"转换出 {len(cases)} 条评测集草稿，已保存到 {args.output}")


if __name__ == "__main__":
    main()

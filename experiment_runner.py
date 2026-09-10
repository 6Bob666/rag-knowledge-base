"""运行并对比 RAG 检索实验。

运行一次实验：
    python experiment_runner.py run --name exp-reranker-001 \
        --dataset evaluation_dataset.json --top-k 3 --threshold 0.5

对比已保存的实验：
    python experiment_runner.py compare
"""

import argparse
from pathlib import Path

from evaluate_retrieval import run as run_retrieval_eval
from services.experiment_store import (
    format_comparison,
    load_experiments,
    make_experiment_id,
    save_experiment,
)


def cmd_run(args: argparse.Namespace) -> None:
    report = run_retrieval_eval(
        dataset_path=args.dataset,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    experiment_id = args.name or make_experiment_id()
    target = save_experiment(
        report,
        output_dir=Path(args.output_dir),
        experiment_id=experiment_id,
    )
    print(f"实验报告已保存到: {target}")


def cmd_compare(args: argparse.Namespace) -> None:
    reports = load_experiments(Path(args.output_dir))
    if not reports:
        print(f"{args.output_dir} 下没有实验报告")
        return
    print(format_comparison(reports, strategy=args.strategy))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索实验运行器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="运行一次检索实验")
    run_parser.add_argument("--name", default=None, help="实验名称")
    run_parser.add_argument("--dataset", default="evaluation_dataset.json")
    run_parser.add_argument("--top-k", type=int, default=3)
    run_parser.add_argument("--threshold", type=float, default=0.0)
    run_parser.add_argument(
        "--output-dir",
        default="experiment_results",
        help="报告保存目录",
    )
    run_parser.set_defaults(func=cmd_run)

    compare_parser = subparsers.add_parser("compare", help="对比已保存的实验")
    compare_parser.add_argument(
        "--output-dir",
        default="experiment_results",
        help="报告保存目录",
    )
    compare_parser.add_argument(
        "--strategy",
        choices=["baseline", "reranked"],
        default="reranked",
    )
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

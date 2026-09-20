"""Prepare a reproducible, small T2Retrieval slice for the local RAG project.

The official MTEB dataset keeps corpus, queries, and qrels in separate parquet
files. This script preserves the original corpus IDs so retrieval results can
be evaluated with the existing ``relevant_chunk_ids`` metric.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


REPO_ID = "mteb/T2Retrieval"
REPO_TYPE = "dataset"
FILES = {
    "corpus": "corpus/dev-00000-of-00001.parquet",
    "queries": "queries/dev-00000-of-00001.parquet",
    "qrels": "data/dev-00000-of-00001.parquet",
}


def download_paths() -> dict[str, Path]:
    return {
        name: Path(
            hf_hub_download(
                REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
            )
        )
        for name, filename in FILES.items()
    }


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = defaultdict(list)
    for row in pq.ParquetFile(path).iter_batches():
        for item in row.to_pylist():
            if int(item["score"]) > 0:
                qrels[str(item["query-id"])].append(str(item["corpus-id"]))
    return dict(qrels)


def load_queries(path: Path, selected_ids: set[str]) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in pq.ParquetFile(path).iter_batches():
        for item in row.to_pylist():
            query_id = str(item.get("_id", item.get("id")))
            if query_id in selected_ids:
                queries[query_id] = str(item["text"])
    return queries


def select_corpus(path: Path, required_ids: set[str], target_size: int) -> list[dict]:
    selected: list[dict] = []
    selected_ids: set[str] = set()

    # First pass: preserve every positive document needed by the selected qrels.
    for row in pq.ParquetFile(path).iter_batches():
        for item in row.to_pylist():
            corpus_id = str(item.get("_id", item.get("id")))
            if corpus_id in required_ids:
                selected.append({"id": corpus_id, "text": str(item["text"])})
                selected_ids.add(corpus_id)

    # Second pass: add deterministic filler documents to make the corpus large
    # enough for a meaningful local retrieval experiment.
    if len(selected) < target_size:
        for row in pq.ParquetFile(path).iter_batches():
            for item in row.to_pylist():
                corpus_id = str(item.get("_id", item.get("id")))
                if corpus_id in selected_ids:
                    continue
                selected.append({"id": corpus_id, "text": str(item["text"])})
                selected_ids.add(corpus_id)
                if len(selected) >= target_size:
                    return selected
    return selected


def prepare(output_dir: Path, query_count: int, corpus_size: int) -> dict:
    paths = download_paths()
    all_qrels = load_qrels(paths["qrels"])
    selected_query_ids = sorted(all_qrels)[:query_count]
    selected_query_set = set(selected_query_ids)
    queries = load_queries(paths["queries"], selected_query_set)
    selected_query_ids = [query_id for query_id in selected_query_ids if query_id in queries]
    selected_qrels = {query_id: all_qrels[query_id] for query_id in selected_query_ids}
    required_corpus_ids = {
        corpus_id
        for corpus_ids in selected_qrels.values()
        for corpus_id in corpus_ids
    }
    corpus = select_corpus(paths["corpus"], required_corpus_ids, corpus_size)
    corpus_ids = {item["id"] for item in corpus}

    # Every evaluation question must have all its positive documents available.
    evaluation = [
        {
            "question": queries[query_id],
            "answer": "",
            "relevant_chunk_ids": [
                f"t2:{corpus_id}"
                for corpus_id in selected_qrels[query_id]
                if corpus_id in corpus_ids
            ],
            "source_query_id": query_id,
        }
        for query_id in selected_query_ids
    ]
    evaluation = [item for item in evaluation if item["relevant_chunk_ids"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as file:
        for item in corpus:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "evaluation_dataset.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "dataset": REPO_ID,
        "dataset_revision": "official MTEB T2Retrieval dev",
        "license": "Apache-2.0",
        "query_count": len(evaluation),
        "corpus_count": len(corpus),
        "positive_document_count": len(required_corpus_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "corpus": str(corpus_path.name),
            "evaluation": "evaluation_dataset.json",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/t2retrieval")
    parser.add_argument("--query-count", type=int, default=500)
    parser.add_argument("--corpus-size", type=int, default=5000)
    args = parser.parse_args()
    if args.query_count <= 0 or args.corpus_size <= 0:
        raise SystemExit("query-count 和 corpus-size 必须大于 0")
    manifest = prepare(Path(args.output_dir), args.query_count, args.corpus_size)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

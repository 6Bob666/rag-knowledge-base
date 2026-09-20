"""Import the prepared T2Retrieval corpus into an isolated Chroma collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.vector_store import VectorStore


def import_corpus(
    corpus_path: Path,
    *,
    collection_name: str = "t2_kb_docs",
    batch_size: int = 128,
    id_prefix: str = "t2:",
) -> int:
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    records = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("语料文件为空")

    store = VectorStore(collection_name=collection_name)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [str(item["text"]) for item in batch]
        ids = [f"{id_prefix}{item['id']}" for item in batch]
        metadatas = [
            {
                "source": "mteb/T2Retrieval",
                "dataset": "T2Retrieval",
                "dataset_split": "dev",
                "document_id": "t2retrieval-dev",
                "corpus_id": str(item["id"]),
                "chunk_id": doc_id,
                "license": "Apache-2.0",
            }
            for item, doc_id in zip(batch, ids)
        ]
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        print(f"imported {min(start + len(batch), len(records))}/{len(records)}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/t2retrieval/corpus.jsonl")
    parser.add_argument("--collection", default="t2_kb_docs")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    count = import_corpus(
        Path(args.corpus),
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    print(f"imported_total={count}, collection={args.collection}")


if __name__ == "__main__":
    main()

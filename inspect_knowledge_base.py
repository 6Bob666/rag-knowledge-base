"""查看知识库中的 chunk，便于人工标注评测数据。"""

import argparse
import sys

import chromadb

from config import settings


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_chunks() -> dict[str, list[tuple[str, str, dict]]]:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(name="kb_docs")
    data = collection.get(include=["documents", "metadatas"])

    grouped: dict[str, list[tuple[str, str, dict]]] = {}
    for chunk_id, document, metadata in zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    ):
        metadata = metadata or {}
        source = str(metadata.get("source", "unknown"))
        grouped.setdefault(source, []).append((chunk_id, document, metadata))
    return grouped


def main(limit: int, full: bool) -> None:
    grouped = load_chunks()
    for source, chunks in grouped.items():
        print(f"\n=== {source}（{len(chunks)} 个 chunk）===")
        for chunk_id, document, metadata in chunks:
            chunk_index = metadata.get("chunk_index", "?")
            document = document.replace("\u200b", "").replace("\ufeff", "")
            shown = document if full else document[:limit]
            suffix = "" if full or len(document) <= limit else "..."
            print(f"[{chunk_index}] {chunk_id}")
            print(f"    {shown}{suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120, help="每个 chunk 显示的字符数")
    parser.add_argument("--full", action="store_true", help="显示完整文本")
    args = parser.parse_args()
    main(args.limit, args.full)

"""向量索引规模压测：HNSW 参数 vs Recall@K 与延迟。

评测的是**索引层**，不是端到端问答：用合成向量，因为只有合成数据才能控制
"结构有多难"（簇数、簇内散度），把索引行为单独拎出来看。真实语料的端到端
延迟由 evaluate_retrieval.py 那条链路覆盖。

数据集构造：

    1. 随机取一个 intrinsic_dim 维子空间作为"有效语义空间"；
    2. 在这个子空间里撒 n_clusters 个中心，向量 = 中心 + 高斯噪声；
    3. 再映射回 dim 维并归一化（余弦距离）。
    查询 = 随机取一个语料向量 + 噪声，模拟"用户问题与某个 chunk 语义相近"。

**intrinsic_dim 是这份压测里最关键的变量**，因为它决定近邻之间是不是"分得开"：

    intrinsic_dim=32   接近真实 embedding 的低内在维度，近邻分得开，ANN 好使
    intrinsic_dim=512  满维随机数据，大量近邻分数几乎相同，是 ANN 的最坏情况

同一个索引、同一份代码，这两种数据下的召回能差出一大截。所以报告必须带上
这个参数，否则数字没有意义——这也是"ANN 压测不能只看规模"的原因。

对照的四种配置对应真实项目里最容易踩的坑：

    hnsw_default     M=16, construction_ef=200, search_ef=100（Chroma 常用档）
    hnsw_small_ef    search_ef 小于等于 top_k：图几乎没展开，召回暴跌
    hnsw_small_m     M=8：每个节点邻居太少，连图都连不好
    hnsw_high_recall M=32, construction_ef=400, search_ef=400：拿延迟换召回
    exact_numpy      暴力检索基线，用来判断"这个规模到底还需不需要 ANN"

用法：
    python benchmark_index.py --sizes 5000,20000,50000
    python benchmark_index.py --sizes 50000 --intrinsic-dim 32
    python benchmark_index.py --sizes 50000 --intrinsic-dim 512 --label high_dim
"""

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from services.metrics import percentile, recall_at_k_from_ids


@dataclass(frozen=True)
class IndexConfig:
    """一个待测索引配置；exact 为真时表示不走 Chroma，用 numpy 暴力算。"""

    name: str
    space: str = "cosine"
    m: int = 16
    construction_ef: int = 200
    search_ef: int = 100
    exact: bool = False

    def collection_metadata(self) -> dict:
        return {
            "hnsw:space": self.space,
            "hnsw:M": self.m,
            "hnsw:construction_ef": self.construction_ef,
            "hnsw:search_ef": self.search_ef,
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "space": self.space,
            "M": self.m,
            "construction_ef": self.construction_ef,
            "search_ef": self.search_ef,
            "exact": self.exact,
        }


DEFAULT_CONFIGS = (
    IndexConfig("hnsw_default"),
    IndexConfig("hnsw_small_ef", m=16, construction_ef=200, search_ef=10),
    IndexConfig("hnsw_small_m", m=8, construction_ef=100, search_ef=100),
    IndexConfig(
        "hnsw_high_recall", m=32, construction_ef=400, search_ef=400
    ),
    IndexConfig("exact_numpy", exact=True),
)


def build_corpus(
    *,
    size: int,
    dim: int,
    queries: int,
    seed: int = 7,
    intrinsic_dim: int = 32,
    cluster_noise: float = 0.35,
    query_noise: float = 0.30,
):
    """生成有簇结构的语料与查询，返回 (corpus, query_vectors, exact_ids)。"""
    if size <= 0 or dim <= 0 or queries <= 0:
        raise ValueError("size / dim / queries 必须大于 0")
    if not 1 <= intrinsic_dim <= dim:
        raise ValueError("intrinsic_dim 必须在 1 到 dim 之间")

    rng = np.random.default_rng(seed)
    # 用一个正交基把低维语义空间嵌进 dim 维，保证数据"看起来是 512 维"，
    # 但真正变化的只有 intrinsic_dim 个方向。
    if intrinsic_dim < dim:
        raw_basis = rng.normal(size=(dim, intrinsic_dim)).astype(np.float32)
        basis, _ = np.linalg.qr(raw_basis)
    else:
        basis = np.eye(dim, dtype=np.float32)

    n_clusters = max(1, min(size, size // 200 or 1))
    centers = rng.normal(size=(n_clusters, intrinsic_dim)).astype(np.float32)

    assignment = rng.integers(0, n_clusters, size=size)
    corpus = centers[assignment] + cluster_noise * rng.normal(
        size=(size, intrinsic_dim)
    ).astype(np.float32)
    corpus = corpus @ basis.T
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

    seeds = corpus[rng.integers(0, size, size=queries)]
    noise = query_noise * rng.normal(size=(queries, intrinsic_dim)).astype(
        np.float32
    )
    query_vectors = seeds + noise @ basis.T
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    exact_ids = [
        [str(index) for index in row]
        for row in exact_topk(query_vectors, corpus, k=100)
    ]
    return corpus, query_vectors, exact_ids


def exact_topk(
    query_vectors: np.ndarray, corpus: np.ndarray, *, k: int
) -> np.ndarray:
    """暴力检索的精确 Top-K（向量已归一化，点积即余弦相似度）。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")
    scores = query_vectors @ corpus.T
    # 用 argpartition 取前 k 再排序，避免对全量做排序。
    top_k = np.argpartition(-scores, kth=min(k, corpus.shape[0]) - 1, axis=1)
    top_k = top_k[:, :k]
    ordered = np.take_along_axis(scores, top_k, axis=1)
    order = np.argsort(-ordered, axis=1)
    return np.take_along_axis(top_k, order, axis=1)


def measure_exact(
    *,
    query_vectors: np.ndarray,
    corpus: np.ndarray,
    exact_ids: list[list[str]],
    top_k: int,
) -> dict:
    """暴力检索基线：既不建索引，也不会有召回损失。"""
    latencies = []
    for query in query_vectors:
        start = time.perf_counter()
        exact_topk(
            query.reshape(1, -1), corpus, k=top_k
        )
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "build_seconds": 0.0,
        "recall_at_k": 1.0,
        "query_ms_p50": round(percentile(latencies, 0.5), 2),
        "query_ms_p95": round(percentile(latencies, 0.95), 2),
        "query_ms_mean": round(sum(latencies) / len(latencies), 2),
        "qps": round(1000 / (sum(latencies) / len(latencies)), 1),
        "recall_source": "exact_ids",
    }


def measure_hnsw(
    *,
    client,
    config: IndexConfig,
    corpus: np.ndarray,
    query_vectors: np.ndarray,
    exact_ids: list[list[str]],
    top_k: int,
    batch_size: int = 2000,
) -> dict:
    """把语料灌进 Chroma 的 HNSW 索引，再测召回与单条查询延迟。"""
    name = f"bench_{config.name}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name=name, metadata=config.collection_metadata()
    )

    build_start = time.perf_counter()
    for offset in range(0, corpus.shape[0], batch_size):
        batch = corpus[offset : offset + batch_size]
        collection.add(
            embeddings=batch.tolist(),
            ids=[str(index) for index in range(offset, offset + len(batch))],
        )
    build_seconds = time.perf_counter() - build_start

    latencies = []
    returned = []
    # 一次查询一个向量，模拟线上"一个用户问题一次检索"的调用方式。
    for query in query_vectors:
        start = time.perf_counter()
        result = collection.query(
            query_embeddings=[query.tolist()],
            n_results=top_k,
            include=[],
        )
        latencies.append((time.perf_counter() - start) * 1000)
        returned.append(list(result["ids"][0]))

    recalls = [
        recall_at_k_from_ids(got, exact, top_k)
        for got, exact in zip(returned, exact_ids)
    ]
    client.delete_collection(name)
    return {
        "build_seconds": round(build_seconds, 2),
        "ingest_per_second": round(corpus.shape[0] / build_seconds, 1),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "query_ms_p50": round(percentile(latencies, 0.5), 2),
        "query_ms_p95": round(percentile(latencies, 0.95), 2),
        "query_ms_mean": round(sum(latencies) / len(latencies), 2),
        "qps": round(1000 / (sum(latencies) / len(latencies)), 1),
        "recall_source": "exact_ids",
    }


def run(
    *,
    sizes: list[int],
    dim: int = 512,
    intrinsic_dim: int = 32,
    queries: int = 20,
    top_k: int = 10,
    seed: int = 7,
    label: str = "",
    configs: tuple[IndexConfig, ...] = DEFAULT_CONFIGS,
) -> dict:
    import chromadb

    workdir = Path(tempfile.mkdtemp(prefix="rag-index-bench-"))
    client = chromadb.PersistentClient(path=str(workdir))
    results = []
    try:
        for size in sizes:
            corpus, query_vectors, exact_ids = build_corpus(
                size=size,
                dim=dim,
                queries=queries,
                seed=seed,
                intrinsic_dim=intrinsic_dim,
            )
            print(
                f"\n=== size={size} dim={dim} intrinsic={intrinsic_dim} "
                f"queries={queries} top_k={top_k} ==="
            )
            print(
                "config              recall@K  build(s)  ingest/s  "
                "p50(ms)  p95(ms)   qps"
            )
            for config in configs:
                if config.exact:
                    metrics = measure_exact(
                        query_vectors=query_vectors,
                        corpus=corpus,
                        exact_ids=exact_ids,
                        top_k=top_k,
                    )
                else:
                    metrics = measure_hnsw(
                        client=client,
                        config=config,
                        corpus=corpus,
                        query_vectors=query_vectors,
                        exact_ids=exact_ids,
                        top_k=top_k,
                    )
                print(
                    f"{config.name:18s}"
                    f"{metrics['recall_at_k']:8.3f}  "
                    f"{metrics['build_seconds']:8.2f}  "
                    f"{metrics.get('ingest_per_second', 0.0):8.1f}  "
                    f"{metrics['query_ms_p50']:7.2f}  "
                    f"{metrics['query_ms_p95']:7.2f}  "
                    f"{metrics['qps']:6.1f}"
                )
                results.append(
                    {
                        "size": size,
                        "config": config.to_dict(),
                        **metrics,
                    }
                )
    finally:
        # 压测会产生大量临时索引文件，跑完必须清掉。
        shutil.rmtree(workdir, ignore_errors=True)

    report = {
        "metadata": {
            "sizes": sizes,
            "dim": dim,
            "intrinsic_dim": intrinsic_dim,
            "queries": queries,
            "top_k": top_k,
            "seed": seed,
            "label": label or f"intrinsic{intrinsic_dim}",
            "vector_source": "synthetic_clustered_low_rank",
            "chromadb_version": chromadb.__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    report_path = Path(
        f"index_benchmark_report_intrinsic{intrinsic_dim}.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n压测报告已保存到 {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="向量索引规模压测")
    parser.add_argument(
        "--sizes", default="5000,20000,50000", help="逗号分隔的语料规模"
    )
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument(
        "--intrinsic-dim",
        type=int,
        default=32,
        help="有效语义维度；32 接近真实 embedding，512 为最坏情况",
    )
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    run(
        sizes=sizes,
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        queries=args.queries,
        top_k=args.top_k,
        seed=args.seed,
        label=args.label,
    )


if __name__ == "__main__":
    main()

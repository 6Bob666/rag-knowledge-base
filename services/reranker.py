# services/reranker.py
from pathlib import Path

from config import PROJECT_ROOT, settings
from logging_config import logger

DEFAULT_MODEL_PATH = Path(settings.reranker_model_path)


class ReRanker:
    def __init__(
        self,
        model_path=None,
        score_threshold=None,
        model=None,
        predict_batch_size: int = 32,
    ):
        configured_path = model_path or settings.reranker_model_path
        self.model_path = Path(configured_path)
        self.score_threshold = (
            settings.reranker_score_threshold
            if score_threshold is None else score_threshold
        )
        if predict_batch_size <= 0:
            raise ValueError("predict_batch_size 必须大于 0")
        self.predict_batch_size = predict_batch_size
        # 测试时注入假的模型，避免下载和加载真实模型。
        if model is not None:
            self.model = model
            self.device = "test"
            return
        import torch
        from sentence_transformers import CrossEncoder

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Reranker 模型不存在: {self.model_path}。"
                "请先下载模型，或设置 RERANKER_MODEL_PATH。"
            )
        logger.info("正在加载 Reranker 模型: %s", self.model_path)
        model_kwargs = (
            {"torch_dtype": torch.float16}
            if self.device == "cuda" else {}
        )
        self.model = CrossEncoder(
            str(self.model_path),
            trust_remote_code=True,
            device=self.device,
            model_kwargs=model_kwargs,
        )
        logger.info("Reranker 模型加载完毕，device=%s", self.device)

    def _predict(self, pairs):
        """统一模型推理入口，并兼容测试替身的简化 predict 签名。"""
        import torch

        while True:
            try:
                return self.model.predict(
                    pairs,
                    batch_size=self.predict_batch_size,
                    show_progress_bar=False,
                )
            except TypeError:
                return self.model.predict(pairs, show_progress_bar=False)
            except torch.cuda.OutOfMemoryError:
                if self.device != "cuda" or self.predict_batch_size <= 1:
                    raise
                self.predict_batch_size = max(1, self.predict_batch_size // 2)
                torch.cuda.empty_cache()
                logger.warning(
                    "Reranker 显存不足，自动将 predict_batch_size 降为 %s",
                    self.predict_batch_size,
                )

    def _rank_records(
        self,
        records: list[dict],
        scores,
        top_k: int,
    ) -> list[dict]:
        scored = [
            (record, float(score))
            for record, score in zip(records, scores)
            if float(score) >= self.score_threshold
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [record for record, _ in scored[:top_k]]

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[str]:
        """排序、过滤低分文档，并返回前 top_k 个。"""
        if not documents or top_k <= 0:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._predict(pairs)
        scored_docs = [
            (doc, float(score)) for doc, score in zip(documents, scores)
            if float(score) >= self.score_threshold
        ]
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def rerank_records(self, query: str, records: list[dict], top_k: int = 3) -> list[dict]:
        """重排记录但保留来源 metadata。"""
        if not records or top_k <= 0:
            return []
        pairs = [[query, record["text"]] for record in records]
        scores = self._predict(pairs)
        return self._rank_records(records, scores, top_k)

    def rerank_records_batch(
        self,
        queries: list[str],
        record_groups: list[list[dict]],
        top_k: int = 3,
    ) -> list[list[dict]]:
        """批量重排多道问题，减少模型调用和 GPU 调度开销。"""
        if len(queries) != len(record_groups):
            raise ValueError("queries 和 record_groups 长度必须一致")
        if top_k <= 0:
            return [[] for _ in record_groups]

        pairs = []
        spans = []
        for query, records in zip(queries, record_groups):
            start = len(pairs)
            pairs.extend([[query, record["text"]] for record in records])
            spans.append((start, len(pairs)))
        if not pairs:
            return [[] for _ in record_groups]

        scores = self._predict(pairs)
        if len(scores) != len(pairs):
            raise ValueError("Reranker 返回的分数数量与候选数量不一致")
        return [
            self._rank_records(records, scores[start:end], top_k)
            for records, (start, end) in zip(record_groups, spans)
        ]

# services/reranker.py
from pathlib import Path

from config import PROJECT_ROOT, settings
from logging_config import logger

DEFAULT_MODEL_PATH = Path(settings.reranker_model_path)


class ReRanker:
    def __init__(self, model_path=None, score_threshold=None, model=None):
        configured_path = model_path or settings.reranker_model_path
        self.model_path = Path(configured_path)
        self.score_threshold = (
            settings.reranker_score_threshold
            if score_threshold is None else score_threshold
        )
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
        self.model = CrossEncoder(str(self.model_path), trust_remote_code=True, device=self.device)
        logger.info("Reranker 模型加载完毕，device=%s", self.device)

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[str]:
        """排序、过滤低分文档，并返回前 top_k 个。"""
        if not documents or top_k <= 0:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
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
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = [
            (record, float(score))
            for record, score in zip(records, scores)
            if float(score) >= self.score_threshold
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [record for record, _ in scored[:top_k]]

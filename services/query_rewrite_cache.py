"""带版本化缓存键的查询改写器。"""

import hashlib
import json
from collections.abc import Callable

from services.cache import TTLCache


class CachedQueryRewriter:
    def __init__(
        self,
        rewriter: Callable,
        *,
        cache: TTLCache[str],
        model_name: str,
        knowledge_base_version: str,
    ):
        self.rewriter = rewriter
        self.cache = cache
        self.model_name = model_name
        self.knowledge_base_version = knowledge_base_version

    def _key(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        payload = {
            "question": question,
            "history": history or [],
            "model": self.model_name,
            "knowledge_base_version": self.knowledge_base_version,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"query-rewrite:{digest}"

    def __call__(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        key = self._key(question, history)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        rewritten = self.rewriter(question, history)
        rewritten = (rewritten or "").strip() or question
        self.cache.set(key, rewritten)
        return rewritten

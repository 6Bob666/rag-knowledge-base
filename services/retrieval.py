# services/retrieval.py
import math
import re
from collections import defaultdict
from typing import List, Tuple

# ---------- 1. BM25 检索器（纯 Python 实现） ----------
class BM25Retriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.avgdl = 0
        self.idf = {}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # 简单的分词：按非字母数字字符分割，转为小写
        return re.findall(r'\w+', text.lower())

    def _build_index(self):
        # 计算文档长度和平均长度
        doc_lengths = [len(tokens) for tokens in self.tokenized_corpus]
        total_docs = len(doc_lengths)
        self.avgdl = sum(doc_lengths) / total_docs if total_docs > 0 else 0

        # 计算 IDF（逆文档频率）
        df = defaultdict(int)  # 每个词出现在多少个文档中
        for tokens in self.tokenized_corpus:
            for word in set(tokens):
                df[word] += 1

        # BM25 的 IDF 公式
        for word, freq in df.items():
            self.idf[word] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores = [0.0] * len(self.documents)
        for q_word in query_tokens:
            if q_word not in self.idf:
                continue
            idf_value = self.idf[q_word]
            for i, doc_tokens in enumerate(self.tokenized_corpus):
                if q_word in doc_tokens:
                    # BM25 分数（简化版，不考虑词频饱和）
                    doc_len = len(doc_tokens)
                    score = idf_value * (1 + 1) / (1 + 0.5 * (doc_len / self.avgdl))
                    scores[i] += score

        # 排序并返回 (索引, 分数)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------- 2. TF-IDF 向量检索（使用 scikit-learn） ----------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class TfidfRetriever:
    def __init__(self, documents: List[str]):
        if not SKLEARN_AVAILABLE:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_vec = self.vectorizer.transform([query])
        # 计算余弦相似度（TF-IDF 矩阵乘以查询向量）
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------- 3. RRF 融合 ----------
def reciprocal_rank_fusion(
    results_list: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    多路召回结果 RRF 融合
    results_list: 每路检索结果，格式为 [(doc_index, score), ...]
    """
    fused_scores = defaultdict(float)
    for results in results_list:
        for rank, (doc_idx, _) in enumerate(results):
            fused_scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# ---------- 4. 统一的检索服务 ----------
class RetrievalService:
    def __init__(self):
        self.documents: List[str] = []
        self.bm25_retriever = None
        self.tfidf_retriever = None

    def update_documents(self, documents: List[str]):
        """更新文档库，重建索引"""
        self.documents = documents
        # 重建 BM25
        self.bm25_retriever = BM25Retriever(documents)
        # 重建 TF-IDF（如果可用）
        if SKLEARN_AVAILABLE:
            self.tfidf_retriever = TfidfRetriever(documents)
        else:
            self.tfidf_retriever = None

    def add_document(self, document: str):
        """添加单个文档"""
        self.documents.append(document)
        self.update_documents(self.documents)

    def hybrid_search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[str]]:
        """
        混合检索：BM25 + TF-IDF 向量检索，RRF 融合
        返回 (索引列表, 文档内容列表)
        """
        if not self.documents:
            return [], []

        # 1. BM25 检索（取 top_k*2 用于融合）
        bm25_results = self.bm25_retriever.search(query, top_k=top_k*2) if self.bm25_retriever else []

        # 2. TF-IDF 向量检索
        tfidf_results = []
        if self.tfidf_retriever:
            tfidf_results = self.tfidf_retriever.search(query, top_k=top_k*2)

        # 3. 收集所有非空结果进行融合
        results_list = []
        if bm25_results:
            results_list.append(bm25_results)
        if tfidf_results:
            results_list.append(tfidf_results)

        if not results_list:
            return [], []

        # 4. RRF 融合
        fused = reciprocal_rank_fusion(results_list, k=60)

        # 5. 取 top_k
        final_indices = [idx for idx, _ in fused[:top_k]]
        final_sources = [self.documents[idx] for idx in final_indices]
        return final_indices, final_sources


# 全局单例
retriever = RetrievalService()


# 为了方便原有代码调用，保留一个函数接口
def retrieve(query: str, top_k: int = 5) -> Tuple[List[int], List[str]]:
    return retriever.hybrid_search(query, top_k=top_k)
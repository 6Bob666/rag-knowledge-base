import uuid
import jieba
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import settings

class VectorStore:
    def __init__(self, collection_name="kb_docs"):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer(settings.embedding_model_path)
        # BM25 相关属性（首次使用时惰性初始化）
        self._bm25 = None
        self._all_docs = None
        self._all_metadatas = None
        self._all_ids = None

    def add_texts(self, texts, metadatas=None, ids=None):
        """将文本列表存入向量库"""
        if not texts:
            return
        embeddings = self.model.encode(texts).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{"source": "upload"} for _ in texts],
            ids=ids or [str(uuid.uuid4()) for _ in texts]
        )
        # 清空 BM25 缓存，下次检索时会重新构建
        self._bm25 = None
        self._all_docs = None
        self._all_metadatas = None
        self._all_ids = None

    def delete_by_document_id(self, document_id: str) -> int:
        """删除指定文档的全部 chunk，返回删除数量。"""
        records = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        ids_to_delete = records.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            self._bm25 = None
            self._all_docs = None
            self._all_metadatas = None
            self._all_ids = None
        return len(ids_to_delete)

    def list_documents(self) -> list[dict]:
        """按 document_id 汇总已入库文档。"""
        records = self.collection.get(include=["metadatas"])
        documents = {}
        for metadata in records.get("metadatas", []):
            metadata = metadata or {}
            document_id = metadata.get("document_id")
            if document_id is None:
                continue
            item = documents.setdefault(
                document_id,
                {"document_id": document_id, "source": metadata.get("source", document_id), "chunk_count": 0},
            )
            item["chunk_count"] += 1
        return list(documents.values())

    def search(self, query, top_k=3):
        """纯向量检索"""
        q_emb = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents"]
        )
        return results['documents'][0] if results['documents'] else []

    def search_records(self, query, top_k=3):
        q_emb = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return [
            {"id": doc_id, "text": text, "metadata": metadata or {}}
            for doc_id, text, metadata in zip(ids, documents, metadatas)
        ]

    def _ensure_bm25_index(self):
        """构建或重建 BM25 索引"""
        if self._bm25 is not None:
            return
        # 从 Chroma 获取所有文档
        all_data = self.collection.get(include=['documents', 'metadatas'])
        self._all_docs = all_data['documents']
        self._all_metadatas = all_data.get('metadatas') or [{} for _ in self._all_docs]
        self._all_ids = all_data['ids']
        if not self._all_docs:
            self._bm25 = None
            return
        # 中文分词
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self._all_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query, top_k=3, k_rrf=60):
        """
        混合检索：向量检索 + BM25 关键词检索，RRF 融合
        :param query: 查询字符串
        :param top_k: 最终返回的文档数量
        :param k_rrf: RRF 平滑参数（默认 60）
        :return: 文档列表
        """
        # 1. 向量检索（多取一些，便于融合）
        vec_records = self.search_records(query, top_k=top_k * 3)
        vec_results = [record["text"] for record in vec_records]

        # 2. BM25 检索
        self._ensure_bm25_index()
        bm25_results = []
        if self._bm25 is not None and self._all_docs:
            query_tokens = list(jieba.cut(query))
            scores = self._bm25.get_scores(query_tokens)
            # 取 top_k*3 个
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]
            bm25_results = [self._all_docs[i] for i in top_indices]

        # 3. RRF 融合
        combined = {}
        # 向量检索排名
        for rank, doc in enumerate(vec_results):
            combined[doc] = combined.get(doc, 0) + 1 / (k_rrf + rank + 1)
        # BM25 检索排名
        for rank, doc in enumerate(bm25_results):
            combined[doc] = combined.get(doc, 0) + 1 / (k_rrf + rank + 1)

        # 按总分降序排列，取 top_k
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc for doc, _ in ranked]

    def hybrid_search_records(self, query, top_k=3, k_rrf=60):
        """混合检索并保留文档 ID 和来源 metadata。"""
        vec_records = self.search_records(query, top_k=top_k * 3)
        self._ensure_bm25_index()
        combined = {}

        for rank, record in enumerate(vec_records):
            combined[record["id"]] = [record, 1 / (k_rrf + rank + 1)]

        if self._bm25 is not None and self._all_docs:
            query_tokens = list(jieba.cut(query))
            scores = self._bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]
            for rank, index in enumerate(top_indices):
                doc_id = self._all_ids[index]
                record = {
                    "id": doc_id,
                    "text": self._all_docs[index],
                    "metadata": self._all_metadatas[index] or {},
                }
                if doc_id in combined:
                    combined[doc_id][1] += 1 / (k_rrf + rank + 1)
                else:
                    combined[doc_id] = [record, 1 / (k_rrf + rank + 1)]

        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        return [record for record, _ in ranked[:top_k]]

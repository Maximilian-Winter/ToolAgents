# bm25_vector_database.py
from __future__ import annotations
import re
import math
from typing import Any, List, Optional, Dict, Tuple

from ToolAgents.knowledge.vector_database.vector_database_provider import (
    VectorDatabaseProvider,
    VectorSearchResult,
    VectorCollection,
)
from ToolAgents.knowledge.vector_database.embedding_provider import EmbeddingProvider
from ToolAgents.knowledge.vector_database.reranking_provider import RerankingProvider

# Minimal, dependency-free BM25Okapi-style scorer.
# (If you prefer, you can swap this out for rank-bm25 with minor changes.)
class _BM25:
    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(d) for d in tokenized_docs) / self.N if self.N else 0.0

        self.df = {}
        self.term_freqs = []
        for d in tokenized_docs:
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1

        # idf with add-one smoothing to avoid negatives on very common terms
        self.idf = {t: math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0) for t in self.df}

    def score(self, tokenized_query: List[str], idx: int) -> float:
        tf = self.term_freqs[idx]
        dl = len(self.docs[idx]) or 1
        score = 0.0
        for q in tokenized_query:
            if q not in tf:
                continue
            freq = tf[q]
            idf = self.idf.get(q, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return score

    def get_scores(self, tokenized_query: List[str]) -> List[float]:
        return [self.score(tokenized_query, i) for i in range(self.N)]

class BM25VectorDatabaseProvider(VectorDatabaseProvider):
    """
    A sparse keyword retriever that conforms to your VectorDatabaseProvider API.
    Uses in-memory BM25. RerankingProvider is optional and applied post-retrieval.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,         # not used; kept for API consistency
        reranking_provider: Optional[RerankingProvider] = None,
        tokenizer: str = r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+",
        lowercase: bool = True,
    ):
        super().__init__(embedding_provider, reranking_provider)
        self._token_re = re.compile(tokenizer)
        self._lower = lowercase

        self._ids: List[str] = []
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._tokens: List[List[str]] = []
        self._bm25: Optional[_BM25] = None
        self._collection_name = "bm25_default"

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower() if self._lower else text
        return self._token_re.findall(text)

    def _rebuild(self):
        self._tokens = [self._tokenize(d) for d in self._docs]
        self._bm25 = _BM25(self._tokens) if self._docs else None

    # --- VectorDatabaseProvider API ---
    def add_documents(self, documents) -> None:
        # If you need chunk-aware ingestion, map like Chroma does.
        # Minimal path: treat each document as-is with empty metadata.
        texts = []
        metas = []
        ids = []
        for document in documents:
            for chunk in document.document_chunks:
                ids.append(chunk.id)
                texts.append(chunk.content)
                meta = {} if document.metadata is None else dict(document.metadata)
                meta["parent_doc_id"] = chunk.parent_doc_id
                meta["chunk_index"] = chunk.chunk_index
                metas.append(meta)
        self.add_texts_with_id(ids, texts, metas)

    def add_texts(self, texts: List[str], metadata: List[dict]) -> None:
        ids = [self.generate_unique_id() for _ in texts]
        self.add_texts_with_id(ids, texts, metadata or [{} for _ in texts])

    def add_texts_with_id(self, ids: List[str], texts: List[str], metadata: List[dict]) -> None:
        self._ids.extend(ids)
        self._docs.extend(texts)
        self._metas.extend(metadata or [{} for _ in texts])
        self._rebuild()

    def query(self, query: str, query_filter: Any = None, k: int = 3, **kwargs) -> VectorSearchResult:
        if not self._docs:
            return VectorSearchResult([], [], [], [])

        toks = self._tokenize(query)
        scores = self._bm25.get_scores(toks) if self._bm25 else [0.0] * len(self._docs)

        # Optional filtering on metadata (simple AND over dict equality / subset)
        candidates: List[Tuple[int, float]] = []
        for i, s in enumerate(scores):
            if query_filter:
                meta = self._metas[i] or {}
                # simple subset check: all key/values in filter must match meta
                ok = all(meta.get(k2) == v2 for k2, v2 in query_filter.items())
                if not ok:
                    continue
            candidates.append((i, s))

        # top-n by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[: max(k * 4, k)]

        docs = [self._docs[i] for i, _ in top]
        ids = [self._ids[i] for i, _ in top]
        metas = [self._metas[i] for i, _ in top]
        top_scores = [s for _, s in top]

        # optional cross-encoder rerank on text
        if self.reranking_provider is not None and docs:
            rr = self.reranking_provider.rerank_texts(query, docs, k=k, return_documents=True)
            # map reranked content back to IDs/scores
            text_to_idx = {self._docs[i]: i for i, _ in top}
            reranked_ids = []
            reranked_docs = []
            reranked_scores = []
            reranked_metas = []
            for rd in rr.reranked_documents:
                i = text_to_idx.get(rd.content)
                if i is None:
                    continue
                reranked_ids.append(self._ids[i])
                reranked_docs.append(self._docs[i])
                reranked_metas.append(self._metas[i])
                reranked_scores.append(rd.additional_data.get("score", 0.0))
            return VectorSearchResult(reranked_ids, reranked_docs, reranked_scores, reranked_metas)

        # fall back (no reranker)
        return VectorSearchResult(ids[:k], docs[:k], top_scores[:k], metas[:k])

    def create_or_set_current_collection(self, collection_name: str) -> None:
        self._collection_name = collection_name
        # in-memory only; you could maintain a dict of collections if needed

    def remove_by_ids(self, ids: List[str]) -> None:
        keep = [i for i, _id in enumerate(self._ids) if _id not in ids]
        self._ids   = [self._ids[i] for i in keep]
        self._docs  = [self._docs[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        self._rebuild()

    def get_all_entries(self) -> VectorCollection:
        # embeddings not applicable; return empty list to satisfy type
        return VectorCollection(ids=self._ids, chunks=self._docs, embeddings=[], metadata=self._metas)

    def delete_collection(self, collection_name: str) -> None:
        if collection_name == self._collection_name:
            self._ids, self._docs, self._metas, self._tokens = [], [], [], []
            self._bm25 = None

    def update_metadata(self, ids: List[str], metadata: List[Dict[str, Any]]) -> None:
        idx_by_id = {id_: i for i, id_ in enumerate(self._ids)}
        for id_, meta in zip(ids, metadata):
            i = idx_by_id.get(id_)
            if i is not None:
                self._metas[i] = meta

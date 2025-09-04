# ensemble_vector_database.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math

from ToolAgents.knowledge.vector_database.vector_database_provider import (
    VectorDatabaseProvider,
    VectorSearchResult,
    VectorCollection,
)
from ToolAgents.knowledge.vector_database.embedding_provider import EmbeddingProvider
from ToolAgents.knowledge.vector_database.reranking_provider import RerankingProvider

def _min_max_norm(scores: List[float], higher_is_better: bool) -> List[float]:
    if not scores:
        return []
    # For distance metrics (e.g., Chroma distances), smaller is better.
    # Convert to "higher is better" before min-max.
    s = scores
    if not higher_is_better:
        # invert: larger is better -> use negative, or 1/x for >0
        # Use simple negation assuming distances >= 0
        s = [-x for x in scores]
    mn, mx = min(s), max(s)
    if math.isclose(mx, mn):
        return [0.5] * len(s)
    return [(v - mn) / (mx - mn) for v in s]

class EnsembleVectorDatabaseProvider(VectorDatabaseProvider):
    """
    Query two providers (e.g., BM25 + Chroma), fuse scores, optionally rerank with cross-encoder.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,                 # not used directly; API compat
        dense_provider: VectorDatabaseProvider,                # e.g., ChromaDbVectorDatabaseProvider
        sparse_provider: VectorDatabaseProvider,               # e.g., BM25VectorDatabaseProvider
        reranking_provider: Optional[RerankingProvider] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        dense_scores_are_similarities: bool = False,          # Chroma returns distances by default
        shortlist_multiplier: int = 4,                         # how many to gather before final k
    ):
        super().__init__(embedding_provider, reranking_provider)
        self.dense = dense_provider
        self.sparse = sparse_provider
        self.dw = dense_weight
        self.sw = sparse_weight
        self.dense_scores_are_similarities = dense_scores_are_similarities
        self.shortlist_multiplier = shortlist_multiplier

    # --- delegate writes to both stores (so either path can recall) ---
    def add_documents(self, documents) -> None:
        self.dense.add_documents(documents)
        self.sparse.add_documents(documents)

    def add_texts(self, texts: List[str], metadata: List[dict]) -> None:
        self.dense.add_texts(texts, metadata)
        self.sparse.add_texts(texts, metadata)

    def add_texts_with_id(self, ids: List[str], texts: List[str], metadata: List[dict]) -> None:
        self.dense.add_texts_with_id(ids, texts, metadata)
        self.sparse.add_texts_with_id(ids, texts, metadata)

    # --- reads ---
    def query(self, query: str, query_filter: Any = None, k: int = 3, **kwargs) -> VectorSearchResult:
        n = k * self.shortlist_multiplier

        d = self.dense.query(query, query_filter=query_filter, k=n, **kwargs)
        s = self.sparse.query(query, query_filter=query_filter, k=n, **kwargs)

        # normalize to [0,1]; convert dense to similarity if needed
        d_norm = _min_max_norm(d.scores, higher_is_better=self.dense_scores_are_similarities)
        s_norm = _min_max_norm(s.scores, higher_is_better=True)

        # fuse by document string; prefer IDs if you guarantee uniqueness across stores
        items: Dict[str, Dict[str, Any]] = {}
        for src, ids, docs, metas, norm_scores, raw_scores in [
            ("dense", d.ids, d.chunks, getattr(d, "metadata", None) or [{}]*len(d.ids), d_norm, d.scores),
            ("sparse", s.ids, s.chunks, getattr(s, "metadata", None) or [{}]*len(s.ids), s_norm, s.scores),
        ]:
            for i, (id_, doc, meta) in enumerate(zip(ids, docs, metas)):
                rec = items.setdefault(doc, {"ids": set(), "meta": meta, "dense": None, "sparse": None})
                rec["ids"].add(id_)
                if src == "dense":
                    rec["dense"] = norm_scores[i] if i < len(norm_scores) else 0.0
                else:
                    rec["sparse"] = norm_scores[i] if i < len(norm_scores) else 0.0

        fused: List[Tuple[str, float, Dict[str, Any], List[str]]] = []
        for doc, rec in items.items():
            ds = rec["dense"] if rec["dense"] is not None else 0.0
            ss = rec["sparse"] if rec["sparse"] is not None else 0.0
            fused_score = self.dw * ds + self.sw * ss
            fused.append((doc, fused_score, rec["meta"], list(rec["ids"])))

        fused.sort(key=lambda x: x[1], reverse=True)
        docs = [f[0] for f in fused[:n]]
        metas = [f[2] for f in fused[:n]]
        ids = ["/".join(f[3]) for f in fused[:n]]  # merged IDs

        # optional cross-encoder rerank on fused shortlist
        if self.reranking_provider is not None and docs:
            rr = self.reranking_provider.rerank_texts(query, docs, k=k, return_documents=True)
            text_to_idx = {doc: i for i, doc in enumerate(docs)}
            out_ids, out_docs, out_scores, out_metas = [], [], [], []
            for rd in rr.reranked_documents:
                i = text_to_idx.get(rd.content)
                if i is None:
                    continue
                out_ids.append(ids[i])
                out_docs.append(docs[i])
                out_metas.append(metas[i])
                out_scores.append(rd.additional_data.get("score", 0.0))
            return VectorSearchResult(out_ids, out_docs, out_scores, out_metas)

        # no reranker path
        out = fused[:k]
        return VectorSearchResult(
            ["/".join(f[3]) for f in out],
            [f[0] for f in out],
            [f[1] for f in out],
            [f[2] for f in out],
        )

    # --- misc / plumbing ---
    def create_or_set_current_collection(self, collection_name: str) -> None:
        self.dense.create_or_set_current_collection(collection_name)
        self.sparse.create_or_set_current_collection(collection_name)

    def remove_by_ids(self, ids: List[str]) -> None:
        self.dense.remove_by_ids(ids)
        self.sparse.remove_by_ids(ids)

    def get_all_entries(self) -> VectorCollection:
        # Return dense entries for convenience (embeddings present there)
        return self.dense.get_all_entries()

    def delete_collection(self, collection_name: str) -> None:
        self.dense.delete_collection(collection_name)
        self.sparse.delete_collection(collection_name)

    def update_metadata(self, ids: List[str], metadata: List[Dict[str, Any]]) -> None:
        self.dense.update_metadata(ids, metadata)
        self.sparse.update_metadata(ids, metadata)

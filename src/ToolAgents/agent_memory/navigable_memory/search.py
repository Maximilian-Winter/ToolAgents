"""Search operations for NavigableMemory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import Document


class NavigableSearchMixin:
    def list_at(self, prefix: str = "") -> List[Document]:
        """List documents under a path prefix."""
        return self.backend.list(prefix)

    def search(self, query: str) -> List[Document]:
        """Search across all documents."""
        return self.backend.search(query)

    def semantic_search(
        self,
        query: str,
        k: int = 8,
        *,
        path_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Any]:
        """Search the optional semantic index.

        Returns path-aware semantic search result objects. If no semantic index
        is attached, returns an empty list.
        """
        if not self._has_semantic_support():
            return []
        return self.semantic_index.search(  # type: ignore[union-attr]
            query, k=k, path_prefix=path_prefix, tags=tags,
        )

    def build_semantic_search_context(
        self,
        query: str,
        k: int = 5,
        *,
        path_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_chars: int = 320,
    ) -> str:
        """Render semantic search hits as a prompt context block."""
        if not self._has_semantic_support():
            return "## Semantic search\nNo semantic index is attached."
        return self.semantic_index.build_search_context(  # type: ignore[union-attr]
            query, k=k, path_prefix=path_prefix, tags=tags, max_chars=max_chars,
        )

    def hybrid_search(
        self,
        query: str,
        k: int = 8,
        *,
        path_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_references: bool = True,
    ) -> List[Any]:
        """Combine semantic, lexical, tag, and nearby reference signals."""
        from .semantic_index import NavigableSearchResult

        candidates: Dict[str, Dict[str, Any]] = {}
        semantic_hits = self.semantic_search(
            query, k=max(k * 3, k), path_prefix=path_prefix, tags=tags,
        )
        semantic_scores = self._normalize_scores([h.score for h in semantic_hits])
        for hit, score in zip(semantic_hits, semantic_scores):
            self._merge_search_candidate(
                candidates, hit, 0.70 * score, source="semantic",
            )

        lexical_docs = [
            doc for doc in self.search(query)
            if self._doc_matches_search_filters(doc, path_prefix, tags)
        ]
        lexical_scores = self._normalize_scores(
            [1.0 / (idx + 1) for idx, _ in enumerate(lexical_docs)]
        )
        for doc, score in zip(lexical_docs, lexical_scores):
            hit = NavigableSearchResult(
                path=doc.path, title=doc.title, chunk=doc.content,
                score=score, version=doc.version, tags=list(doc.tags),
                mime_type=doc.mime_type, updated_at=doc.updated_at,
            )
            self._merge_search_candidate(
                candidates, hit, 0.25 * score, source="lexical",
            )

        if tags:
            for doc in self.find_by_tags(tags, mode="all"):
                if not self._doc_matches_search_filters(doc, path_prefix, tags):
                    continue
                hit = NavigableSearchResult(
                    path=doc.path, title=doc.title, chunk=doc.content,
                    score=1.0, version=doc.version, tags=list(doc.tags),
                    mime_type=doc.mime_type, updated_at=doc.updated_at,
                )
                self._merge_search_candidate(
                    candidates, hit, 0.10, source="tags",
                )

        if include_references and self.location.current_path and self._has_references_support():
            ref_paths = {
                ref.to_path for ref in self.references_from(self.location.current_path)
            }
            ref_paths.update(
                ref.from_path for ref in self.references_to(self.location.current_path)
            )
            for path in ref_paths:
                doc = self.read(path)
                if doc is None or not self._doc_matches_search_filters(
                    doc, path_prefix, tags,
                ):
                    continue
                hit = NavigableSearchResult(
                    path=doc.path, title=doc.title, chunk=doc.content,
                    score=1.0, version=doc.version, tags=list(doc.tags),
                    mime_type=doc.mime_type, updated_at=doc.updated_at,
                )
                self._merge_search_candidate(
                    candidates, hit, 0.10, source="reference",
                )

        ranked: List[NavigableSearchResult] = []
        for data in candidates.values():
            hit = data["hit"]
            ranked.append(NavigableSearchResult(
                path=hit.path, title=hit.title, chunk=hit.chunk,
                score=data["score"], chunk_index=hit.chunk_index,
                version=hit.version, tags=list(hit.tags),
                mime_type=hit.mime_type, updated_at=hit.updated_at,
                raw_score=hit.raw_score,
                metadata={**hit.metadata, "sources": sorted(data["sources"])},
            ))
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:k]

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        low = min(scores)
        high = max(scores)
        if high == low:
            return [1.0 for _ in scores]
        return [(score - low) / (high - low) for score in scores]

    @staticmethod
    def _merge_search_candidate(
        candidates: Dict[str, Dict[str, Any]],
        hit: Any,
        score: float,
        *,
        source: str,
    ) -> None:
        data = candidates.setdefault(
            hit.path, {"hit": hit, "score": 0.0, "sources": set()}
        )
        data["score"] += score
        data["sources"].add(source)
        if getattr(hit, "score", 0.0) > getattr(data["hit"], "score", 0.0):
            data["hit"] = hit

    @staticmethod
    def _doc_matches_search_filters(
        doc: Document,
        path_prefix: Optional[str],
        tags: Optional[List[str]],
    ) -> bool:
        if path_prefix and not doc.path.startswith(path_prefix):
            return False
        if tags and not set(tags).issubset(set(doc.tags)):
            return False
        return True

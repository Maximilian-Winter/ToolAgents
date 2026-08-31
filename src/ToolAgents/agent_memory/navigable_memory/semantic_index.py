"""Optional semantic retrieval index for NavigableMemory.

This module intentionally uses duck typing for vector stores so importing
``ToolAgents.agent_memory.navigable_memory`` does not require optional vector
database, embedding, or reranker dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ToolAgents.knowledge.document import DocumentGenerator
from ToolAgents.knowledge.text_processing.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

if TYPE_CHECKING:
    from .models import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NavigableSearchResult:
    """A path-aware semantic search hit."""

    path: str
    title: str
    chunk: str
    score: float
    chunk_index: int = 0
    version: int = 1
    tags: List[str] = field(default_factory=list)
    mime_type: str = "text/markdown"
    updated_at: Optional[str] = None
    raw_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snippet(self, max_chars: int = 240) -> str:
        text = " ".join(self.chunk.split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."


class NavigableSemanticIndex:
    """Secondary semantic index for documents stored in NavigableMemory.

    Args:
        vector_database_provider: Object implementing the existing
            VectorDatabaseProvider-style methods used here:
            ``add_texts_with_id``, ``query``, ``remove_by_ids``, and
            ``get_all_entries``.
        text_splitter: Optional splitter. A conservative recursive character
            splitter is used by default.
        include_binary_captions: If true, binary document captions/content are
            indexed. Raw binary bytes are never embedded.
        scores_are_distances: Set true for providers such as Chroma where lower
            query scores are better distances. Results exposed by this adapter
            are normalized so higher is better.
        namespace: Metadata marker used to distinguish this index's entries
            from other entries in a shared vector collection.
    """

    def __init__(
        self,
        vector_database_provider: Any,
        *,
        text_splitter: Optional[TextSplitter] = None,
        include_binary_captions: bool = False,
        scores_are_distances: bool = False,
        query_multiplier: int = 4,
        namespace: str = "navigable_memory",
    ):
        self.vector_database_provider = vector_database_provider
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            chunk_size=800,
            chunk_overlap=120,
            keep_separator=True,
        )
        self.include_binary_captions = include_binary_captions
        self.scores_are_distances = scores_are_distances
        self.query_multiplier = max(1, query_multiplier)
        self.namespace = namespace
        self.memory: Any = None
        self._indexed_ids: Dict[str, List[str]] = {}

    def attach_memory(self, memory: Any) -> "NavigableSemanticIndex":
        self.memory = memory
        return self

    def rebuild(self, memory: Any = None, prefix: str = "") -> int:
        """Rebuild index entries for every document under ``prefix``."""
        memory = self._resolve_memory(memory)
        self._remove_entries(prefix=prefix)
        indexed = 0
        for doc in memory.list_at(prefix):
            if self.index_document(doc.path, memory=memory):
                indexed += 1
        return indexed

    def index_document(self, path: str, memory: Any = None) -> bool:
        """Index the current version of one document by path."""
        memory = self._resolve_memory(memory)
        doc = memory.read(path)
        self.remove_document(path)
        if doc is None or not self._should_index(doc):
            return False

        generated = DocumentGenerator(self.text_splitter).generate_document(
            doc.content,
            metadata=self._base_metadata(doc),
        )
        chunks = [chunk for chunk in generated.document_chunks if chunk.content.strip()]
        if not chunks:
            return False

        ids: List[str] = []
        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = self._chunk_id(doc.path, doc.version, idx)
            meta = dict(generated.metadata or {})
            meta.update({
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "chunk_size": chunk.size_in_characters,
            })
            ids.append(chunk_id)
            texts.append(chunk.content)
            metadata.append(meta)

        self.vector_database_provider.add_texts_with_id(ids, texts, metadata)
        self._indexed_ids[doc.path] = ids
        return True

    def remove_document(self, path: str) -> int:
        """Remove all indexed chunks for one document path."""
        return self._remove_entries(path=path)

    def search(
        self,
        query: str,
        k: int = 8,
        *,
        path_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[NavigableSearchResult]:
        """Search indexed chunks and return path-aware results."""
        requested = max(k * self.query_multiplier, k)
        result = self.vector_database_provider.query(query, k=requested)
        hits = self._coerce_results(result)
        filtered = [
            hit for hit in hits
            if self._matches_filters(hit, path_prefix=path_prefix, tags=tags)
        ]
        return filtered[:k]

    def build_search_context(
        self,
        query: str,
        k: int = 5,
        *,
        path_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_chars: int = 320,
    ) -> str:
        """Render semantic results as a compact prompt context block."""
        results = self.search(query, k=k, path_prefix=path_prefix, tags=tags)
        if not results:
            return f"## Semantic search\nNo semantic results for '{query}'."

        lines = [f"## Semantic search results for '{query}'"]
        for idx, hit in enumerate(results, start=1):
            tag_text = f" tags={hit.tags}" if hit.tags else ""
            lines.append(
                f"{idx}. {hit.title} ({hit.path}) "
                f"score={hit.score:.3f} v{hit.version} chunk={hit.chunk_index}{tag_text}"
            )
            lines.append(f"   {hit.snippet(max_chars)}")
        return "\n".join(lines)

    def _resolve_memory(self, memory: Any = None) -> Any:
        resolved = memory or self.memory
        if resolved is None:
            raise ValueError("NavigableSemanticIndex is not attached to a NavigableMemory.")
        return resolved

    def _should_index(self, doc: Document) -> bool:
        if doc.is_binary and not self.include_binary_captions:
            return False
        return bool(doc.content and doc.content.strip())

    def _base_metadata(self, doc: Document) -> Dict[str, Any]:
        tags = list(doc.tags or [])
        return {
            "navmem_index_namespace": self.namespace,
            "path": doc.path,
            "title": doc.title,
            "version": doc.version,
            "tags": ",".join(tags),
            "tags_json": json.dumps(tags, ensure_ascii=False),
            "mime_type": doc.mime_type,
            "updated_at": doc.updated_at or "",
        }

    def _coerce_results(self, result: Any) -> List[NavigableSearchResult]:
        ids = list(getattr(result, "ids", []) or [])
        chunks = list(getattr(result, "chunks", []) or [])
        scores = list(getattr(result, "scores", []) or [])
        metadata = list(getattr(result, "metadata", []) or [{} for _ in chunks])
        hits: List[NavigableSearchResult] = []
        for idx, chunk in enumerate(chunks):
            meta = dict(metadata[idx] or {}) if idx < len(metadata) else {}
            if meta.get("navmem_index_namespace") != self.namespace:
                continue
            fallback_id = ids[idx] if idx < len(ids) else ""
            raw_score = float(scores[idx]) if idx < len(scores) else 0.0
            score = self._normalize_provider_score(raw_score)
            hits.append(NavigableSearchResult(
                path=str(meta.get("path") or ""),
                title=str(meta.get("title") or meta.get("path") or fallback_id),
                chunk=str(chunk),
                score=score,
                raw_score=raw_score,
                chunk_index=int(meta.get("chunk_index") or 0),
                version=int(meta.get("version") or 1),
                tags=self._parse_tags(meta),
                mime_type=str(meta.get("mime_type") or "text/markdown"),
                updated_at=str(meta.get("updated_at") or "") or None,
                metadata=meta,
            ))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def _normalize_provider_score(self, score: float) -> float:
        if self.scores_are_distances:
            return 1.0 / (1.0 + max(score, 0.0))
        return score

    def _matches_filters(
        self,
        hit: NavigableSearchResult,
        *,
        path_prefix: Optional[str],
        tags: Optional[List[str]],
    ) -> bool:
        if path_prefix and not hit.path.startswith(path_prefix):
            return False
        if tags and not set(tags).issubset(set(hit.tags)):
            return False
        return True

    def _remove_entries(
        self,
        *,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> int:
        ids = self._entry_ids(path=path, prefix=prefix)
        if not ids:
            return 0
        self.vector_database_provider.remove_by_ids(ids)
        if path:
            self._indexed_ids.pop(path, None)
        if prefix is not None:
            for known_path in list(self._indexed_ids):
                if known_path.startswith(prefix):
                    self._indexed_ids.pop(known_path, None)
        return len(ids)

    def _entry_ids(
        self,
        *,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> List[str]:
        ids: List[str] = []
        if path:
            ids.extend(self._indexed_ids.get(path, []))

        try:
            collection = self.vector_database_provider.get_all_entries()
        except Exception as exc:
            logger.debug("Could not inspect semantic index entries: %s", exc)
            return list(dict.fromkeys(ids))

        collection_ids = list(getattr(collection, "ids", []) or [])
        metadata = list(getattr(collection, "metadata", []) or [])
        for entry_id, meta in zip(collection_ids, metadata):
            meta = meta or {}
            if meta.get("navmem_index_namespace") != self.namespace:
                continue
            entry_path = str(meta.get("path") or "")
            if path and entry_path == path:
                ids.append(entry_id)
            elif prefix is not None and entry_path.startswith(prefix):
                ids.append(entry_id)
        return list(dict.fromkeys(ids))

    def _chunk_id(self, path: str, version: int, chunk_index: int) -> str:
        digest = hashlib.sha1(
            f"{self.namespace}|{path}".encode("utf-8")
        ).hexdigest()[:20]
        return f"navmem:{digest}:v{version}:c{chunk_index}"

    @staticmethod
    def _parse_tags(meta: Dict[str, Any]) -> List[str]:
        raw = meta.get("tags_json")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(tag) for tag in parsed]
            except json.JSONDecodeError:
                pass
        tags = meta.get("tags")
        if isinstance(tags, str):
            return [tag for tag in tags.split(",") if tag]
        if isinstance(tags, Iterable):
            return [str(tag) for tag in tags]
        return []

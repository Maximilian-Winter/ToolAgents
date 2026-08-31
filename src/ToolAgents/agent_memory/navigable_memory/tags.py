"""Tag operations for NavigableMemory."""

from __future__ import annotations

from typing import List

from .models import Document
from .protocols import TagStorage


class NavigableTagMixin:
    def _has_tag_support(self) -> bool:
        return isinstance(self.backend, TagStorage)

    def list_tags(self) -> List[str]:
        """List every unique tag in the knowledge base, sorted."""
        if self._has_tag_support():
            return self.backend.list_tags()  # type: ignore[attr-defined]
        # Fallback for backends without TagStorage: scan
        all_tags: set = set()
        for doc in self.backend.list(""):
            for t in doc.tags:
                if t:
                    all_tags.add(t)
        return sorted(all_tags)

    def list_by_tag(self, tag: str) -> List[Document]:
        """List documents that carry the given tag."""
        if self._has_tag_support():
            return self.backend.list_by_tag(tag)  # type: ignore[attr-defined]
        return [d for d in self.backend.list("") if tag in d.tags]

    def find_by_tags(self, tags: List[str], mode: str = "any") -> List[Document]:
        """Find documents matching a set of tags.

        Args:
            tags: Tags to match.
            mode: 'any' (OR), 'all' (AND), or 'none' (exclusion).
        """
        if self._has_tag_support():
            return self.backend.find_by_tags(tags, mode)  # type: ignore[attr-defined]
        # Fallback
        if not tags:
            return []
        target = set(tags)
        results: List[Document] = []
        for doc in self.backend.list(""):
            doc_tags = set(doc.tags)
            if mode == "all" and target.issubset(doc_tags):
                results.append(doc)
            elif mode == "none" and target.isdisjoint(doc_tags):
                results.append(doc)
            elif mode == "any" and target & doc_tags:
                results.append(doc)
        return results

    def set_tags(self, path: str, tags: List[str]) -> bool:
        """Replace a document's tag list. Does not bump version."""
        if self._has_tag_support():
            ok = self.backend.set_tags(path, list(tags))  # type: ignore[attr-defined]
            if ok:
                self._refresh_semantic_index(path)
            return ok
        # Fallback: read + rewrite (this DOES bump version on most backends)
        doc = self.backend.read(path)
        if doc is None:
            return False
        ok = self.backend.write(
            path, doc.title, doc.content, list(tags), dict(doc.metadata),
        )
        if ok:
            self._refresh_semantic_index(path)
        return ok

    def add_tags(self, path: str, *new_tags: str) -> bool:
        """Add tags to a document, preserving existing ones (set semantics)."""
        doc = self.backend.read(path)
        if doc is None:
            return False
        merged = list(dict.fromkeys(list(doc.tags) + list(new_tags)))  # ordered dedup
        return self.set_tags(path, merged)

    def remove_tags(self, path: str, *tags_to_remove: str) -> bool:
        """Remove specific tags from a document."""
        doc = self.backend.read(path)
        if doc is None:
            return False
        drop = set(tags_to_remove)
        kept = [t for t in doc.tags if t not in drop]
        return self.set_tags(path, kept)

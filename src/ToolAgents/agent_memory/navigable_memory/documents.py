"""Document read/write operations for NavigableMemory."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Document
from .state import LocationState

logger = logging.getLogger(__name__)


class NavigableDocumentMixin:
    def read(self, path: str) -> Optional[Document]:
        """Read a document without navigating to it."""
        return self.backend.read(path)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              author: str = "", change_note: str = "") -> bool:
        """Write a textual document to storage.

        If the backend supports versioning, the previous content is
        preserved as a historical version automatically.
        """
        # Try the extended signature first (with author/change_note)
        try:
            ok = self.backend.write(  # type: ignore[call-arg]
                path, title, content, tags, metadata,
                author=author, change_note=change_note,
            )
        except TypeError:
            ok = self.backend.write(path, title, content, tags, metadata)
        if ok:
            self._refresh_semantic_index(path)
        return ok

    def append(self, path: str, content: str) -> str:
        """Append content to an existing document.

        Useful for event logs, notes, and accumulating observations.
        """
        doc = self.backend.read(path)
        if doc is None:
            return f"Document not found: '{path}'"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated = doc.content + f"\n\n### Log — {timestamp}\n{content}"
        self.write(path, doc.title, updated, doc.tags,
                   change_note=f"append @ {timestamp}")

        # If we're currently AT this document, refresh the content
        if self.location.current_path == path:
            self.location.current_content = updated

        return f"Appended to '{doc.title}'."

    # ── Binary Operations ─────────────────────────────────────────

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     author: str = "", change_note: str = "") -> bool:
        """Store a binary blob (image, audio, PDF, etc.).

        Args:
            path: Document path (e.g., 'assets/diagrams/arch.png').
            title: Human-readable title.
            mime_type: MIME type (e.g., 'image/png', 'audio/mpeg').
            data: The raw bytes to store.
            caption: Optional text caption used in context display.
            tags, metadata: Same as text documents.
            author, change_note: Versioning context.

        Raises:
            NotImplementedError: If the backend does not support binary storage.
        """
        if not self._has_binary_support():
            raise NotImplementedError(
                f"{type(self.backend).__name__} does not implement BinaryStorage."
            )
        try:
            ok = self.backend.write_binary(  # type: ignore[call-arg]
                path, title, mime_type, data, caption,
                tags, metadata, author=author, change_note=change_note,
            )
        except TypeError:
            ok = self.backend.write_binary(  # type: ignore[attr-defined]
                path, title, mime_type, data, caption, tags, metadata,
            )
        if ok:
            self._refresh_semantic_index(path)
        return ok

    def read_binary(self, path: str) -> Optional[bytes]:
        """Return the raw bytes of a binary document, or None if not found."""
        if not self._has_binary_support():
            return None
        return self.backend.read_binary(path)  # type: ignore[attr-defined]

    def delete(self, path: str) -> bool:
        """Delete a document and remove any attached semantic index entries."""
        ok = self.backend.delete(path)
        if ok:
            self._remove_semantic_index(path)
            if self.location.current_path == path:
                self.location = LocationState()
        return ok

    def _refresh_semantic_index(self, path: str) -> None:
        if not self._has_semantic_support():
            return
        try:
            self.semantic_index.index_document(path, memory=self)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Semantic index refresh failed for '%s': %s", path, exc)

    def _remove_semantic_index(self, path: str) -> None:
        if not self._has_semantic_support():
            return
        try:
            self.semantic_index.remove_document(path)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Semantic index removal failed for '%s': %s", path, exc)

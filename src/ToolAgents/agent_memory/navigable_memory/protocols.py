"""Storage protocols for NavigableMemory backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import Document, DocumentVersion, Reference, RefType


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for knowledge storage backends.

    Implement this to plug SQLite knowledge base,
    filesystem, or any other document store.
    """

    def read(self, path: str) -> Optional[Document]:
        """Read a document by its full path."""
        ...

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write or update a document. Returns True on success."""
        ...

    def list(self, prefix: str = "") -> List[Document]:
        """List documents under a path prefix."""
        ...

    def search(self, query: str) -> List[Document]:
        """Full-text search across all documents."""
        ...

    def delete(self, path: str) -> bool:
        """Delete a document. Returns True if it existed."""
        ...


@runtime_checkable
class BinaryStorage(Protocol):
    """Optional protocol for backends that support binary blobs.

    Backends implementing this can store images, audio, PDFs, and other
    non-text data. NavigableMemory detects support via isinstance().
    """

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write a binary document. Returns True on success."""
        ...

    def read_binary(self, path: str) -> Optional[bytes]:
        """Return the raw bytes of a binary document, or None."""
        ...


@runtime_checkable
class VersionedStorage(Protocol):
    """Optional protocol for backends that retain version history."""

    def list_versions(self, path: str) -> List[DocumentVersion]:
        """List all versions of a document, newest first."""
        ...

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        """Read a specific historical version."""
        ...

    def rollback(self, path: str, version: int, author: str = "",
                 change_note: str = "") -> bool:
        """Restore a document to a previous version (creates new version)."""
        ...

    def prune_versions(self, path: str, keep_last_n: int) -> int:
        """Drop old versions, keeping the most recent N. Returns count removed."""
        ...


@runtime_checkable
class ReferenceStorage(Protocol):
    """Optional protocol for backends that track inter-document references."""

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        """Create a reference. Idempotent (same triple = no duplicate)."""
        ...

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        """Remove matching references. Returns count removed."""
        ...

    def list_references_from(self, path: str) -> List[Reference]:
        """List outgoing references from a document."""
        ...

    def list_references_to(self, path: str) -> List[Reference]:
        """List incoming references to a document."""
        ...

    def list_all_references(self) -> List[Reference]:
        """List every reference in the store (used for migration / inspection)."""
        ...


@runtime_checkable
class TagStorage(Protocol):
    """Optional protocol for backends that support efficient tag queries.

    Backends without this protocol can still store tags on documents
    (the ``Document.tags`` field is always available) — this protocol
    exposes specialized operations that are faster than scanning all
    documents in NavigableMemory.
    """

    def list_tags(self) -> List[str]:
        """Return every unique tag used across all documents, sorted."""
        ...

    def list_by_tag(self, tag: str) -> List[Document]:
        """List documents that have a specific tag."""
        ...

    def find_by_tags(self, tags: List[str], mode: str = "any") -> List[Document]:
        """Find documents matching a tag set.

        Args:
            tags: Tags to match.
            mode: 'any' (OR), 'all' (AND), or 'none' (exclusion).
        """
        ...

    def set_tags(self, path: str, tags: List[str]) -> bool:
        """Replace a document's tag list in place.

        Tag updates do NOT create a new version — tags are organizational
        metadata, not content. Updates ``updated_at`` only.
        """
        ...

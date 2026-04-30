"""
導航記憶 — Navigable Memory

A generalized location-based memory system where context loads
automatically based on the agent's current position in a knowledge space.

Core insight: don't make the LLM search — make the LLM NAVIGATE.
When the agent moves to a new location, relevant context loads
automatically. When the agent departs, what happened gets written back.

Architecture:
    NavigableMemory
      ├── StorageBackend (read/write/list/search by path)
      ├── LocationState (current position + history)
      ├── PromptComposer integration (content_fn for auto-loading)
      └── SmartMessageManager integration (TTL-based location history)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Storage Backend Protocol
# ═══════════════════════════════════════════════════════════════════

# Common reference types — string constants, also extensible
class RefType:
    LINKS_TO = "links_to"
    DEPENDS_ON = "depends_on"
    SUPERSEDES = "supersedes"
    SEE_ALSO = "see_also"
    EMBEDS = "embeds"
    REPLIES_TO = "replies_to"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True)
class Document:
    """A single document in the knowledge space.

    A document can be either textual (mime_type starts with 'text/') or
    binary (image/audio/etc.). For binary documents, ``content`` may hold
    a caption or human-readable description while ``binary_data`` carries
    the raw bytes.
    """
    path: str
    title: str
    content: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[str] = None
    mime_type: str = "text/markdown"
    binary_data: Optional[bytes] = None
    size_bytes: int = 0
    version: int = 1

    @property
    def is_binary(self) -> bool:
        return not self.mime_type.startswith("text/")

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    @property
    def is_audio(self) -> bool:
        return self.mime_type.startswith("audio/")

    @property
    def human_size(self) -> str:
        n = self.size_bytes or (len(self.binary_data) if self.binary_data else len(self.content.encode("utf-8")))
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"


@dataclass(frozen=True)
class DocumentVersion:
    """A historical snapshot of a document."""
    path: str
    version: int
    title: str
    content: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    mime_type: str = "text/markdown"
    binary_data: Optional[bytes] = None
    size_bytes: int = 0
    created_at: str = ""
    author: str = ""
    change_note: str = ""


@dataclass(frozen=True)
class Reference:
    """A directed link between two documents.

    Attributes:
        from_path: Source document path.
        to_path: Target document path.
        ref_type: Kind of reference (see ``RefType`` for common values).
        note: Free-form annotation, e.g. why this link exists.
        created_at: ISO timestamp when the reference was added.
    """
    from_path: str
    to_path: str
    ref_type: str = RefType.LINKS_TO
    note: str = ""
    created_at: str = ""


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


# ═══════════════════════════════════════════════════════════════════
# In-Memory Backend (for testing and lightweight use)
# ═══════════════════════════════════════════════════════════════════

class InMemoryBackend:
    """Simple dict-based storage backend for testing.

    Documents are stored in memory and lost on restart.
    Implements StorageBackend, BinaryStorage, VersionedStorage,
    and ReferenceStorage protocols. Useful for unit tests and
    quick prototyping.
    """

    def __init__(self, track_versions: bool = True):
        self._docs: Dict[str, Document] = {}
        self._versions: Dict[str, List[DocumentVersion]] = {}
        self._refs: List[Reference] = []
        self._track_versions = track_versions

    # ── Internal helpers ─────────────────────────────────────────

    def _record_version(self, doc: Document, author: str = "",
                        change_note: str = "") -> None:
        if not self._track_versions:
            return
        ver = DocumentVersion(
            path=doc.path, version=doc.version, title=doc.title,
            content=doc.content, tags=list(doc.tags),
            metadata=dict(doc.metadata), mime_type=doc.mime_type,
            binary_data=doc.binary_data, size_bytes=doc.size_bytes,
            created_at=doc.updated_at or datetime.now().isoformat(),
            author=author, change_note=change_note,
        )
        self._versions.setdefault(doc.path, []).append(ver)

    # ── StorageBackend ───────────────────────────────────────────

    def read(self, path: str) -> Optional[Document]:
        return self._docs.get(path)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              author: str = "", change_note: str = "") -> bool:
        existing = self._docs.get(path)
        next_version = (existing.version + 1) if existing else 1
        size = len(content.encode("utf-8"))
        doc = Document(
            path=path, title=title, content=content,
            tags=tags or [], metadata=metadata or {},
            updated_at=datetime.now().isoformat(),
            mime_type="text/markdown",
            binary_data=None,
            size_bytes=size,
            version=next_version,
        )
        self._docs[path] = doc
        self._record_version(doc, author=author, change_note=change_note)
        return True

    def list(self, prefix: str = "") -> List[Document]:
        return [
            doc for path, doc in sorted(self._docs.items())
            if path.startswith(prefix)
        ]

    def search(self, query: str) -> List[Document]:
        query_lower = query.lower()
        results = []
        for doc in self._docs.values():
            if (query_lower in doc.content.lower()
                    or query_lower in doc.title.lower()
                    or any(query_lower in t.lower() for t in doc.tags)):
                results.append(doc)
        return results

    def delete(self, path: str) -> bool:
        if path in self._docs:
            del self._docs[path]
            self._versions.pop(path, None)
            self._refs = [
                r for r in self._refs
                if r.from_path != path and r.to_path != path
            ]
            return True
        return False

    # ── BinaryStorage ────────────────────────────────────────────

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     author: str = "", change_note: str = "") -> bool:
        existing = self._docs.get(path)
        next_version = (existing.version + 1) if existing else 1
        doc = Document(
            path=path, title=title, content=caption,
            tags=tags or [], metadata=metadata or {},
            updated_at=datetime.now().isoformat(),
            mime_type=mime_type,
            binary_data=bytes(data),
            size_bytes=len(data),
            version=next_version,
        )
        self._docs[path] = doc
        self._record_version(doc, author=author, change_note=change_note)
        return True

    def read_binary(self, path: str) -> Optional[bytes]:
        doc = self._docs.get(path)
        return doc.binary_data if doc else None

    # ── VersionedStorage ─────────────────────────────────────────

    def list_versions(self, path: str) -> List[DocumentVersion]:
        return list(reversed(self._versions.get(path, [])))

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        for ver in self._versions.get(path, []):
            if ver.version == version:
                return ver
        return None

    def rollback(self, path: str, version: int, author: str = "",
                 change_note: str = "") -> bool:
        target = self.get_version(path, version)
        if target is None:
            return False
        existing = self._docs.get(path)
        next_version = (existing.version + 1) if existing else 1
        note = change_note or f"Rolled back to v{version}"
        doc = Document(
            path=path, title=target.title, content=target.content,
            tags=list(target.tags), metadata=dict(target.metadata),
            updated_at=datetime.now().isoformat(),
            mime_type=target.mime_type,
            binary_data=target.binary_data,
            size_bytes=target.size_bytes,
            version=next_version,
        )
        self._docs[path] = doc
        self._record_version(doc, author=author, change_note=note)
        return True

    def prune_versions(self, path: str, keep_last_n: int) -> int:
        versions = self._versions.get(path, [])
        if len(versions) <= keep_last_n:
            return 0
        removed = len(versions) - keep_last_n
        self._versions[path] = versions[-keep_last_n:]
        return removed

    # ── ReferenceStorage ─────────────────────────────────────────

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        # Idempotent: skip duplicate triples
        for r in self._refs:
            if (r.from_path == from_path and r.to_path == to_path
                    and r.ref_type == ref_type):
                return False
        self._refs.append(Reference(
            from_path=from_path, to_path=to_path,
            ref_type=ref_type, note=note,
            created_at=datetime.now().isoformat(),
        ))
        return True

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        before = len(self._refs)
        self._refs = [
            r for r in self._refs
            if not (r.from_path == from_path and r.to_path == to_path
                    and (ref_type is None or r.ref_type == ref_type))
        ]
        return before - len(self._refs)

    def list_references_from(self, path: str) -> List[Reference]:
        return [r for r in self._refs if r.from_path == path]

    def list_references_to(self, path: str) -> List[Reference]:
        return [r for r in self._refs if r.to_path == path]

    def list_all_references(self) -> List[Reference]:
        return list(self._refs)

    @property
    def document_count(self) -> int:
        return len(self._docs)


# ═══════════════════════════════════════════════════════════════════
# Location State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LocationState:
    """Tracks the agent's current position and movement history."""
    current_path: Optional[str] = None
    current_title: str = "None"
    current_content: str = ""
    history: List[str] = field(default_factory=list)
    max_history: int = 50

    def move_to(self, path: str, title: str, content: str) -> Optional[str]:
        """Move to a new location. Returns the old path or None."""
        old_path = self.current_path
        self.current_path = path
        self.current_title = title
        self.current_content = content
        self.history.append(path)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return old_path

    @property
    def has_location(self) -> bool:
        return self.current_path is not None

    @property
    def visit_count(self) -> int:
        if not self.current_path:
            return 0
        return sum(1 for p in self.history if p == self.current_path)


# ═══════════════════════════════════════════════════════════════════
# Departure Record
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DepartureRecord:
    """What happened when leaving a location."""
    path: str
    title: str
    content: str
    summary: Optional[str] = None
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════
# NavigableMemory
# ═══════════════════════════════════════════════════════════════════

class NavigableMemory:
    """Location-based memory system with automatic context loading.

    The core pattern:
    1. Agent calls navigate(path) → context at that path loads
    2. On next turn, PromptComposer module renders the location context
    3. When agent navigates away, a departure callback fires
       (for summarization, event logging, etc.)
    4. Old location context can be injected as a TTL message
       (rolling window of recent locations)

    The NavigableMemory is backend-agnostic — it works with any
    storage that implements the StorageBackend protocol.

    Usage:
        backend = InMemoryBackend()  # or SQLiteBackend, etc.
        memory = NavigableMemory(backend)

        # Seed some content
        memory.write("projects/vr/status.md", "VR Project Status",
                     "Public test in May. Voice chat needs testing.")

        # Navigate
        result = memory.navigate("projects/vr/status.md")
        print(result)  # "Navigated to: VR Project Status"

        # Get context for PromptComposer
        print(memory.build_context())  # Renders current location + nearby docs
    """

    def __init__(
        self,
        backend: StorageBackend,
        on_depart: Optional[Callable[[DepartureRecord], None]] = None,
        context_window: int = 3,
        include_siblings: bool = True,
        include_parent: bool = True,
    ):
        """Initialize NavigableMemory.

        Args:
            backend: Storage backend implementing the StorageBackend protocol.
            on_depart: Optional callback fired when leaving a location.
                Receives a DepartureRecord with the old location's data.
                Use this for summarization, archiving, event logging, etc.
            context_window: Number of recent locations to keep accessible.
            include_siblings: Whether to list sibling documents in context.
            include_parent: Whether to include parent overview in context.
        """
        self.backend = backend
        self.on_depart = on_depart
        self.context_window = context_window
        self.include_siblings = include_siblings
        self.include_parent = include_parent

        self.location = LocationState()
        self._departure_history: List[DepartureRecord] = []

    # ── Navigation ────────────────────────────────────────────────

    def navigate(self, path: str) -> str:
        """Navigate to a document path. Loads its content as current context.

        If already at a location, fires the on_depart callback with the
        old location's data before moving.

        Args:
            path: Full document path (e.g., "projects/vr/status.md").

        Returns:
            Status message describing what happened.
        """
        doc = self.backend.read(path)
        if doc is None:
            return f"Location not found: '{path}'"

        # Depart from current location
        if self.location.has_location:
            departure = DepartureRecord(
                path=self.location.current_path,
                title=self.location.current_title,
                content=self.location.current_content,
                timestamp=datetime.now().isoformat(),
            )
            self._departure_history.append(departure)
            if len(self._departure_history) > self.context_window * 2:
                self._departure_history = self._departure_history[-(self.context_window * 2):]

            if self.on_depart:
                try:
                    self.on_depart(departure)
                except Exception as e:
                    logger.error("Error in on_depart callback: %s", e)

        # Arrive at new location
        self.location.move_to(path, doc.title, doc.content)

        visit = self.location.visit_count
        visit_note = f" (visit #{visit})" if visit > 1 else ""
        logger.info("Navigated to: %s%s", doc.title, visit_note)
        return f"Navigated to: {doc.title}{visit_note}"

    def navigate_up(self) -> str:
        """Navigate to the parent directory's overview document.

        Looks for an overview.md in the parent path. If not found,
        lists what's available at the parent level.
        """
        if not self.location.has_location:
            return "No current location."

        parent = self._get_parent_prefix(self.location.current_path)
        if not parent:
            return "Already at root level."

        overview_path = f"{parent}overview.md"
        doc = self.backend.read(overview_path)
        if doc:
            return self.navigate(overview_path)

        # No overview — list what's available
        items = self.backend.list(parent)
        if items:
            listing = "\n".join(f"  - {d.title} ({d.path})" for d in items[:10])
            return f"No overview at '{parent}'. Available:\n{listing}"
        return f"Nothing found at '{parent}'."

    # ── Context Building ──────────────────────────────────────────

    def build_context(self) -> str:
        """Build the full context string for the current location.

        This is designed to be used as a PromptComposer content_fn:
            composer.add_module("context", content_fn=memory.build_context)

        Returns:
            Assembled context string with current location, parent overview,
            and sibling listings.
        """
        if not self.location.has_location:
            return "No location loaded. Use navigate to move to a knowledge path."

        parts = []

        # Parent overview for broader context
        if self.include_parent:
            parent_content = self._get_parent_overview()
            if parent_content:
                parts.append(parent_content)

        # Current location (full detail)
        # Re-read so we get mime_type/binary metadata + version freshness
        current_doc = self.backend.read(self.location.current_path)
        parts.append(self._format_current_doc(current_doc))

        # Outgoing references (graph edges)
        if self._has_references_support():
            outgoing = self.backend.list_references_from(self.location.current_path)
            if outgoing:
                lines = ["## References from here:"]
                for r in outgoing:
                    target = self.backend.read(r.to_path)
                    label = target.title if target else r.to_path
                    note = f" — {r.note}" if r.note else ""
                    lines.append(f"  → [{r.ref_type}] {label} ({r.to_path}){note}")
                parts.append("\n".join(lines))

        # Sibling documents (what else is nearby)
        if self.include_siblings:
            siblings = self._get_siblings()
            if siblings:
                listing = "\n".join(
                    f"  - {d.title} ({d.path})" for d in siblings
                )
                parts.append(f"## Nearby:\n{listing}")

        return "\n\n---\n\n".join(parts)

    def _format_current_doc(self, doc: Optional[Document]) -> str:
        """Render the current location, handling binary documents gracefully."""
        if doc is None:
            # Backend doesn't return a doc — fall back to cached state
            return (
                f"## Current: {self.location.current_title}\n"
                f"Path: {self.location.current_path}\n\n"
                f"{self.location.current_content}"
            )

        version_tag = f" (v{doc.version})" if doc.version > 1 else ""
        if doc.is_binary:
            kind = "image" if doc.is_image else ("audio" if doc.is_audio else "binary")
            header = (
                f"## Current: {doc.title}{version_tag}\n"
                f"Path: {doc.path}\n"
                f"Type: {doc.mime_type} ({doc.human_size})\n\n"
                f"[📎 {kind} attachment — bytes available via read_binary('{doc.path}')]"
            )
            if doc.content:
                header += f"\n\nCaption: {doc.content}"
            return header

        return (
            f"## Current: {doc.title}{version_tag}\n"
            f"Path: {doc.path}\n\n"
            f"{doc.content}"
        )

    def build_history_context(self) -> str:
        """Build a summary of recently visited locations.

        Useful as a secondary PromptComposer module showing
        where the agent has been.
        """
        if not self._departure_history:
            return ""

        recent = self._departure_history[-self.context_window:]
        lines = ["## Recent locations:"]
        for dep in reversed(recent):
            snippet = dep.content[:150].replace("\n", " ")
            lines.append(f"  - {dep.title} ({dep.path}): {snippet}...")
        return "\n".join(lines)

    # ── Capability checks ─────────────────────────────────────────

    def _has_binary_support(self) -> bool:
        return isinstance(self.backend, BinaryStorage)

    def _has_versioning_support(self) -> bool:
        return isinstance(self.backend, VersionedStorage)

    def _has_references_support(self) -> bool:
        return isinstance(self.backend, ReferenceStorage)

    # ── Storage Operations (pass-through + convenience) ───────────

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
            return self.backend.write(  # type: ignore[call-arg]
                path, title, content, tags, metadata,
                author=author, change_note=change_note,
            )
        except TypeError:
            return self.backend.write(path, title, content, tags, metadata)

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
            return self.backend.write_binary(  # type: ignore[call-arg]
                path, title, mime_type, data, caption,
                tags, metadata, author=author, change_note=change_note,
            )
        except TypeError:
            return self.backend.write_binary(  # type: ignore[attr-defined]
                path, title, mime_type, data, caption, tags, metadata,
            )

    def read_binary(self, path: str) -> Optional[bytes]:
        """Return the raw bytes of a binary document, or None if not found."""
        if not self._has_binary_support():
            return None
        return self.backend.read_binary(path)  # type: ignore[attr-defined]

    # ── Versioning Operations ─────────────────────────────────────

    def list_versions(self, path: str) -> List[DocumentVersion]:
        """List all historical versions of a document, newest first."""
        if not self._has_versioning_support():
            return []
        return self.backend.list_versions(path)  # type: ignore[attr-defined]

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        """Read a specific historical version."""
        if not self._has_versioning_support():
            return None
        return self.backend.get_version(path, version)  # type: ignore[attr-defined]

    def rollback(self, path: str, version: int, author: str = "",
                 change_note: str = "") -> bool:
        """Restore a document to a previous version (creating a new version)."""
        if not self._has_versioning_support():
            return False
        ok = self.backend.rollback(  # type: ignore[attr-defined]
            path, version, author=author, change_note=change_note,
        )
        # Refresh current content if we rolled back the active location
        if ok and self.location.current_path == path:
            doc = self.backend.read(path)
            if doc is not None:
                self.location.current_title = doc.title
                self.location.current_content = doc.content
        return ok

    def prune_versions(self, path: str, keep_last_n: int) -> int:
        """Drop old versions, keeping only the most recent N."""
        if not self._has_versioning_support():
            return 0
        return self.backend.prune_versions(path, keep_last_n)  # type: ignore[attr-defined]

    # ── Reference Operations ──────────────────────────────────────

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        """Create a directed reference between two documents."""
        if not self._has_references_support():
            return False
        return self.backend.add_reference(  # type: ignore[attr-defined]
            from_path, to_path, ref_type, note,
        )

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        """Remove a reference. Returns number of edges removed."""
        if not self._has_references_support():
            return 0
        return self.backend.remove_reference(  # type: ignore[attr-defined]
            from_path, to_path, ref_type,
        )

    def references_from(self, path: str) -> List[Reference]:
        """List outgoing references from a document."""
        if not self._has_references_support():
            return []
        return self.backend.list_references_from(path)  # type: ignore[attr-defined]

    def references_to(self, path: str) -> List[Reference]:
        """List incoming references to a document (backlinks)."""
        if not self._has_references_support():
            return []
        return self.backend.list_references_to(path)  # type: ignore[attr-defined]

    def all_references(self) -> List[Reference]:
        """List every reference in the store."""
        if not self._has_references_support():
            return []
        return self.backend.list_all_references()  # type: ignore[attr-defined]

    def list_at(self, prefix: str = "") -> List[Document]:
        """List documents under a path prefix."""
        return self.backend.list(prefix)

    def search(self, query: str) -> List[Document]:
        """Search across all documents."""
        return self.backend.search(query)

    # ── Tool Generation ───────────────────────────────────────────

    def create_tools(self) -> list:
        """Create FunctionTool-compatible Pydantic models for the LLM.

        Returns a list of Pydantic model classes with run() methods
        that can be wrapped with FunctionTool().

        Usage:
            from ToolAgents import FunctionTool
            tools = [FunctionTool(t) for t in memory.create_tools()]
            harness.add_tools(tools)
        """
        # Capture self in closure
        nav_memory = self

        from pydantic import BaseModel, Field

        class Navigate(BaseModel):
            """Navigate to a location in the knowledge space.
            This loads the document's content into the active context.
            Use list_locations first to discover available paths."""
            path: str = Field(
                ..., description="Full document path (e.g., 'projects/vr/status.md')."
            )

            def run(self) -> str:
                return nav_memory.navigate(self.path)

        class NavigateUp(BaseModel):
            """Navigate up to the parent area. Shows the overview
            or lists available documents at the parent level."""

            def run(self) -> str:
                return nav_memory.navigate_up()

        class ListLocations(BaseModel):
            """List available documents under a path prefix.
            Use to discover what knowledge is available."""
            prefix: str = Field(
                "", description="Path prefix with trailing slash (e.g., 'projects/')."
            )

            def run(self) -> str:
                docs = nav_memory.list_at(self.prefix)
                if not docs:
                    return f"No documents under '{self.prefix}'."
                lines = [f"Documents under '{self.prefix}':"]
                for d in docs:
                    lines.append(f"  - {d.title} ({d.path})")
                return "\n".join(lines)

        class SearchKnowledge(BaseModel):
            """Search the knowledge base for a term or topic."""
            query: str = Field(
                ..., description="Search term."
            )

            def run(self) -> str:
                results = nav_memory.search(self.query)
                if not results:
                    return f"No results for '{self.query}'."
                lines = [f"Search results for '{self.query}':"]
                for d in results[:8]:
                    snippet = d.content[:120].replace("\n", " ")
                    lines.append(f"  - {d.title} ({d.path}): {snippet}...")
                return "\n".join(lines)

        class ReadDocument(BaseModel):
            """Read a specific document without navigating to it.
            Use when you need to check content elsewhere without
            changing the current context."""
            path: str = Field(
                ..., description="Full document path."
            )

            def run(self) -> str:
                doc = nav_memory.read(self.path)
                if doc is None:
                    return f"Not found: '{self.path}'"
                return f"## {doc.title}\n\n{doc.content}"

        class WriteDocument(BaseModel):
            """Write or update a document in the knowledge base.
            Creates the document if it doesn't exist, overwrites if it does."""
            path: str = Field(
                ..., description="Full document path."
            )
            title: str = Field(
                ..., description="Document title."
            )
            content: str = Field(
                ..., description="Document content (markdown)."
            )

            def run(self) -> str:
                ok = nav_memory.write(self.path, self.title, self.content)
                return f"Written: '{self.title}'" if ok else f"Failed to write '{self.path}'."

        class AppendToDocument(BaseModel):
            """Append content to an existing document.
            Adds a timestamped log entry. Useful for event logs and notes."""
            path: str = Field(
                ..., description="Full document path."
            )
            content: str = Field(
                ..., description="Content to append."
            )

            def run(self) -> str:
                return nav_memory.append(self.path, self.content)

        # ── Versioning Tools ─────────────────────────────────────

        class ListVersions(BaseModel):
            """List historical versions of a document, newest first.
            Use to inspect change history before reading or rolling back."""
            path: str = Field(..., description="Full document path.")

            def run(self) -> str:
                versions = nav_memory.list_versions(self.path)
                if not versions:
                    return (
                        f"No version history for '{self.path}' "
                        "(or backend does not support versioning)."
                    )
                lines = [f"Versions of '{self.path}':"]
                for v in versions:
                    note = f" — {v.change_note}" if v.change_note else ""
                    author = f" by {v.author}" if v.author else ""
                    lines.append(
                        f"  v{v.version} ({v.created_at}){author}{note}"
                    )
                return "\n".join(lines)

        class ReadVersion(BaseModel):
            """Read a specific historical version of a document.
            Does not navigate or change current location."""
            path: str = Field(..., description="Full document path.")
            version: int = Field(..., description="Version number to read.")

            def run(self) -> str:
                ver = nav_memory.get_version(self.path, self.version)
                if ver is None:
                    return f"Version {self.version} of '{self.path}' not found."
                header = (
                    f"## {ver.title} (v{ver.version})\n"
                    f"Saved: {ver.created_at}"
                )
                if ver.author:
                    header += f" by {ver.author}"
                if ver.change_note:
                    header += f"\nNote: {ver.change_note}"
                if ver.mime_type and not ver.mime_type.startswith("text/"):
                    return (
                        f"{header}\n\n[binary {ver.mime_type}, "
                        f"{ver.size_bytes} bytes]"
                        + (f"\nCaption: {ver.content}" if ver.content else "")
                    )
                return f"{header}\n\n{ver.content}"

        class RollbackToVersion(BaseModel):
            """Restore a document to a previous version.
            Creates a NEW version on top — history is never lost."""
            path: str = Field(..., description="Full document path.")
            version: int = Field(..., description="Version number to restore.")
            reason: str = Field(
                "", description="Optional change note explaining the rollback."
            )

            def run(self) -> str:
                ok = nav_memory.rollback(
                    self.path, self.version,
                    change_note=self.reason or f"rolled back to v{self.version}",
                )
                if not ok:
                    return (
                        f"Could not rollback '{self.path}' to v{self.version} "
                        "(version not found or backend lacks versioning)."
                    )
                return f"Restored '{self.path}' to v{self.version}."

        # ── Reference Tools ──────────────────────────────────────

        class AddReference(BaseModel):
            """Create a directed reference between two documents.
            Use to capture relationships: links, dependencies, supersedes, etc."""
            from_path: str = Field(..., description="Source document path.")
            to_path: str = Field(..., description="Target document path.")
            ref_type: str = Field(
                "links_to",
                description=(
                    "Relationship kind. Common values: 'links_to', 'depends_on', "
                    "'supersedes', 'see_also', 'embeds', 'replies_to', 'derived_from'."
                ),
            )
            note: str = Field(
                "", description="Optional annotation explaining the link."
            )

            def run(self) -> str:
                ok = nav_memory.add_reference(
                    self.from_path, self.to_path, self.ref_type, self.note,
                )
                if not ok:
                    return (
                        f"Reference '{self.from_path}' →[{self.ref_type}]→ "
                        f"'{self.to_path}' already exists or backend lacks support."
                    )
                return (
                    f"Linked: '{self.from_path}' →[{self.ref_type}]→ "
                    f"'{self.to_path}'"
                )

        class RemoveReference(BaseModel):
            """Remove a reference between two documents."""
            from_path: str = Field(..., description="Source document path.")
            to_path: str = Field(..., description="Target document path.")
            ref_type: Optional[str] = Field(
                None,
                description=(
                    "If given, only remove references of this type. "
                    "Otherwise remove all edges between the two paths."
                ),
            )

            def run(self) -> str:
                n = nav_memory.remove_reference(
                    self.from_path, self.to_path, self.ref_type,
                )
                return f"Removed {n} reference(s)."

        class ListReferences(BaseModel):
            """List references for a document.
            Direction: 'from' = outgoing links; 'to' = backlinks; 'both' = both."""
            path: str = Field(..., description="Full document path.")
            direction: str = Field(
                "both",
                description="Direction: 'from', 'to', or 'both'. Default 'both'.",
            )

            def run(self) -> str:
                direction = self.direction.lower().strip()
                lines: List[str] = []
                if direction in ("from", "both"):
                    out = nav_memory.references_from(self.path)
                    if out:
                        lines.append(f"Outgoing from '{self.path}':")
                        for r in out:
                            note = f" — {r.note}" if r.note else ""
                            lines.append(
                                f"  → [{r.ref_type}] {r.to_path}{note}"
                            )
                if direction in ("to", "both"):
                    incoming = nav_memory.references_to(self.path)
                    if incoming:
                        lines.append(f"Backlinks to '{self.path}':")
                        for r in incoming:
                            note = f" — {r.note}" if r.note else ""
                            lines.append(
                                f"  ← [{r.ref_type}] {r.from_path}{note}"
                            )
                if not lines:
                    return f"No references found for '{self.path}'."
                return "\n".join(lines)

        # ── Binary Tools ─────────────────────────────────────────

        class DescribeBinary(BaseModel):
            """Inspect a binary document's metadata (mime type, size, caption).
            Useful before deciding whether to retrieve raw bytes."""
            path: str = Field(..., description="Full document path.")

            def run(self) -> str:
                doc = nav_memory.read(self.path)
                if doc is None:
                    return f"Not found: '{self.path}'"
                if not doc.is_binary:
                    return f"'{self.path}' is a text document ({doc.mime_type})."
                lines = [
                    f"## {doc.title}",
                    f"Path: {doc.path}",
                    f"Type: {doc.mime_type}",
                    f"Size: {doc.human_size}",
                    f"Version: {doc.version}",
                ]
                if doc.content:
                    lines.append(f"Caption: {doc.content}")
                if doc.tags:
                    lines.append(f"Tags: {', '.join(doc.tags)}")
                return "\n".join(lines)

        tools = [
            Navigate, NavigateUp, ListLocations, SearchKnowledge,
            ReadDocument, WriteDocument, AppendToDocument,
        ]
        if self._has_versioning_support():
            tools.extend([ListVersions, ReadVersion, RollbackToVersion])
        if self._has_references_support():
            tools.extend([AddReference, RemoveReference, ListReferences])
        if self._has_binary_support():
            tools.append(DescribeBinary)
        return tools

    # ── Internal Helpers ──────────────────────────────────────────

    def _get_parent_prefix(self, path: str) -> Optional[str]:
        """Get the parent directory prefix for a path."""
        parts = path.rsplit("/", 1)
        if len(parts) > 1:
            return parts[0] + "/"
        return None

    def _get_parent_overview(self) -> Optional[str]:
        """Load the parent directory's overview document."""
        if not self.location.current_path:
            return None
        parent = self._get_parent_prefix(self.location.current_path)
        if not parent:
            return None

        overview_path = f"{parent}overview.md"
        if overview_path == self.location.current_path:
            return None

        doc = self.backend.read(overview_path)
        if doc:
            return f"## Area: {doc.title}\n{doc.content}"
        return None

    def _get_siblings(self) -> List[Document]:
        """Get sibling documents in the same directory."""
        if not self.location.current_path:
            return []
        parent = self._get_parent_prefix(self.location.current_path)
        if not parent:
            return []

        docs = self.backend.list(parent)
        return [
            d for d in docs
            if d.path != self.location.current_path
            and not d.path.endswith("overview.md")
        ]

    # ── State Access ──────────────────────────────────────────────

    @property
    def current_path(self) -> Optional[str]:
        return self.location.current_path

    @property
    def current_title(self) -> str:
        return self.location.current_title

    @property
    def history(self) -> List[str]:
        return self.location.history

    @property
    def departure_records(self) -> List[DepartureRecord]:
        return list(self._departure_history)

    def __repr__(self) -> str:
        loc = self.location.current_title if self.location.has_location else "None"
        return (
            f"NavigableMemory(at='{loc}', "
            f"history={len(self.location.history)}, "
            f"departures={len(self._departure_history)})"
        )

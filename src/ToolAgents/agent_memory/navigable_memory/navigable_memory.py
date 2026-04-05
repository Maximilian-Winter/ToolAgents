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

@dataclass(frozen=True)
class Document:
    """A single document in the knowledge space."""
    path: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[str] = None


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


# ═══════════════════════════════════════════════════════════════════
# In-Memory Backend (for testing and lightweight use)
# ═══════════════════════════════════════════════════════════════════

class InMemoryBackend:
    """Simple dict-based storage backend for testing.

    Documents are stored in memory and lost on restart.
    Useful for unit tests and quick prototyping.
    """

    def __init__(self):
        self._docs: Dict[str, Document] = {}

    def read(self, path: str) -> Optional[Document]:
        return self._docs.get(path)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        self._docs[path] = Document(
            path=path, title=title, content=content,
            tags=tags or [], metadata=metadata or {},
            updated_at=datetime.now().isoformat(),
        )
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
            return True
        return False

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
        parts.append(
            f"## Current: {self.location.current_title}\n"
            f"Path: {self.location.current_path}\n\n"
            f"{self.location.current_content}"
        )

        # Sibling documents (what else is nearby)
        if self.include_siblings:
            siblings = self._get_siblings()
            if siblings:
                listing = "\n".join(
                    f"  - {d.title} ({d.path})" for d in siblings
                )
                parts.append(f"## Nearby:\n{listing}")

        return "\n\n---\n\n".join(parts)

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

    # ── Storage Operations (pass-through + convenience) ───────────

    def read(self, path: str) -> Optional[Document]:
        """Read a document without navigating to it."""
        return self.backend.read(path)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None) -> bool:
        """Write a document to storage."""
        return self.backend.write(path, title, content, tags)

    def append(self, path: str, content: str) -> str:
        """Append content to an existing document.

        Useful for event logs, notes, and accumulating observations.
        """
        doc = self.backend.read(path)
        if doc is None:
            return f"Document not found: '{path}'"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated = doc.content + f"\n\n### Log — {timestamp}\n{content}"
        self.backend.write(path, doc.title, updated, doc.tags)

        # If we're currently AT this document, refresh the content
        if self.location.current_path == path:
            self.location.current_content = updated

        return f"Appended to '{doc.title}'."

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

        return [Navigate, NavigateUp, ListLocations, SearchKnowledge,
                ReadDocument, WriteDocument, AppendToDocument]

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

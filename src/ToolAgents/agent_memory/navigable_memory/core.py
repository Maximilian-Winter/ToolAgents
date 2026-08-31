"""Core NavigableMemory class."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from .context import NavigableContextMixin
from .documents import NavigableDocumentMixin
from .models import DepartureRecord
from .protocols import BinaryStorage, ReferenceStorage, StorageBackend, VersionedStorage
from .references import NavigableReferenceMixin
from .search import NavigableSearchMixin
from .state import LocationState
from .tags import NavigableTagMixin
from .tools import create_navigable_memory_tools
from .versions import NavigableVersionMixin

logger = logging.getLogger(__name__)


class NavigableMemory(
    NavigableContextMixin,
    NavigableDocumentMixin,
    NavigableVersionMixin,
    NavigableReferenceMixin,
    NavigableTagMixin,
    NavigableSearchMixin,
):
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
        semantic_index: Any = None,
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
            semantic_index: Optional secondary semantic index. The index is
                attached by duck typing and is not a storage backend.
        """
        self.backend = backend
        self.on_depart = on_depart
        self.context_window = context_window
        self.include_siblings = include_siblings
        self.include_parent = include_parent
        self.semantic_index = None

        self.location = LocationState()
        self._departure_history: List[DepartureRecord] = []
        if semantic_index is not None:
            self.set_semantic_index(semantic_index)

    def _has_binary_support(self) -> bool:
        return isinstance(self.backend, BinaryStorage)

    def _has_versioning_support(self) -> bool:
        return isinstance(self.backend, VersionedStorage)

    def _has_references_support(self) -> bool:
        return isinstance(self.backend, ReferenceStorage)

    def _has_semantic_support(self) -> bool:
        return self.semantic_index is not None

    def set_semantic_index(self, semantic_index: Any) -> Any:
        """Attach an optional secondary semantic index.

        The index object should provide ``index_document``,
        ``remove_document``, ``search``, and ``build_search_context`` methods.
        If it exposes ``attach_memory()``, this memory instance is passed in.
        """
        self.semantic_index = semantic_index
        attach = getattr(semantic_index, "attach_memory", None)
        if callable(attach):
            attach(self)
        return semantic_index


    def create_tools(self) -> list:
        """Create FunctionTool-compatible Pydantic models for the LLM."""
        return create_navigable_memory_tools(self)

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

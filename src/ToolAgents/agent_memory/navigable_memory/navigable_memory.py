"""Compatibility re-exports for the split NavigableMemory modules.

Prefer importing from ``ToolAgents.agent_memory.navigable_memory``. This module
remains so direct imports from the historical ``navigable_memory.py`` path keep
working.
"""

from .core import NavigableMemory
from .in_memory_backend import InMemoryBackend
from .models import DepartureRecord, Document, DocumentVersion, Reference, RefType
from .protocols import (
    BinaryStorage,
    ReferenceStorage,
    StorageBackend,
    TagStorage,
    VersionedStorage,
)
from .state import LocationState

__all__ = [
    "NavigableMemory",
    "InMemoryBackend",
    "DepartureRecord",
    "Document",
    "DocumentVersion",
    "Reference",
    "RefType",
    "StorageBackend",
    "BinaryStorage",
    "VersionedStorage",
    "ReferenceStorage",
    "TagStorage",
    "LocationState",
]

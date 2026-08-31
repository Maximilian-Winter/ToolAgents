"""Public data models for NavigableMemory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        n = self.size_bytes or (
            len(self.binary_data)
            if self.binary_data else len(self.content.encode("utf-8"))
        )
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
        n = self.size_bytes or (
            len(self.binary_data)
            if self.binary_data else len(self.content.encode("utf-8"))
        )
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"


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


@dataclass(frozen=True)
class DepartureRecord:
    """What happened when leaving a location."""
    path: str
    title: str
    content: str
    summary: Optional[str] = None
    timestamp: str = ""

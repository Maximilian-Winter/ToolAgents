# src/ToolAgents/extensions/handler.py
"""FolderHandler protocol — the contract for folder-based extension handlers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ToolAgents.function_tool import FunctionTool
    from .manager import ExtensionManager
    from .models import ActivationResult, ExtensionEntry


@runtime_checkable
class FolderHandler(Protocol):
    """Protocol for folder-based extension handlers.

    Each handler knows how to discover, catalog, and activate extensions
    from folders containing a specific marker file (e.g., SKILL.md).

    Attributes:
        name: Handler type identifier (e.g., "skills").
        marker_file: Filename to look for in directories (e.g., "SKILL.md").
    """

    name: str
    marker_file: str

    def discover(self, path: Path) -> Optional["ExtensionEntry"]:
        """Parse a folder's marker file into an ExtensionEntry."""
        ...

    def build_catalog(self, entries: List["ExtensionEntry"]) -> str:
        """Build a system prompt section listing available extensions."""
        ...

    def activate(self, entry: "ExtensionEntry") -> "ActivationResult":
        """Load full extension content for injection into context."""
        ...

    def get_tools(self, manager: "ExtensionManager") -> List["FunctionTool"]:
        """Return tools this handler wants registered with the harness."""
        ...

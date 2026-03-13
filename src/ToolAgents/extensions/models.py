# src/ToolAgents/extensions/models.py
"""Data models for the extension system."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ToolAgents.function_tool import FunctionTool

logger = logging.getLogger(__name__)


@dataclass
class ExtensionEntry:
    """Lightweight metadata for a discovered extension.

    Attributes:
        name: Extension identifier from frontmatter (e.g., "pdf-processing").
        description: What the extension does and when to use it.
        handler_type: Which handler owns this entry (e.g., "skills").
        path: Absolute path to the marker file (e.g., SKILL.md).
        scope: Where this was discovered — "project", "user", or "builtin".
        priority: From the scan path — higher wins on name collision.
        metadata: Additional frontmatter fields (license, compatibility, etc.).
    """

    name: str
    description: str
    handler_type: str
    path: Path
    scope: str
    priority: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ActivationResult:
    """What gets returned when an extension is activated.

    Attributes:
        content: Full instructions/content to inject into context.
        pin_in_context: Whether to protect this content from context trimming.
        tools: Optional tools to register with the harness on activation.
        resources: Bundled file paths for model awareness (not eagerly loaded).
    """

    content: str
    pin_in_context: bool = True
    tools: Optional[List["FunctionTool"]] = None
    resources: Optional[List[str]] = None


@dataclass
class ExtensionScanPath:
    """Configures a directory to scan for extensions.

    Attributes:
        path: Directory to scan.
        scope: "project", "user", or "builtin".
        priority: Higher priority wins on name collision (project=10, user=0).
    """

    path: Path
    scope: str = "project"
    priority: int = 0

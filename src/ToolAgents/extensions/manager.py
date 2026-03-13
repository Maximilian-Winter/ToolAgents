# src/ToolAgents/extensions/manager.py
"""ExtensionManager — orchestrator for folder-based extensions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ToolAgents.function_tool import FunctionTool
    from .handler import FolderHandler

from .models import ActivationResult, ExtensionEntry, ExtensionScanPath

logger = logging.getLogger(__name__)

SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".eggs",
})


class ExtensionManager:
    """Orchestrator for folder-based extensions."""

    def __init__(self) -> None:
        self._handlers: Dict[str, "FolderHandler"] = {}
        self._scan_paths: List[ExtensionScanPath] = []
        self._entries: Dict[str, ExtensionEntry] = {}
        self._active: Dict[str, ActivationResult] = {}
        self._pending_activations: Dict[str, ActivationResult] = {}

    def register_handler(self, handler: "FolderHandler") -> None:
        self._handlers[handler.name] = handler

    def add_scan_path(self, scan_path: ExtensionScanPath) -> None:
        self._scan_paths.append(scan_path)

    def discover(self) -> Dict[str, List[ExtensionEntry]]:
        self._entries.clear()
        self._active.clear()
        self._pending_activations.clear()

        sorted_paths = sorted(self._scan_paths, key=lambda sp: sp.priority, reverse=True)

        for scan_path in sorted_paths:
            if not scan_path.path.is_dir():
                logger.debug("Scan path does not exist: %s", scan_path.path)
                continue

            for subdir in sorted(scan_path.path.iterdir()):
                if not subdir.is_dir():
                    continue
                if subdir.name in SKIP_DIRS:
                    continue

                for handler in self._handlers.values():
                    marker = subdir / handler.marker_file
                    if not marker.is_file():
                        continue

                    entry = handler.discover(marker)
                    if entry is None:
                        continue

                    entry.scope = scan_path.scope
                    entry.priority = scan_path.priority

                    if entry.name in self._entries:
                        existing = self._entries[entry.name]
                        if existing.priority >= entry.priority:
                            logger.warning(
                                "Extension '%s' from %s (priority %d) shadowed by existing from %s (priority %d)",
                                entry.name, scan_path.scope, scan_path.priority, existing.scope, existing.priority,
                            )
                            continue
                        else:
                            logger.warning(
                                "Extension '%s' from %s (priority %d) replaces existing from %s (priority %d)",
                                entry.name, scan_path.scope, scan_path.priority, existing.scope, existing.priority,
                            )

                    self._entries[entry.name] = entry

        grouped: Dict[str, List[ExtensionEntry]] = {}
        for entry in self._entries.values():
            grouped.setdefault(entry.handler_type, []).append(entry)
        return grouped

    def build_catalog(self) -> str:
        if not self._entries:
            return ""
        grouped: Dict[str, List[ExtensionEntry]] = {}
        for entry in self._entries.values():
            grouped.setdefault(entry.handler_type, []).append(entry)
        sections = []
        for handler_name, entries in grouped.items():
            handler = self._handlers.get(handler_name)
            if handler is None:
                continue
            catalog = handler.build_catalog(entries)
            if catalog:
                sections.append(catalog)
        return "\n\n".join(sections)

    def activate(self, name: str) -> Optional[ActivationResult]:
        if name in self._active:
            return self._active[name]
        entry = self._entries.get(name)
        if entry is None:
            return None
        handler = self._handlers.get(entry.handler_type)
        if handler is None:
            return None
        result = handler.activate(entry)
        self._active[name] = result
        return result

    def deactivate(self, name: str) -> None:
        self._active.pop(name, None)

    def is_active(self, name: str) -> bool:
        return name in self._active

    def try_handle_command(self, command: str) -> Optional[ActivationResult]:
        if command in self._entries:
            return self.activate(command)
        return None

    def get_tools(self) -> List["FunctionTool"]:
        tools = []
        for handler in self._handlers.values():
            tools.extend(handler.get_tools(self))
        return tools

    @property
    def entries(self) -> Dict[str, ExtensionEntry]:
        return dict(self._entries)

    @property
    def active_entries(self) -> Dict[str, ExtensionEntry]:
        return {name: self._entries[name] for name in self._active if name in self._entries}

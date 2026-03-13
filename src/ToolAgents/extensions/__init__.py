# src/ToolAgents/extensions/__init__.py
"""
ToolAgents Extensions — Folder-based extension system.

A protocol-based framework for loading capabilities from folder structures.
Starts with Agent Skills (SKILL.md) support, extensible to other folder types.
"""

__all__ = [
    "ExtensionEntry",
    "ActivationResult",
    "ExtensionScanPath",
    "FolderHandler",
    "ExtensionManager",
    "SkillFolderHandler",
]


def __getattr__(name: str):
    if name in {"ExtensionEntry", "ActivationResult", "ExtensionScanPath"}:
        from .models import ActivationResult, ExtensionEntry, ExtensionScanPath
        return {"ExtensionEntry": ExtensionEntry, "ActivationResult": ActivationResult, "ExtensionScanPath": ExtensionScanPath}[name]

    if name == "FolderHandler":
        from .handler import FolderHandler
        return FolderHandler

    if name == "ExtensionManager":
        from .manager import ExtensionManager
        return ExtensionManager

    if name == "SkillFolderHandler":
        from .skill_handler import SkillFolderHandler
        return SkillFolderHandler

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

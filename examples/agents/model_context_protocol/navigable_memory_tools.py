"""Shared NavigableMemory tool source for CLI and MCP examples."""

from __future__ import annotations

import os
from pathlib import Path

from ToolAgents import FunctionTool
from ToolAgents.agent_memory.navigable_memory import JSONBackend, NavigableMemory


def create_memory() -> NavigableMemory:
    memory_path = Path(
        os.environ.get("TOOLAGENTS_NAV_MEMORY_FILE", "navigable_memory_example.json")
    )
    memory = NavigableMemory(JSONBackend(str(memory_path)))
    if memory.read("notes/welcome.md") is None:
        memory.write(
            "notes/welcome.md",
            "Welcome",
            "This is a small persistent NavigableMemory example.",
            tags=["example"],
        )
    return memory


def create_tools() -> list[FunctionTool]:
    memory = create_memory()
    return [FunctionTool(tool) for tool in memory.create_tools()]

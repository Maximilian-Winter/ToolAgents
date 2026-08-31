"""Small entry point for trying NavigableMemory through the generic tool CLI.

Examples:
    python -m ToolAgents.tool_adapters.cli list --module navigable_memory_tools:create_tools
    python -m ToolAgents.tool_adapters.cli call ReadDocument --module navigable_memory_tools:create_tools --json "{\"path\":\"notes/welcome.md\"}"
"""

from __future__ import annotations

from ToolAgents.tool_adapters.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

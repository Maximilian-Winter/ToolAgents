"""Schema helpers shared by CLI and MCP adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ToolAgents.function_tool import FunctionTool
from ToolAgents.utilities.mcp_conversion import convert_json_schema


@dataclass(frozen=True)
class ToolAdapterMetadata:
    """Portable metadata for a ToolAgents tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


def function_tool_to_input_schema(tool: FunctionTool) -> dict[str, Any]:
    """Return the JSON input schema for a FunctionTool."""

    root = tool.to_openai_tool()
    return root["function"]["parameters"]


def function_tool_to_mcp_metadata(tool: FunctionTool) -> ToolAdapterMetadata:
    """Return MCP-compatible metadata for a FunctionTool."""

    root = tool.to_openai_tool()["function"]
    return ToolAdapterMetadata(
        name=root["name"],
        description=root.get("description", "") or "",
        input_schema=root["parameters"],
    )


def json_schema_to_pydantic_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON schema into a Pydantic model class."""

    return convert_json_schema(schema)

"""Shared execution helpers for portable ToolAgents tools."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel

from ToolAgents.function_tool import AsyncFunctionTool, FunctionTool, ToolRegistry


ToolLike = FunctionTool | type[BaseModel]
ToolCollection = ToolRegistry | Mapping[str, FunctionTool] | Iterable[ToolLike]


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured result for adapter callers that want success/error metadata."""

    tool_name: str
    ok: bool
    result: Any = None
    error: str | None = None


def ensure_function_tool(tool: ToolLike) -> FunctionTool:
    """Return a FunctionTool, wrapping Pydantic model classes when needed."""

    if isinstance(tool, FunctionTool):
        return tool
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        return FunctionTool(tool)
    raise TypeError(f"Unsupported tool type: {type(tool)!r}")


def normalize_tools(tools: ToolCollection | Sequence[ToolLike]) -> list[FunctionTool]:
    """Normalize registries, mappings, and iterables to FunctionTool instances."""

    if isinstance(tools, ToolRegistry):
        return list(tools.get_tools())
    if isinstance(tools, Mapping):
        return [ensure_function_tool(tool) for tool in tools.values()]
    return [ensure_function_tool(tool) for tool in tools]


def tool_name(tool: FunctionTool) -> str:
    """Return the public name of a FunctionTool."""

    return tool.model.__name__


def find_tool(tools: ToolCollection | Sequence[ToolLike], name: str) -> FunctionTool | None:
    """Find a tool by public name."""

    if isinstance(tools, ToolRegistry):
        return tools.get_tool(name)
    for tool in normalize_tools(tools):
        if tool_name(tool) == name:
            return tool
    return None


def execute_tool(tool: FunctionTool, arguments: dict[str, Any] | None = None) -> Any:
    """Execute a ToolAgents tool synchronously."""

    if isinstance(tool, AsyncFunctionTool):
        raise TypeError(
            f"{tool_name(tool)} is an AsyncFunctionTool; use async_execute_tool()."
        )
    return tool.execute(arguments or {})


def execute_tool_by_name(
    tools: ToolCollection | Sequence[ToolLike],
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Execute a named tool from a registry, mapping, or iterable."""

    tool = find_tool(tools, name)
    if tool is None:
        raise KeyError(f"Unknown tool: {name}")
    return execute_tool(tool, arguments)


async def async_execute_tool(
    tool: FunctionTool,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Execute a ToolAgents tool in an async context."""

    maybe_result = tool.execute_async(arguments or {})
    if inspect.isawaitable(maybe_result):
        return await maybe_result
    return maybe_result


async def async_execute_tool_by_name(
    tools: ToolCollection | Sequence[ToolLike],
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Execute a named tool from a registry, mapping, or iterable asynchronously."""

    tool = find_tool(tools, name)
    if tool is None:
        raise KeyError(f"Unknown tool: {name}")
    return await async_execute_tool(tool, arguments)


def run_tool(
    tool: FunctionTool,
    arguments: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    """Execute a tool and capture exceptions in a structured result."""

    name = tool_name(tool)
    try:
        return ToolExecutionResult(name, True, execute_tool(tool, arguments))
    except Exception as exc:  # pragma: no cover - convenience wrapper
        return ToolExecutionResult(name, False, error=str(exc))


async def async_run_tool(
    tool: FunctionTool,
    arguments: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    """Execute a tool asynchronously and capture exceptions."""

    name = tool_name(tool)
    try:
        return ToolExecutionResult(name, True, await async_execute_tool(tool, arguments))
    except Exception as exc:  # pragma: no cover - convenience wrapper
        return ToolExecutionResult(name, False, error=str(exc))


def format_tool_result(result: Any) -> str:
    """Convert arbitrary tool output into CLI/MCP friendly text."""

    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json(indent=2)
    try:
        return json.dumps(result, indent=2, default=str)
    except TypeError:
        return str(result)

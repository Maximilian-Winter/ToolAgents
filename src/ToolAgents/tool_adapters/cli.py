"""Generic command line adapter for ToolAgents tools."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from typing import Any

from pydantic import BaseModel

from ToolAgents.function_tool import ToolRegistry
from ToolAgents.tool_adapters.execution import (
    ToolCollection,
    execute_tool_by_name,
    find_tool,
    format_tool_result,
    normalize_tools,
)
from ToolAgents.tool_adapters.schemas import function_tool_to_input_schema


def load_tools_from_spec(spec: str) -> list:
    """Load tools from 'module:attribute'."""

    if ":" not in spec:
        raise ValueError("Tool module must use the form 'module:attribute'.")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    if (
        inspect.isfunction(value)
        or inspect.ismethod(value)
    ) and not isinstance(value, ToolRegistry):
        value = value()
    elif isinstance(value, type) and issubclass(value, BaseModel):
        value = [value]
    return normalize_tools(value)


def _parse_json_argument(json_text: str | None, json_file: str | None) -> dict[str, Any]:
    if json_text and json_file:
        raise ValueError("Use either --json or --json-file, not both.")
    if json_file:
        with open(json_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if json_text:
        return json.loads(json_text)
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolagents-tools",
        description="List and call ToolAgents FunctionTool collections.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available tools.")
    list_parser.add_argument("--module", required=True, help="Tool source as module:attribute.")

    schema_parser = subparsers.add_parser("schema", help="Print a tool input schema.")
    schema_parser.add_argument("tool_name", help="Tool name.")
    schema_parser.add_argument("--module", required=True, help="Tool source as module:attribute.")

    call_parser = subparsers.add_parser("call", help="Call a tool with JSON arguments.")
    call_parser.add_argument("tool_name", help="Tool name.")
    call_parser.add_argument("--module", required=True, help="Tool source as module:attribute.")
    call_parser.add_argument("--json", help="JSON object with tool arguments.")
    call_parser.add_argument("--json-file", help="Path to a JSON file with tool arguments.")

    return parser


def _print_tool_list(tools: ToolCollection) -> None:
    for tool in normalize_tools(tools):
        description = (tool.model.__doc__ or "").strip().splitlines()
        summary = f" - {description[0]}" if description else ""
        print(f"{tool.model.__name__}{summary}")


DEPRECATION_NOTICE = (
    "note: 'toolagents-tools' is deprecated and will be removed in a future "
    "release. The same commands live under 'tool-agents tools', which can "
    "also read tool modules from a .tool-agents workspace."
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(DEPRECATION_NOTICE, file=sys.stderr)

    try:
        tools = load_tools_from_spec(args.module)
        if args.command == "list":
            _print_tool_list(tools)
            return 0

        tool = find_tool(tools, args.tool_name)
        if tool is None:
            print(f"Unknown tool: {args.tool_name}", file=sys.stderr)
            return 2

        if args.command == "schema":
            print(json.dumps(function_tool_to_input_schema(tool), indent=2))
            return 0

        if args.command == "call":
            arguments = _parse_json_argument(args.json, args.json_file)
            result = execute_tool_by_name(tools, args.tool_name, arguments)
            print(format_tool_result(result))
            return 0

    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

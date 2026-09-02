"""The ``tool-agents`` command line interface.

Runs workflows from a :mod:`~ToolAgents.workspace` directory::

    tool-agents init
    tool-agents list
    tool-agents show digest
    tool-agents run digest --arg topic=otters --allow-writes

The tool inspection commands that used to be ``toolagents-tools`` live under
``tool-agents tools``; the old entry point still works and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ToolAgents.workspace import (
    DEFAULT_WORKSPACE_DIRNAME,
    Workspace,
    WorkspaceError,
)

__all__ = ["build_parser", "main", "parse_argument"]

PROGRAM = "tool-agents"


def parse_argument(pair: str) -> tuple[str, Any]:
    """Parse one ``key=value`` argument.

    The value is decoded as JSON when it can be, so ``--arg n=3`` gives an
    integer, ``--arg tags=["a","b"]`` a list, and ``--arg name=Ada`` the plain
    string it looks like.
    """

    if "=" not in pair:
        raise ValueError(f"Argument must be key=value, got {pair!r}.")
    key, _, raw = pair.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"Argument is missing a name: {pair!r}.")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def collect_arguments(
    pairs: Sequence[str] | None,
    json_text: str | None,
    json_file: str | None,
) -> dict[str, Any]:
    """Merge ``--arg``, ``--json`` and ``--json-file`` into one mapping."""

    arguments: dict[str, Any] = {}
    if json_file:
        with open(json_file, "r", encoding="utf-8") as handle:
            arguments.update(json.load(handle))
    if json_text:
        arguments.update(json.loads(json_text))
    for pair in pairs or []:
        key, value = parse_argument(pair)
        arguments[key] = value
    return arguments


def build_parser() -> argparse.ArgumentParser:
    """Build the ``tool-agents`` argument parser."""

    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Run ToolAgents workflows from a .tool-agents folder.",
    )
    parser.add_argument(
        "--env-file",
        help="A .env file to read before building providers. Defaults to "
        f"{DEFAULT_WORKSPACE_DIRNAME}/.env when it exists.",
    )
    parser.add_argument(
        "--workspace",
        help=f"Path to a {DEFAULT_WORKSPACE_DIRNAME} folder. Defaults to the "
        "nearest one at or above the working directory.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create a workspace folder.")
    init.add_argument(
        "path", nargs="?", help="Where to create it. Defaults to the working directory."
    )

    listing = commands.add_parser("list", help="List what the workspace holds.")
    listing.add_argument(
        "kind",
        nargs="?",
        choices=("workflows", "tools", "prompts", "providers", "adapters"),
        help="Limit the listing to one kind.",
    )

    show = commands.add_parser("show", help="Show one workflow.")
    show.add_argument("workflow", help="Workflow name.")

    run = commands.add_parser("run", help="Run one workflow.")
    run.add_argument("workflow", help="Workflow name.")
    run.add_argument(
        "--arg",
        action="append",
        metavar="KEY=VALUE",
        help="An input argument. Repeatable. Values parse as JSON when they can.",
    )
    run.add_argument("--json", help="JSON object of input arguments.")
    run.add_argument("--json-file", help="Path to a JSON file of input arguments.")
    run.add_argument(
        "--allow-writes",
        action="store_true",
        help="Permit sinks that write files or make HTTP requests.",
    )
    run.add_argument(
        "--output",
        metavar="PATH",
        help="Print only this results path, such as outputs/draft.",
    )
    run.add_argument(
        "--json-output",
        action="store_true",
        help="Print the whole results object as JSON instead of a summary.",
    )

    tools = commands.add_parser("tools", help="Inspect and call tools directly.")
    tool_commands = tools.add_subparsers(dest="tools_command", required=True)

    tool_list = tool_commands.add_parser("list", help="List available tools.")
    tool_list.add_argument("--module", help="Tool source as module:attribute.")
    tool_list.add_argument("--plugin", help="Workspace tool module name.")

    tool_schema = tool_commands.add_parser("schema", help="Print a tool input schema.")
    tool_schema.add_argument("tool_name", help="Tool name.")
    tool_schema.add_argument("--module", help="Tool source as module:attribute.")
    tool_schema.add_argument("--plugin", help="Workspace tool module name.")

    tool_call = tool_commands.add_parser("call", help="Call a tool with JSON arguments.")
    tool_call.add_argument("tool_name", help="Tool name.")
    tool_call.add_argument("--module", help="Tool source as module:attribute.")
    tool_call.add_argument("--plugin", help="Workspace tool module name.")
    tool_call.add_argument("--json", help="JSON object with tool arguments.")
    tool_call.add_argument("--json-file", help="Path to a JSON file with arguments.")

    return parser


def _workspace(args: argparse.Namespace) -> Workspace:
    if args.workspace:
        return Workspace(Path(args.workspace))
    return Workspace.discover()


def _command_init(args: argparse.Namespace) -> int:
    workspace = Workspace.create(args.path)
    print(f"Created {workspace.root}")
    for name in sorted(workspace.summary()):
        print(f"  {name}/")
    return 0


def _command_list(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    summary = workspace.summary()
    kinds = [args.kind] if args.kind else sorted(summary)

    print(f"{workspace.root}")
    for kind in kinds:
        members = summary[kind]
        print(f"\n{kind} ({len(members)})")
        for member in members:
            print(f"  {member}")
        if not members:
            print("  <none>")
    return 0


def _command_show(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    document = workspace.load_workflow_document(args.workflow)

    print(f"{workspace.workflow_path(args.workflow)}")
    print(f"schema_version: {document.get('schema_version', 1)}")

    agents = document.get("agents", [])
    if agents:
        print("\nagents")
        for agent in agents:
            provider = agent.get("provider", {})
            print(
                f"  {agent.get('name')}: {provider.get('type')} "
                f"{provider.get('model')}"
            )
    if document.get("default_agent"):
        print(f"  default: {document['default_agent']}")

    print("\nprocesses")
    for process in document.get("processes", []):
        print(
            f"  {process.get('process_type')}: {process.get('process_name', '<unnamed>')}"
        )

    referenced = sorted(_referenced_inputs(document))
    print("\ninputs referenced")
    for name in referenced or ["  <none found>"]:
        print(f"  {name}" if referenced else name)
    return 0


def _referenced_inputs(document: Any) -> set[str]:
    """Find ``{inputs/...}`` placeholders anywhere in a workflow document."""

    import re

    found: set[str] = set()
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, str):
            found.update(re.findall(r"\{inputs/([\w/]+)\}", node))
    return found


def _command_run(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    arguments = collect_arguments(args.arg, args.json, args.json_file)
    results = workspace.run_workflow(
        args.workflow,
        arguments=arguments,
        allow_writes=args.allow_writes,
        env_file=args.env_file,
    )

    if args.output:
        found, value = results.resolve_path(args.output)
        if not found:
            print(f"No such results path: {args.output}", file=sys.stderr)
            return 2
        print(value if isinstance(value, str) else json.dumps(value, indent=2, default=str))
        return 0

    if args.json_output:
        print(json.dumps(results.to_dict(), indent=2, default=str))
        return 0

    for key, value in results.outputs.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(text) > 200:
            text = text[:200] + f"... ({len(text)} chars)"
        print(f"{key}: {text}")
    return 0


def _tool_collection(args: argparse.Namespace) -> Any:
    """Resolve tools from either a module spec or a workspace plugin."""

    from ToolAgents.tool_adapters.cli import load_tools_from_spec

    if args.module and args.plugin:
        raise ValueError("Use either --module or --plugin, not both.")
    if args.module:
        return load_tools_from_spec(args.module)
    if args.plugin:
        workspace = _workspace(args)
        registry = workspace.build_tool_registry()
        tools = [
            tool
            for name in [args.plugin]
            for tool in registry.get_tools()
            if registry.reference_for_tool(tool)["plugin"] == name
        ]
        if not tools:
            known = ", ".join(p.stem for p in workspace.tool_files()) or "<none>"
            raise ValueError(
                f"Unknown workspace tool module: '{args.plugin}'. Available: {known}."
            )
        return tools
    raise ValueError("Pass --module or --plugin to say where the tools come from.")


def _command_tools(args: argparse.Namespace) -> int:
    from ToolAgents.tool_adapters.cli import _parse_json_argument, _print_tool_list
    from ToolAgents.tool_adapters.execution import execute_tool_by_name, find_tool, format_tool_result
    from ToolAgents.tool_adapters.schemas import function_tool_to_input_schema

    tools = _tool_collection(args)

    if args.tools_command == "list":
        _print_tool_list(tools)
        return 0

    tool = find_tool(tools, args.tool_name)
    if tool is None:
        print(f"Unknown tool: {args.tool_name}", file=sys.stderr)
        return 2

    if args.tools_command == "schema":
        print(json.dumps(function_tool_to_input_schema(tool), indent=2))
        return 0

    arguments = _parse_json_argument(args.json, args.json_file)
    print(format_tool_result(execute_tool_by_name(tools, args.tool_name, arguments)))
    return 0


COMMANDS = {
    "init": _command_init,
    "list": _command_list,
    "show": _command_show,
    "run": _command_run,
    "tools": _command_tools,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``tool-agents`` command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse rejects this first
        parser.error(f"Unsupported command: {args.command}")
        return 2

    try:
        return handler(args)
    except WorkspaceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

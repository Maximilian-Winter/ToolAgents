"""The ``.tool-agents`` project folder.

A workspace is a directory a team commits alongside their code, holding the
workflows they run and everything those workflows need::

    .tool-agents/
      workflows/        *.json   pipeline documents, named by file stem
      tools/            *.py     modules whose tools become plugins
      prompts/          *.md     reusable prompt text
      providers/        *.json   shared agent and endpoint declarations
      adapter/
        input/          *.py     modules registering custom source types
        output/         *.py     modules registering custom sink types

Nothing here is a new concept for a workflow author. Tools arrive through the
existing :class:`~ToolAgents.pipelines.PipelineToolRegistry`, providers through
the existing ``agents`` block, adapters through the existing source and sink
registries, and prompts through a ``prompts`` results section addressed the
same way as every other section::

    {"system_message": "{prompts/reviewer}"}

The folder is found by walking up from the working directory, the way ``git``
finds ``.git``, so a workflow can be run from anywhere inside a project.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ToolAgents.pipelines import (
    AgentConfig,
    Pipeline,
    PipelineResults,
    PipelineToolRegistry,
)
from ToolAgents.pipelines.agent_config import load_env_file
from ToolAgents.tool_adapters.execution import normalize_tools

__all__ = [
    "DEFAULT_WORKSPACE_DIRNAME",
    "Workspace",
    "WorkspaceError",
    "WorkspaceNotFoundError",
]

#: Directory name searched for when walking up from the working directory.
DEFAULT_WORKSPACE_DIRNAME = ".tool-agents"

#: Subdirectories a workspace is scaffolded with.
WORKSPACE_SUBDIRS = (
    "workflows",
    "tools",
    "prompts",
    "providers",
    "adapter/input",
    "adapter/output",
)

#: Conventional env file inside a workspace, loaded before agents are built.
WORKSPACE_ENV_FILE = ".env"

#: Extensions treated as prompt text.
PROMPT_SUFFIXES = (".md", ".txt", ".prompt")


class WorkspaceError(RuntimeError):
    """Raised when a workspace is malformed or a member is missing."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when no workspace directory could be found."""


def _import_path(path: Path, prefix: str) -> Any:
    """Import a standalone ``.py`` file under a unique module name."""

    module_name = f"_toolagents_ws_{prefix}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkspaceError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a module that imports itself, or that
    # relies on being in sys.modules during import, behaves normally.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise WorkspaceError(f"Failed to import {path}: {exc}") from exc
    return module


def _collect_tools(module: Any, path: Path) -> list[Any]:
    """Return the tools a workspace tool module exposes.

    A module may define ``TOOLS``, or a zero-argument ``create_tools()``. If it
    defines neither, every public ``FunctionTool`` or tool-shaped object bound
    at module level is collected, so a small file needs no boilerplate.
    """

    from ToolAgents.function_tool import FunctionTool, ToolRegistry

    if hasattr(module, "TOOLS"):
        return normalize_tools(getattr(module, "TOOLS"))
    if callable(getattr(module, "create_tools", None)):
        return normalize_tools(module.create_tools())

    found = [
        value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and isinstance(value, (FunctionTool, ToolRegistry))
    ]
    if not found:
        raise WorkspaceError(
            f"Tool module {path.name} exposes no tools. Define TOOLS, a "
            "create_tools() function, or bind FunctionTool objects at module "
            "level."
        )
    return normalize_tools(found)


@dataclass
class Workspace:
    """A loaded ``.tool-agents`` directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.is_dir():
            raise WorkspaceNotFoundError(f"Not a workspace directory: {self.root}")

    # -- discovery ---------------------------------------------------------

    @classmethod
    def discover(
        cls,
        start: Path | str | None = None,
        dirname: str = DEFAULT_WORKSPACE_DIRNAME,
    ) -> "Workspace":
        """Find the nearest workspace at or above ``start``.

        Raises:
            WorkspaceNotFoundError: if no such directory exists, naming every
                directory that was searched so the failure is diagnosable.
        """

        current = Path(start or Path.cwd()).resolve()
        searched = []
        for candidate in [current, *current.parents]:
            searched.append(str(candidate))
            target = candidate / dirname
            if target.is_dir():
                return cls(target)
        raise WorkspaceNotFoundError(
            f"No {dirname} directory found in {searched[0]} or any parent. "
            f"Run 'tool-agents init' to create one."
        )

    @classmethod
    def create(
        cls,
        parent: Path | str | None = None,
        dirname: str = DEFAULT_WORKSPACE_DIRNAME,
    ) -> "Workspace":
        """Create an empty workspace under ``parent`` and return it."""

        root = Path(parent or Path.cwd()) / dirname
        for subdir in WORKSPACE_SUBDIRS:
            (root / subdir).mkdir(parents=True, exist_ok=True)
        return cls(root)

    # -- members -----------------------------------------------------------

    def directory(self, name: str) -> Path:
        """Return a subdirectory path, whether or not it exists."""

        return self.root / name

    def _files(self, subdir: str, suffixes: Sequence[str]) -> list[Path]:
        folder = self.directory(subdir)
        if not folder.is_dir():
            return []
        return sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix in suffixes
        )

    def workflow_files(self) -> list[Path]:
        """Return every workflow document, sorted by name."""

        return self._files("workflows", (".json",))

    def workflow_names(self) -> list[str]:
        """Return the names workflows are referred to by."""

        return [p.stem for p in self.workflow_files()]

    def workflow_path(self, name: str) -> Path:
        """Return the path of one workflow, by name or filename."""

        for path in self.workflow_files():
            if name in (path.stem, path.name):
                return path
        known = ", ".join(self.workflow_names()) or "<none>"
        raise WorkspaceError(f"Unknown workflow: '{name}'. Available: {known}.")

    def tool_files(self) -> list[Path]:
        """Return every tool module."""

        return self._files("tools", (".py",))

    def prompt_files(self) -> list[Path]:
        """Return every prompt file."""

        return self._files("prompts", PROMPT_SUFFIXES)

    def provider_files(self) -> list[Path]:
        """Return every provider declaration file."""

        return self._files("providers", (".json",))

    def adapter_files(self) -> list[Path]:
        """Return every adapter module, inputs before outputs."""

        return self._files("adapter/input", (".py",)) + self._files(
            "adapter/output", (".py",)
        )

    # -- loading -----------------------------------------------------------

    def load_prompts(self) -> dict[str, str]:
        """Return prompt text keyed by file stem."""

        prompts: dict[str, str] = {}
        for path in self.prompt_files():
            if path.stem in prompts:
                raise WorkspaceError(
                    f"Two prompt files are both named '{path.stem}'. Prompt "
                    "names come from the file stem, so they must be unique."
                )
            prompts[path.stem] = path.read_text(encoding="utf-8").strip()
        return prompts

    def load_adapters(self) -> list[str]:
        """Import adapter modules so they register their types.

        Returns the module names imported, for reporting.
        """

        imported = []
        for path in self.adapter_files():
            kind = path.parent.name
            _import_path(path, f"adapter_{kind}")
            imported.append(f"{kind}/{path.stem}")
        return imported

    def build_tool_registry(self) -> PipelineToolRegistry:
        """Register every tool module as a plugin named for its file."""

        registry = PipelineToolRegistry()
        for path in self.tool_files():
            module = _import_path(path, "tools")
            registry.register_plugin(path.stem, _collect_tools(module, path))
        return registry

    def load_agent_configs(self) -> tuple[list[dict[str, Any]], str | None]:
        """Return shared agent declarations and an optional default name.

        A provider file may hold a list of agent declarations, or an object
        with ``agents`` and optionally ``default_agent``.
        """

        agents: list[dict[str, Any]] = []
        seen: set[str] = set()
        default_agent: str | None = None

        for path in self.provider_files():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise WorkspaceError(f"{path.name} is not valid JSON: {exc}") from exc

            if isinstance(document, Mapping):
                declared = document.get("agents", [])
                if document.get("default_agent") and default_agent is None:
                    default_agent = str(document["default_agent"])
            elif isinstance(document, list):
                declared = document
            else:
                raise WorkspaceError(
                    f"{path.name} must hold a list of agents, or an object "
                    "with an 'agents' key."
                )

            for entry in declared:
                # Validate eagerly so a broken declaration is reported against
                # the file it came from, not against whichever workflow used it.
                config = AgentConfig.from_dict(entry)
                if config.name in seen:
                    raise WorkspaceError(
                        f"Agent '{config.name}' is declared twice across "
                        "providers/."
                    )
                seen.add(config.name)
                agents.append(config.to_dict())

        return agents, default_agent

    def load_env(self, env_file: str | Path | None = None) -> bool:
        """Read a ``.env`` file so provider keys can come from one.

        With no argument this reads ``.tool-agents/.env`` if it exists, which
        is where a project's keys naturally live: beside the workflows that
        need them, and easy to gitignore as one path. Values already exported
        in the environment are left alone.

        Returns:
            bool: Whether a file was read.
        """

        if env_file is not None:
            return load_env_file(env_file, required=True)
        return load_env_file(self.root / WORKSPACE_ENV_FILE, required=False)

    # -- running -----------------------------------------------------------

    def load_workflow_document(self, name: str) -> dict[str, Any]:
        """Read one workflow document, merged with shared provider entries.

        A workflow's own ``agents`` entries win over the shared ones, so a
        workflow can override a shared declaration by redeclaring the name.
        """

        path = self.workflow_path(name)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkspaceError(f"{path.name} is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise WorkspaceError(f"{path.name} must hold a JSON object.")

        shared, shared_default = self.load_agent_configs()
        if shared:
            own = list(document.get("agents", []))
            own_names = {str(entry.get("name")) for entry in own}
            document["agents"] = own + [
                entry for entry in shared if entry["name"] not in own_names
            ]
        if shared_default and not document.get("default_agent"):
            document["default_agent"] = shared_default
        return document

    def load_pipeline(
        self,
        name: str,
        allow_writes: bool = False,
        build_agents: bool = True,
        default_agent: Any = None,
        env_file: str | Path | None = None,
    ) -> Pipeline:
        """Load one workflow with the workspace's tools, adapters and env."""

        if build_agents:
            self.load_env(env_file)
        self.load_adapters()
        document = self.load_workflow_document(name)
        return Pipeline.from_dict(
            document,
            tool_registry=self.build_tool_registry(),
            load_tool_plugins=False,
            build_agents=build_agents,
            allow_writes=allow_writes,
            default_agent=default_agent,
        )

    def run_workflow(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        allow_writes: bool = False,
        build_agents: bool = True,
        default_agent: Any = None,
        env_file: str | Path | None = None,
    ) -> PipelineResults:
        """Load and run one workflow, seeding the ``prompts`` section."""

        pipeline = self.load_pipeline(
            name,
            allow_writes=allow_writes,
            build_agents=build_agents,
            default_agent=default_agent,
            env_file=env_file,
        )
        results = PipelineResults(inputs=dict(arguments or {}))
        prompts = self.load_prompts()
        if prompts:
            results.section("prompts", create=True).update(prompts)
        return pipeline.run(results)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, list[str]]:
        """Return what this workspace contains, for ``tool-agents list``."""

        return {
            "workflows": self.workflow_names(),
            "tools": [p.stem for p in self.tool_files()],
            "prompts": [p.stem for p in self.prompt_files()],
            "providers": [p.stem for p in self.provider_files()],
            "adapters": [f"{p.parent.name}/{p.stem}" for p in self.adapter_files()],
        }

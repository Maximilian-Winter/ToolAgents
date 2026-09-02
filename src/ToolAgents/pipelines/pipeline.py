from __future__ import annotations

import abc
import importlib
import inspect
import json
import logging
from dataclasses import dataclass, field, replace as dataclass_replace
from os import PathLike
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel

from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.agents.base_llm_agent import BaseToolAgent
from ToolAgents.utilities.message_template import MessageTemplate
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.tool_adapters.execution import normalize_tools, tool_name
from ToolAgents.pipelines.results import PipelineResults
from ToolAgents.pipelines.agent_config import (
    AgentConfig,
    LazyAgentRegistry,
    build_agents_from_configs,
)


#: Version written by :meth:`Pipeline.to_dict`.
#:
#: Version 2 adds flow-control processes (conditional/loop/map/parallel), the
#: optional ``agents`` block, and per-step/per-process ``agent`` references.
#: It is a strict superset of version 1, so version 1 documents still load.
#: Progress is reported here rather than printed, so a library caller can
#: route it and the CLI can show it. A long step otherwise looks like a hang.
logger = logging.getLogger("ToolAgents.pipelines")

PIPELINE_SCHEMA_VERSION = 2

#: Schema versions this module can read.
SUPPORTED_PIPELINE_SCHEMA_VERSIONS = frozenset({1, 2})


class PipelineSerializationError(ValueError):
    """Raised when a pipeline cannot be serialized or restored."""


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline fails while running, not while loading.

    Kept distinct from :class:`PipelineSerializationError` so that code
    wrapping ``from_dict`` and code wrapping ``run_pipeline`` can catch the
    failures that actually belong to each.
    """


@dataclass(frozen=True)
class PipelineToolPlugin:
    """A named source of tools that can be referenced from pipeline JSON.

    The optional ``source`` uses ``"module:attribute"`` syntax. The attribute can
    be an iterable of tools, a ``ToolRegistry``, a Pydantic model class, or a
    zero-argument factory returning any of those values.
    """

    name: str
    tools: Sequence[FunctionTool]
    source: str | None = None

    @classmethod
    def from_spec(cls, name: str, source: str) -> "PipelineToolPlugin":
        """Load a plugin from a ``"module:attribute"`` tool source."""

        return cls(name=name, tools=load_pipeline_tools_from_spec(source), source=source)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible plugin declaration."""

        if self.source is None:
            raise PipelineSerializationError(
                f"Tool plugin '{self.name}' cannot be written to JSON because it "
                "was registered without a module source."
            )
        return {"name": self.name, "source": self.source}


class PipelineToolRegistry:
    """Resolve pipeline JSON tool references to live ``FunctionTool`` objects."""

    def __init__(self) -> None:
        self._plugins: dict[str, PipelineToolPlugin] = {}
        self._tools: dict[tuple[str, str], FunctionTool] = {}

    def register_plugin(
        self,
        name: str,
        tools: Iterable[FunctionTool | type[BaseModel]] | ToolRegistry,
        source: str | None = None,
    ) -> "PipelineToolRegistry":
        """Register tools under a plugin name and optional import source."""

        if not name:
            raise ValueError("Plugin name cannot be empty.")
        normalized_tools = normalize_tools(tools)
        plugin = PipelineToolPlugin(name=name, tools=normalized_tools, source=source)
        self._plugins[name] = plugin

        for key in list(self._tools):
            if key[0] == name:
                del self._tools[key]

        for tool in normalized_tools:
            key = (name, tool_name(tool))
            if key in self._tools:
                raise ValueError(f"Duplicate tool reference: {name}.{tool_name(tool)}")
            self._tools[key] = tool

        return self

    def load_plugin(self, name: str, source: str) -> "PipelineToolRegistry":
        """Import and register tools from a ``"module:attribute"`` source."""

        plugin = PipelineToolPlugin.from_spec(name, source)
        return self.register_plugin(name, plugin.tools, source=source)

    def load_plugins(
        self, plugin_configs: Iterable[Mapping[str, Any]]
    ) -> "PipelineToolRegistry":
        """Load plugin declarations from pipeline JSON metadata."""

        for plugin_config in plugin_configs:
            try:
                name = plugin_config["name"]
                source = plugin_config["source"]
            except KeyError as exc:
                raise PipelineSerializationError(
                    f"Tool plugin config is missing required field: {exc.args[0]}"
                ) from exc
            self.load_plugin(str(name), str(source))
        return self

    def resolve_tool(self, reference: Mapping[str, Any] | str) -> FunctionTool:
        """Resolve a serialized tool reference."""

        plugin_name, resolved_tool_name = parse_tool_reference(reference)
        if plugin_name is not None:
            tool = self._tools.get((plugin_name, resolved_tool_name))
            if tool is None:
                raise PipelineSerializationError(
                    f"Unknown tool reference: {plugin_name}.{resolved_tool_name}"
                )
            return tool

        matches = [
            tool
            for (_, registered_tool_name), tool in self._tools.items()
            if registered_tool_name == resolved_tool_name
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise PipelineSerializationError(
                f"Unknown tool reference: {resolved_tool_name}"
            )
        raise PipelineSerializationError(
            f"Tool reference '{resolved_tool_name}' is ambiguous; include a plugin name."
        )

    def reference_for_tool(self, tool: FunctionTool) -> dict[str, str]:
        """Return the JSON reference for a registered tool."""

        for (plugin_name, registered_tool_name), registered_tool in self._tools.items():
            if registered_tool is tool:
                return {"plugin": plugin_name, "tool_name": registered_tool_name}

        matches = [
            (plugin_name, registered_tool_name)
            for (plugin_name, registered_tool_name), registered_tool in self._tools.items()
            if tool_name(registered_tool) == tool_name(tool)
        ]
        if len(matches) == 1:
            plugin_name, registered_tool_name = matches[0]
            return {"plugin": plugin_name, "tool_name": registered_tool_name}

        raise PipelineSerializationError(
            f"Tool '{tool_name(tool)}' is not registered in the pipeline tool registry."
        )

    def to_plugin_configs(self) -> list[dict[str, str]]:
        """Return JSON-compatible plugin declarations for importable plugins."""

        return [
            plugin.to_dict()
            for plugin in self._plugins.values()
            if plugin.source is not None
        ]

    def get_tool(self, plugin_name: str, tool_name: str) -> FunctionTool | None:
        """Return a registered tool by plugin and public tool name."""

        return self._tools.get((plugin_name, tool_name))

    def get_tools(self) -> list[FunctionTool]:
        """Return all registered tools."""

        return list(self._tools.values())


def load_pipeline_tools_from_spec(spec: str) -> list[FunctionTool]:
    """Load tools from ``"module:attribute"`` for pipeline plugins."""

    if ":" not in spec:
        raise ValueError("Tool plugin source must use the form 'module:attribute'.")

    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)

    if (inspect.isfunction(value) or inspect.ismethod(value)) and not isinstance(
        value, ToolRegistry
    ):
        value = value()
    elif isinstance(value, FunctionTool):
        value = [value]
    elif isinstance(value, type) and issubclass(value, BaseModel):
        value = [value]

    return normalize_tools(value)


def parse_tool_reference(reference: Mapping[str, Any] | str) -> tuple[str | None, str]:
    """Parse a serialized tool reference into ``(plugin_name, tool_name)``."""

    if isinstance(reference, str):
        if "." in reference:
            plugin_name, parsed_tool_name = reference.split(".", 1)
            return plugin_name, parsed_tool_name
        return None, reference

    parsed_tool_name = reference.get("tool_name", reference.get("name"))
    if not parsed_tool_name:
        raise PipelineSerializationError(
            "Tool references must include 'tool_name' or 'name'."
        )
    plugin_name = reference.get("plugin")
    return str(plugin_name) if plugin_name else None, str(parsed_tool_name)


def _explain_empty_response(process_result: Any) -> str:
    """Say why a step produced nothing, so the cause is actionable.

    An empty response is nearly always a failure, but it is invisible: it is
    stored like any other value, and a condition reading it just evaluates
    false. A reasoning model that exhausts its token budget while thinking is
    the common cause, and the fix is specific enough to name.
    """

    messages = getattr(process_result, "messages", None) or []
    for message in reversed(messages):
        if getattr(message, "contains_reasoning", None) and message.contains_reasoning():
            return (
                "The model produced reasoning but no answer, which usually "
                "means max_tokens was consumed while thinking. Raise it, "
                "disable reasoning, or use a non-reasoning model."
            )
    return (
        "The model returned an empty message. Check the model name, the "
        "prompt, and any max_tokens limit."
    )


# ---------------------------------------------------------------------------
# Load context
# ---------------------------------------------------------------------------


@dataclass
class PipelineLoadContext:
    """Everything a process needs in order to rebuild itself from JSON.

    Flow-control processes contain other processes, so deserialization is
    recursive. Rather than thread five separate keyword arguments down every
    level, the loader carries this single context and hands it to each child.

    Agent resolution follows one rule: **an agent injected from Python wins
    over a name declared in JSON at the same level of specificity, and a more
    specific source wins over a less specific one.** In descending priority:

    1. ``step_agents`` entry for this step
    2. the step's own JSON ``agent`` name
    3. ``process_agents`` entry for this process
    4. the process's own JSON ``agent`` name
    5. the agent resolved for an enclosing flow-control process
    6. ``default_agent`` passed from Python
    7. the JSON ``default_agent`` name
    """

    tool_registry: "PipelineToolRegistry | None" = None
    default_agent: BaseToolAgent | None = None
    process_agents: Mapping[str, BaseToolAgent] = field(default_factory=dict)
    step_agents: Mapping[str, BaseToolAgent] = field(default_factory=dict)
    load_tool_plugins: bool = True

    #: Agents built from the JSON ``agents`` block, keyed by declared name.
    named_agents: Mapping[str, BaseToolAgent] = field(default_factory=dict)

    #: Name from the JSON ``default_agent`` field, if any.
    json_default_agent_name: str | None = None

    #: Whether sinks loaded from this document may write files or make
    #: requests. Off by default: reading a document should not let it reach
    #: outside the process without the caller saying so.
    allow_writes: bool = False

    #: When true, agent names declared in the JSON are ignored rather than
    #: resolved. Set by ``build_agents=False``, whose whole purpose is to
    #: ignore the ``agents`` block and take every agent from Python instead.
    ignore_agent_names: bool = False

    #: Names of the enclosing processes, outermost first. Used so a step
    #: nested inside flow control can still be addressed unambiguously.
    process_path: tuple[str, ...] = ()

    #: Agent resolved for the enclosing process, if any. A child inherits this
    #: ahead of ``default_agent``: an agent injected for a loop is meant for
    #: everything inside the loop, not just the loop object itself.
    parent_agent: BaseToolAgent | None = None

    def nested(
        self,
        process_name: str,
        agent: BaseToolAgent | None = None,
    ) -> "PipelineLoadContext":
        """Return a copy of this context scoped inside ``process_name``."""

        return dataclass_replace(
            self,
            process_path=self.process_path + (process_name,),
            parent_agent=agent if agent is not None else self.parent_agent,
        )

    # -- agent lookup ------------------------------------------------------

    def named_agent(self, name: str | None, *, referenced_by: str) -> BaseToolAgent | None:
        """Resolve a JSON agent reference, or ``None`` when ``name`` is None."""

        if name is None or self.ignore_agent_names:
            return None
        agent = self.named_agents.get(name)
        if agent is None:
            known = ", ".join(sorted(self.named_agents)) or "<none declared>"
            raise PipelineSerializationError(
                f"{referenced_by} references unknown agent '{name}'. "
                f"Declared agents: {known}."
            )
        return agent

    def agent_for_process(
        self,
        process_name: str,
        agent_name: str | None = None,
    ) -> BaseToolAgent | None:
        """Return the agent a process should use, honouring the priority rule."""

        injected = self._lookup(self.process_agents, process_name)
        if injected is not None:
            return injected

        declared = self.named_agent(
            agent_name, referenced_by=f"Process '{process_name}'"
        )
        if declared is not None:
            return declared

        # An agent resolved for an enclosing flow-control process outranks the
        # pipeline-wide default; otherwise process_agents={"refine": big_model}
        # would apply to the loop object while every step inside it quietly ran
        # on default_agent.
        if self.parent_agent is not None:
            return self.parent_agent

        if self.default_agent is not None:
            return self.default_agent

        return self.named_agent(
            self.json_default_agent_name, referenced_by="Pipeline 'default_agent'"
        )

    def agent_for_step(
        self,
        process_name: str,
        step_name: str,
        agent_name: str | None = None,
    ) -> BaseToolAgent | None:
        """Return the agent a step should use, or ``None`` to inherit.

        ``None`` means "no step-specific agent"; the process-level agent is
        used at run time instead.
        """

        injected = self._lookup(self.step_agents, f"{process_name}.{step_name}", step_name)
        if injected is not None:
            return injected

        return self.named_agent(
            agent_name,
            referenced_by=f"Step '{process_name}.{step_name}'",
        )

    def _lookup(
        self,
        mapping: Mapping[str, BaseToolAgent],
        *keys: str,
    ) -> BaseToolAgent | None:
        """Look up the first matching key, trying the full nested path first."""

        if not mapping:
            return None
        for key in keys:
            if self.process_path:
                qualified = ".".join(self.process_path + (key,))
                if qualified in mapping:
                    return mapping[qualified]
            if key in mapping:
                return mapping[key]
        return None


# ---------------------------------------------------------------------------
# Process type registry
# ---------------------------------------------------------------------------

_PROCESS_TYPES: dict[str, type["Process"]] = {}

#: Modules searched for process registrations when a type is not yet known.
#: This keeps ``pipeline`` free of an import cycle with ``flow``.
_PROCESS_TYPE_MODULES = (
    "ToolAgents.pipelines.flow",
    "ToolAgents.pipelines.data_io",
)


def register_process_type(process_cls: type["Process"]) -> type["Process"]:
    """Register a ``Process`` subclass so pipeline JSON can dispatch to it.

    Usable as a decorator. The class must define a non-empty ``process_type``.
    """

    process_type = getattr(process_cls, "process_type", "")
    if not process_type:
        raise ValueError(
            f"{process_cls.__name__} must define a non-empty 'process_type' "
            "to be registered."
        )
    _PROCESS_TYPES[process_type] = process_cls
    return process_cls


def get_process_type(process_type: str) -> type["Process"]:
    """Return the registered ``Process`` subclass for ``process_type``."""

    if process_type not in _PROCESS_TYPES:
        # Built-in flow-control processes live in a sibling module that imports
        # this one, so they are registered lazily on first miss.
        for module_name in _PROCESS_TYPE_MODULES:
            try:
                importlib.import_module(module_name)
            except ImportError as exc:  # pragma: no cover - defensive
                # Tolerate the module being absent from a trimmed install, but
                # never swallow a genuine broken import *inside* it: that would
                # surface as a baffling "unsupported process type" instead.
                if getattr(exc, "name", None) != module_name:
                    raise
                continue

    process_cls = _PROCESS_TYPES.get(process_type)
    if process_cls is None:
        known = ", ".join(sorted(_PROCESS_TYPES)) or "<none>"
        raise PipelineSerializationError(
            f"Unsupported process type: {process_type}. Known types: {known}."
        )
    return process_cls


def process_from_dict(
    data: Mapping[str, Any],
    context: PipelineLoadContext,
) -> "Process":
    """Rebuild a single process from its JSON representation."""

    if not isinstance(data, Mapping):
        raise PipelineSerializationError(
            f"Process config must be an object, got {type(data).__name__}."
        )
    process_type = data.get("process_type", data.get("type"))
    if process_type is None:
        raise PipelineSerializationError("Process config is missing 'process_type'.")
    return get_process_type(str(process_type)).from_dict(data, context)


def processes_from_config(
    configs: Any,
    context: PipelineLoadContext,
    *,
    field_name: str,
) -> list["Process"]:
    """Rebuild a list of processes from a JSON branch/body field.

    A single process object is accepted in place of a one-element list, since
    that reads better for the common ``"else"`` branch.
    """

    if configs is None:
        return []
    if isinstance(configs, Mapping):
        configs = [configs]
    if not isinstance(configs, Sequence) or isinstance(configs, (str, bytes)):
        raise PipelineSerializationError(
            f"'{field_name}' must be a process object or a list of them, "
            f"got {type(configs).__name__}."
        )
    return [process_from_dict(config, context) for config in configs]


def processes_to_config(
    processes: Sequence["Process"],
    tool_registry: "PipelineToolRegistry | None",
) -> list[dict[str, Any]]:
    """Serialize a list of child processes."""

    return [process.to_dict(tool_registry=tool_registry) for process in processes]


class ProcessStep:
    """
    Represents a single step in a process pipeline for LLM tool usage.

    Each step contains the necessary configuration for the LLM to perform
    a specific task, including system message, prompt template, and available tools.

    Attributes:
        step_name (str): The name identifier for the step.
            Can be used to reference results of previous steps in the prompt
            template, for example ``{outputs/step_name}``.
        system_message (str): The system message to provide context to the LLM
        prompt_template (str): Template string for generating the actual prompt
        tools (list[FunctionTool]): List of tools available for this step
        agent (BaseToolAgent): The LLM agent responsible for executing this step
    """

    def __init__(
        self,
        step_name: str,
        system_message: str,
        prompt_template: str,
        tools: list[FunctionTool] = None,
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
    ):
        """
        Initialize a new process step.

        Args:
            step_name: Unique identifier for the step. Its result is written
                to the ``outputs`` section, so a later prompt can reference it
                as ``{outputs/step_name}``.
            system_message: Context message for the LLM
            prompt_template: Template for generating the actual prompt
            tools: Optional list of tools available for this step
            agent: Optional specific agent for this step
            agent_name: Optional name of an agent declared in the pipeline's
                ``agents`` block. The name is what round-trips to JSON; the
                ``agent`` object is what actually runs.
        """
        self.step_name = step_name
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.tools = tools or []
        self.agent = agent
        self.agent_name = agent_name

    def get_name(self) -> str:
        """Return the step name."""
        return self.step_name

    def get_system_message(
        self,
        fields: "PipelineResults | Mapping[str, Any] | None" = None,
    ) -> str:
        """Return the system message, with any placeholders filled in.

        A system message is a prompt as much as ``prompt_template`` is, so it
        is rendered against the same results. That is what lets a shared
        prompt file be addressed as ``{prompts/reviewer}`` in either field.

        Args:
            fields: A results mapping. Omit it to get the raw template.
        """
        if fields is None:
            return self.system_message
        return MessageTemplate.from_string(
            self.system_message
        ).generate_message_content(fields)

    def get_prompt(
        self,
        fields: "PipelineResults | Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate the actual prompt using the template and provided parameters.

        Args:
            fields: A results mapping. Passing the mapping itself, rather than
                unpacking it, is what lets a template address a section:
                ``{outputs/draft}`` as well as a bare ``{draft}``.
            **kwargs: Individual template fields, for callers not using a
                results mapping.

        Returns:
            str: The generated prompt
        """
        msg = MessageTemplate.from_string(self.prompt_template)
        if fields is not None and not kwargs:
            return msg.generate_message_content(fields)
        return msg.generate_message_content(fields, **kwargs)

    def get_tools(self) -> list[FunctionTool]:
        """Return the list of tools available for this step."""
        return self.tools

    def get_agent(self) -> BaseToolAgent:
        """Return the agent assigned to this step."""
        return self.agent

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this step to a JSON-compatible dictionary.

        Agents are runtime objects and are intentionally not serialized.
        """

        data: dict[str, Any] = {
            "step_name": self.step_name,
            "system_message": self.system_message,
            "prompt_template": self.prompt_template,
        }
        if self.tools:
            if tool_registry is None:
                raise PipelineSerializationError(
                    f"Step '{self.step_name}' has tools, so a PipelineToolRegistry "
                    "is required to serialize it."
                )
            data["tools"] = [
                tool_registry.reference_for_tool(tool) for tool in self.tools
            ]
        if self.agent_name:
            data["agent"] = self.agent_name
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        tool_registry: PipelineToolRegistry | None = None,
        agent: BaseToolAgent | None = None,
    ) -> "ProcessStep":
        """Restore a process step from a JSON-compatible dictionary."""

        tool_refs = data.get("tools", [])
        if tool_refs and tool_registry is None:
            raise PipelineSerializationError(
                f"Step '{data.get('step_name', data.get('name', '<unknown>'))}' "
                "references tools, so a PipelineToolRegistry is required."
            )
        tools = [
            tool_registry.resolve_tool(tool_ref) for tool_ref in tool_refs
        ] if tool_refs else []

        step_name = data.get("step_name", data.get("name"))
        if step_name is None:
            raise PipelineSerializationError("Step config is missing 'step_name'.")

        agent_name = data.get("agent")

        return cls(
            step_name=str(step_name),
            system_message=str(data["system_message"]),
            prompt_template=str(data["prompt_template"]),
            tools=tools,
            agent=agent,
            agent_name=str(agent_name) if agent_name else None,
        )


class Process(abc.ABC):
    """
    Abstract base class representing a process in the pipeline.

    A process is a collection of steps that need to be executed in a specific way.
    The actual execution logic is defined by concrete implementations.
    """

    #: Value written to, and dispatched on, the JSON ``process_type`` field.
    #: Subclasses set this and pass themselves to ``register_process_type``.
    process_type: str = ""

    def __init__(
        self,
        process_name: str = "Process",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
    ):
        """
        Initialize a new process.

        Args:
            process_name: Name identifier for the process
            agent: Default agent to use for steps that don't have their own agent
            agent_name: Optional name of an agent declared in the pipeline's
                ``agents`` block. The name round-trips to JSON; the ``agent``
                object is what actually runs.
        """
        self.process_name = process_name
        self.agent = agent
        self.agent_name = agent_name
        self.steps: list[ProcessStep] = []

    def add_step(self, step: ProcessStep) -> "Process":
        """Add a new step to the process and return self, for chaining."""
        self.steps.append(step)
        return self

    def add_steps(self, steps: list[ProcessStep]) -> "Process":
        """Add new steps to the process and return self, for chaining."""
        self.steps.extend(steps)
        return self

    @abc.abstractmethod
    def run_process(self, results: PipelineResults) -> PipelineResults:
        """
        Execute the process steps according to the implementation logic.

        Args:
            results: Sectioned results carried through the pipeline

        Returns:
            PipelineResults: Updated results after process execution
        """
        pass

    def get_name(self) -> str:
        """Return the process name."""
        return self.process_name

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this process to a JSON-compatible dictionary."""

        raise PipelineSerializationError(
            f"Process type '{type(self).__name__}' does not support serialization."
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "Process":
        """Rebuild this process from JSON, using ``context`` for agents/tools.

        Flow-control processes call :func:`processes_from_config` with the same
        context (or a nested one) to rebuild their children.
        """

        raise PipelineSerializationError(
            f"Process type '{cls.__name__}' does not support deserialization."
        )

    # -- helpers for subclasses -------------------------------------------

    def run_child_processes(
        self,
        processes: Sequence["Process"],
        results: PipelineResults,
    ) -> PipelineResults:
        """Run child processes in order, threading the results mapping.

        A child with no agent of its own inherits this process's agent, so a
        loop or branch built in Python needs the agent set only once, at the
        outermost level that has one.
        """

        for process in processes:
            self.lend_agent(process)
            results = process.run_process(results)
        return results

    def lend_agent(self, process: "Process") -> None:
        """Give ``process`` this process's agent if it has none of its own."""

        if process.agent is None and self.agent is not None:
            process.agent = self.agent


class Pipeline:
    """
    Main pipeline class that manages the execution of multiple processes.

    The pipeline maintains a list of processes and executes them in sequence,
    passing results between processes.
    """

    def __init__(
        self,
        agent_configs: Sequence["AgentConfig"] | None = None,
        default_agent_name: str | None = None,
    ):
        """Initialize an empty pipeline.

        Args:
            agent_configs: Optional declarative agent/endpoint configurations.
                These round-trip through JSON; the agents themselves are built
                at load time from environment-held API keys.
            default_agent_name: Name of the declared agent used by processes
                that name none.
        """
        self.processes: list[Process] = []
        self.agent_configs: list["AgentConfig"] = list(agent_configs or [])
        self.default_agent_name = default_agent_name

    def add_agent_config(self, agent_config: "AgentConfig") -> "Pipeline":
        """Declare an agent/endpoint that processes can reference by name."""

        self.agent_configs.append(agent_config)
        return self

    def add_process(self, process: Process):
        """Add a new process to the pipeline."""
        self.processes.append(process)

    def add_processes(self, processes: list[Process]):
        """Add new processes to the pipeline."""
        self.processes.extend(processes)

    def run_pipeline(self, **kwargs) -> PipelineResults:
        """
        Execute all processes in the pipeline sequentially.

        Results from each process are passed as input to the next process.

        Keyword arguments become the ``inputs`` section; step results land in
        ``outputs``. The returned object still reads like the flat dictionary
        it replaced — ``results["greeting"]`` resolves by scope order — so
        existing calling code is unaffected.
        """
        return self.run(PipelineResults(inputs=kwargs))

    def run(self, results: PipelineResults) -> PipelineResults:
        """Execute every process against an existing results object.

        ``run_pipeline`` is the usual entry point; this one is for a caller
        that needs to seed a section other than ``inputs`` -- shared prompt
        text, for instance -- before the run begins.
        """

        for process in self.processes:
            results = process.run_process(results)
        return results

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
        include_tool_plugins: bool = True,
    ) -> dict[str, Any]:
        """Serialize this pipeline to a JSON-compatible dictionary.

        Runtime agents are intentionally omitted. Pass agents back into
        ``from_dict``/``load_from_json`` when restoring a runnable pipeline.
        """

        data: dict[str, Any] = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "processes": [
                process.to_dict(tool_registry=tool_registry)
                for process in self.processes
            ],
        }
        if self.agent_configs:
            data["agents"] = [config.to_dict() for config in self.agent_configs]
        if self.default_agent_name:
            data["default_agent"] = self.default_agent_name
        if include_tool_plugins and tool_registry is not None:
            data["tool_plugins"] = tool_registry.to_plugin_configs()
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        tool_registry: PipelineToolRegistry | None = None,
        default_agent: BaseToolAgent | None = None,
        process_agents: Mapping[str, BaseToolAgent] | None = None,
        step_agents: Mapping[str, BaseToolAgent] | None = None,
        load_tool_plugins: bool = True,
        build_agents: bool = True,
        allow_writes: bool = False,
    ) -> "Pipeline":
        """Restore a pipeline from a JSON-compatible dictionary.

        If ``load_tool_plugins`` is true, plugin declarations in the JSON are
        imported with Python's import machinery. Only load JSON files that you
        trust, or pass a prebuilt ``tool_registry`` and disable plugin loading.

        If ``build_agents`` is true, any ``agents`` block in the JSON is used
        to construct providers, reading API keys from the environment
        variables the config names. Pass ``build_agents=False`` to ignore the
        block entirely and supply every agent from Python instead.

        ``allow_writes`` gates sinks that write files or make HTTP requests.
        It is false by default: loading a document should not let it reach
        outside the process unless the caller says so. Sources that only read
        are always permitted.

        Agents injected here always win over names declared in the JSON at the
        same level of specificity; see :class:`PipelineLoadContext`.
        """

        schema_version = data.get("schema_version", 1)
        if schema_version not in SUPPORTED_PIPELINE_SCHEMA_VERSIONS:
            supported = ", ".join(
                str(version) for version in sorted(SUPPORTED_PIPELINE_SCHEMA_VERSIONS)
            )
            raise PipelineSerializationError(
                f"Unsupported pipeline schema version: {schema_version}. "
                f"Supported versions: {supported}."
            )

        resolved_tool_registry = tool_registry
        plugin_configs = data.get("tool_plugins", [])
        if plugin_configs:
            if resolved_tool_registry is None:
                resolved_tool_registry = PipelineToolRegistry()
            if load_tool_plugins:
                resolved_tool_registry.load_plugins(plugin_configs)

        agent_configs = [
            AgentConfig.from_dict(agent_data)
            for agent_data in data.get("agents", [])
        ]
        named_agents = (
            LazyAgentRegistry(agent_configs) if build_agents else {}
        )

        json_default_agent_name = data.get("default_agent")
        json_default_agent_name = (
            str(json_default_agent_name) if json_default_agent_name else None
        )

        context = PipelineLoadContext(
            tool_registry=resolved_tool_registry,
            default_agent=default_agent,
            process_agents=dict(process_agents or {}),
            step_agents=dict(step_agents or {}),
            load_tool_plugins=load_tool_plugins,
            named_agents=named_agents,
            json_default_agent_name=json_default_agent_name,
            ignore_agent_names=not build_agents,
            allow_writes=allow_writes,
        )

        pipeline = cls(
            agent_configs=agent_configs,
            default_agent_name=json_default_agent_name,
        )
        for process_data in data.get("processes", []):
            pipeline.add_process(process_from_dict(process_data, context))

        return pipeline

    def to_json(
        self,
        tool_registry: PipelineToolRegistry | None = None,
        include_tool_plugins: bool = True,
        indent: int | None = 2,
    ) -> str:
        """Serialize this pipeline to a JSON string."""

        return json.dumps(
            self.to_dict(
                tool_registry=tool_registry,
                include_tool_plugins=include_tool_plugins,
            ),
            indent=indent,
        )

    @classmethod
    def from_json(
        cls,
        json_text: str,
        tool_registry: PipelineToolRegistry | None = None,
        default_agent: BaseToolAgent | None = None,
        process_agents: Mapping[str, BaseToolAgent] | None = None,
        step_agents: Mapping[str, BaseToolAgent] | None = None,
        load_tool_plugins: bool = True,
        build_agents: bool = True,
        allow_writes: bool = False,
    ) -> "Pipeline":
        """Restore a pipeline from a JSON string."""

        return cls.from_dict(
            json.loads(json_text),
            tool_registry=tool_registry,
            default_agent=default_agent,
            process_agents=process_agents,
            step_agents=step_agents,
            load_tool_plugins=load_tool_plugins,
            build_agents=build_agents,
            allow_writes=allow_writes,
        )

    def save_to_json(
        self,
        filepath: str | PathLike[str],
        tool_registry: PipelineToolRegistry | None = None,
        include_tool_plugins: bool = True,
        indent: int | None = 2,
    ) -> None:
        """Write this pipeline to a JSON file."""

        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(
                self.to_json(
                    tool_registry=tool_registry,
                    include_tool_plugins=include_tool_plugins,
                    indent=indent,
                )
            )

    @classmethod
    def load_from_json(
        cls,
        filepath: str | PathLike[str],
        tool_registry: PipelineToolRegistry | None = None,
        default_agent: BaseToolAgent | None = None,
        process_agents: Mapping[str, BaseToolAgent] | None = None,
        step_agents: Mapping[str, BaseToolAgent] | None = None,
        load_tool_plugins: bool = True,
        build_agents: bool = True,
        allow_writes: bool = False,
    ) -> "Pipeline":
        """Load a pipeline from a JSON file."""

        with open(filepath, "r", encoding="utf-8") as handle:
            return cls.from_json(
                handle.read(),
                tool_registry=tool_registry,
                default_agent=default_agent,
                process_agents=process_agents,
                step_agents=step_agents,
                load_tool_plugins=load_tool_plugins,
                build_agents=build_agents,
                allow_writes=allow_writes,
            )


@register_process_type
class SequentialProcess(Process):
    """
    Concrete implementation of Process that executes steps sequentially.

    Each step is executed in order, with results from previous steps available
    to subsequent steps through the results dictionary.
    """

    process_type = "sequential"

    def __init__(
        self,
        process_name: str = "SequentialProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
    ):
        """Initialize a sequential process with optional default agent."""
        Process.__init__(self, process_name, agent, agent_name)

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this sequential process to a JSON-compatible dictionary."""

        data: dict[str, Any] = {
            "process_type": self.process_type,
            "process_name": self.process_name,
            "steps": [
                step.to_dict(tool_registry=tool_registry) for step in self.steps
            ],
        }
        if self.agent_name:
            data["agent"] = self.agent_name
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "SequentialProcess":
        """Restore a sequential process from its JSON representation."""

        process_name_value = data.get("process_name", data.get("name"))
        if process_name_value is None:
            raise PipelineSerializationError(
                "Process config is missing 'process_name'."
            )
        process_name = str(process_name_value)
        agent_name = data.get("agent")
        agent_name = str(agent_name) if agent_name else None

        process = cls(
            process_name=process_name,
            agent=context.agent_for_process(process_name, agent_name),
            agent_name=agent_name,
        )

        for step_data in data.get("steps", []):
            step_name = str(step_data.get("step_name", step_data.get("name")))
            step_agent_name = step_data.get("agent")
            process.add_step(
                ProcessStep.from_dict(
                    step_data,
                    tool_registry=context.tool_registry,
                    agent=context.agent_for_step(
                        process_name,
                        step_name,
                        str(step_agent_name) if step_agent_name else None,
                    ),
                )
            )
        return process

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """
        Execute process steps in sequential order.

        For each step:
        1. Set up tool registry if tools are available
        2. Prepare messages with system message and generated prompt
        3. Execute step using appropriate agent
        4. Store results in results dictionary

        Args:
            results: Dictionary containing results from previous processes

        Returns:
            dict[str, Any]: Updated results dictionary after all steps are executed

        Raises:
            Exception: If no agent is available for a step
        """
        for step in self.steps:
            # Initialize tool registry if step has tools
            tool_registry = ToolRegistry() if step.get_tools() else None

            # Prepare messages for the step
            messages = [
                ChatMessage.create_system_message(
                    step.get_system_message(results)
                ),
                ChatMessage.create_user_message(step.get_prompt(results)),
            ]

            # Add tools to registry if available
            if tool_registry is not None:
                tool_registry.add_tools(step.get_tools())

            logger.info("%s / %s: calling model", self.process_name, step.step_name)

            # Execute step with appropriate agent
            if step.get_agent() is not None:
                # Use step-specific agent if available
                process_result = step.get_agent().get_response(
                    messages=messages, tool_registry=tool_registry
                )
            else:
                if self.agent is not None:
                    # Fall back to process-level agent
                    process_result = self.agent.get_response(
                        messages=messages, tool_registry=tool_registry
                    )
                else:
                    # No agent available
                    raise Exception(
                        f"No agent defined for process '{self.process_name}', "
                        f"step:{step.step_name}"
                    )

            response_text = process_result.response or ""
            logger.info(
                "%s / %s: %d characters",
                self.process_name,
                step.step_name,
                len(response_text),
            )
            if not response_text.strip():
                logger.warning(
                    "%s / %s returned no text. %s",
                    self.process_name,
                    step.step_name,
                    _explain_empty_response(process_result),
                )

            # Store step results in the outputs section, so a step can never
            # shadow a caller argument or a flow-control variable.
            results.outputs[step.step_name] = process_result.response

        return results

from __future__ import annotations

import abc
import importlib
import inspect
import json
from dataclasses import dataclass
from os import PathLike
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel

from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.agents.base_llm_agent import BaseToolAgent
from ToolAgents.utilities.message_template import MessageTemplate
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.tool_adapters.execution import normalize_tools, tool_name


PIPELINE_SCHEMA_VERSION = 1


class PipelineSerializationError(ValueError):
    """Raised when a pipeline cannot be serialized or restored."""


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


def _agent_for_step(
    process_name: str,
    step_name: str,
    step_agents: Mapping[str, BaseToolAgent] | None,
) -> BaseToolAgent | None:
    if step_agents is None:
        return None
    return step_agents.get(f"{process_name}.{step_name}", step_agents.get(step_name))


class ProcessStep:
    """
    Represents a single step in a process pipeline for LLM tool usage.

    Each step contains the necessary configuration for the LLM to perform
    a specific task, including system message, prompt template, and available tools.

    Attributes:
        step_name (str): The name identifier for the step.
        Can be used to reference results of previous steps in the prompt template. Example: '{step_name}'
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
    ):
        """
        Initialize a new process step.

        Args:
            step_name: Unique identifier for the step.
            Can be used to reference results of previous steps in the prompt template. Example: '{step_name}'
            system_message: Context message for the LLM
            prompt_template: Template for generating the actual prompt
            tools: Optional list of tools available for this step
            agent: Optional specific agent for this step
        """
        self.step_name = step_name
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.tools = tools or []
        self.agent = agent

    def get_name(self) -> str:
        """Return the step name."""
        return self.step_name

    def get_system_message(self) -> str:
        """Return the system message for this step."""
        return self.system_message

    def get_prompt(self, **kwargs) -> str:
        """
        Generate the actual prompt using the template and provided parameters.

        Args:
            **kwargs: Keyword arguments to fill in the prompt template

        Returns:
            str: The generated prompt
        """
        msg = MessageTemplate.from_string(self.prompt_template)
        return msg.generate_message_content(**kwargs)

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

        return cls(
            step_name=str(step_name),
            system_message=str(data["system_message"]),
            prompt_template=str(data["prompt_template"]),
            tools=tools,
            agent=agent,
        )


class Process(abc.ABC):
    """
    Abstract base class representing a process in the pipeline.

    A process is a collection of steps that need to be executed in a specific way.
    The actual execution logic is defined by concrete implementations.
    """

    def __init__(self, process_name: str = "Process", agent: BaseToolAgent = None):
        """
        Initialize a new process.

        Args:
            process_name: Name identifier for the process
            agent: Default agent to use for steps that don't have their own agent
        """
        self.process_name = process_name
        self.agent = agent
        self.steps: list[ProcessStep] = []

    def add_step(self, step: ProcessStep):
        """Add a new step to the process."""
        self.steps.append(step)

    def add_steps(self, steps: list[ProcessStep]):
        """Add new steps to the process."""
        self.steps.extend(steps)

    @abc.abstractmethod
    def run_process(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the process steps according to the implementation logic.

        Args:
            results: Dictionary containing results from previous processes

        Returns:
            dict[str, Any]: Updated results dictionary after process execution
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


class Pipeline:
    """
    Main pipeline class that manages the execution of multiple processes.

    The pipeline maintains a list of processes and executes them in sequence,
    passing results between processes.
    """

    def __init__(self):
        """Initialize an empty pipeline."""
        self.processes = []

    def add_process(self, process: Process):
        """Add a new process to the pipeline."""
        self.processes.append(process)

    def add_processes(self, processes: list[Process]):
        """Add new processes to the pipeline."""
        self.processes.extend(processes)

    def run_pipeline(self, **kwargs) -> dict[str, Any]:
        """
        Execute all processes in the pipeline sequentially.

        Results from each process are passed as input to the next process.
        """
        results = kwargs
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
    ) -> "Pipeline":
        """Restore a pipeline from a JSON-compatible dictionary.

        If ``load_tool_plugins`` is true, plugin declarations in the JSON are
        imported with Python's import machinery. Only load JSON files that you
        trust, or pass a prebuilt ``tool_registry`` and disable plugin loading.
        """

        schema_version = data.get("schema_version", 1)
        if schema_version != PIPELINE_SCHEMA_VERSION:
            raise PipelineSerializationError(
                f"Unsupported pipeline schema version: {schema_version}"
            )

        resolved_tool_registry = tool_registry
        plugin_configs = data.get("tool_plugins", [])
        if plugin_configs:
            if resolved_tool_registry is None:
                resolved_tool_registry = PipelineToolRegistry()
            if load_tool_plugins:
                resolved_tool_registry.load_plugins(plugin_configs)

        pipeline = cls()
        for process_data in data.get("processes", []):
            process_type = process_data.get("process_type", process_data.get("type"))
            if process_type != "sequential":
                raise PipelineSerializationError(
                    f"Unsupported process type: {process_type}"
                )
            process_name_value = process_data.get(
                "process_name",
                process_data.get("name"),
            )
            if process_name_value is None:
                raise PipelineSerializationError(
                    "Process config is missing 'process_name'."
                )
            process_name = str(process_name_value)
            process_agent = (
                process_agents.get(process_name)
                if process_agents and process_name in process_agents
                else default_agent
            )
            process = SequentialProcess(
                process_name=process_name,
                agent=process_agent,
            )
            for step_data in process_data.get("steps", []):
                step_name = str(step_data.get("step_name", step_data.get("name")))
                step = ProcessStep.from_dict(
                    step_data,
                    tool_registry=resolved_tool_registry,
                    agent=_agent_for_step(process_name, step_name, step_agents),
                )
                process.add_step(step)
            pipeline.add_process(process)

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
    ) -> "Pipeline":
        """Restore a pipeline from a JSON string."""

        return cls.from_dict(
            json.loads(json_text),
            tool_registry=tool_registry,
            default_agent=default_agent,
            process_agents=process_agents,
            step_agents=step_agents,
            load_tool_plugins=load_tool_plugins,
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
            )


class SequentialProcess(Process):
    """
    Concrete implementation of Process that executes steps sequentially.

    Each step is executed in order, with results from previous steps available
    to subsequent steps through the results dictionary.
    """

    def __init__(
        self, process_name: str = "SequentialProcess", agent: BaseToolAgent = None
    ):
        """Initialize a sequential process with optional default agent."""
        Process.__init__(self, process_name, agent)

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this sequential process to a JSON-compatible dictionary."""

        return {
            "process_type": "sequential",
            "process_name": self.process_name,
            "steps": [
                step.to_dict(tool_registry=tool_registry) for step in self.steps
            ],
        }

    def run_process(self, results: dict[str, Any]) -> dict[str, Any]:
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
                ChatMessage.create_system_message(step.get_system_message()),
                ChatMessage.create_user_message(step.get_prompt(**results)),
            ]

            # Add tools to registry if available
            if tool_registry is not None:
                tool_registry.add_tools(step.get_tools())

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

            # Store step results
            results[step.step_name] = process_result.response

        return results

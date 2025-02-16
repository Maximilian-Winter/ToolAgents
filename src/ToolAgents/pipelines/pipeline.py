import abc
from typing import Any

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.agents.base_llm_agent import BaseToolAgent
from ToolAgents.messages.message_template import MessageTemplate
from ToolAgents.messages import ChatMessage


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

    def __init__(self, step_name: str, system_message: str, prompt_template: str,
                 tools: list[FunctionTool] = None, agent: BaseToolAgent = None):
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
        self.tools = tools
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


class SequentialProcess(Process):
    """
    Concrete implementation of Process that executes steps sequentially.

    Each step is executed in order, with results from previous steps available
    to subsequent steps through the results dictionary.
    """

    def __init__(self, process_name: str = "SequentialProcess", agent: BaseToolAgent = None):
        """Initialize a sequential process with optional default agent."""
        Process.__init__(self, process_name, agent)

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
            messages = [ChatMessage.create_system_message( step.get_system_message()), ChatMessage.create_user_message(step.get_prompt(**results))]

            # Add tools to registry if available
            if tool_registry is not None:
                tool_registry.add_tools(step.get_tools())

            # Execute step with appropriate agent
            if step.get_agent() is not None:
                # Use step-specific agent if available
                process_result = step.get_agent().get_response(
                    messages=messages,
                    tool_registry=tool_registry
                )
            else:
                if self.agent is not None:
                    # Fall back to process-level agent
                    process_result = self.agent.get_response(
                        messages=messages,
                        tool_registry=tool_registry
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
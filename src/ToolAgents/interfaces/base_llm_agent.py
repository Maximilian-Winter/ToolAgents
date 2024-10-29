from abc import ABC, abstractmethod
from typing import Any, List, Dict, Generator, Tuple, Optional, Union
from ToolAgents import FunctionTool
from ToolAgents.function_tool import ToolRegistry


class BaseToolAgent(ABC):
    """
    Abstract base class defining the common interface for chat API and hosted tool agents.
    """

    def __init__(self, debug_output: bool = False):
        self.debug_output = debug_output
        self.last_messages_buffer: List[Dict[str, Any]] = []

    @abstractmethod
    def step(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[Union[List[FunctionTool], ToolRegistry]] = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Tuple[Any, bool]:
        """
        Performs a single step of interaction with the model.

        Args:
            messages: List of conversation messages
            tools: Available tools/functions for the agent
            settings: Model-specific settings
            reset_last_messages_buffer: Whether to clear message buffer

        Returns:
            Tuple of (response, contains_tool_call)
        """
        pass

    @abstractmethod
    def stream_step(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[Union[List[FunctionTool], ToolRegistry]] = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[Tuple[str, bool], None, None]:
        """
        Performs a streaming step of interaction with the model.

        Args:
            messages: List of conversation messages
            tools: Available tools/functions for the agent
            settings: Model-specific settings
            reset_last_messages_buffer: Whether to clear message buffer

        Yields:
            Tuples of (chunk, contains_tool_call)
        """
        pass

    @abstractmethod
    def handle_function_calling_response(
            self,
            tool_calls: Any,
            current_messages: List[Dict[str, Any]]
    ) -> None:
        """
        Handles tool/function calls and updates messages accordingly.

        Args:
            tool_calls: The tool calls to process
            current_messages: Current message history to update
        """
        pass

    def get_response(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[Union[List[FunctionTool], ToolRegistry]] = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> str:
        """
        Gets a complete response from the model, handling any tool calls.

        Args:
            messages: List of conversation messages
            tools: Available tools/functions for the agent
            settings: Model-specific settings
            reset_last_messages_buffer: Whether to clear message buffer

        Returns:
            The model's final response as a string
        """
        result, contains_tool_call = self.step(messages, tools, settings, reset_last_messages_buffer)

        if contains_tool_call:
            self.handle_function_calling_response(result, messages)
            return self.get_response(
                messages=messages,
                tools=tools,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": result})
            return result

    def get_streaming_response(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[Union[List[FunctionTool], ToolRegistry]] = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[str, None, None]:
        """
        Gets a streaming response from the model, handling any tool calls.

        Args:
            messages: List of conversation messages
            tools: Available tools/functions for the agent
            settings: Model-specific settings
            reset_last_messages_buffer: Whether to clear message buffer

        Yields:
            Response chunks as strings
        """
        result = ""
        tool_calls = None

        for chunk, contains_tool_call in self.stream_step(
                messages=messages,
                tools=tools,
                settings=settings,
                reset_last_messages_buffer=reset_last_messages_buffer
        ):
            if contains_tool_call:
                tool_calls = chunk
            else:
                result += chunk
                yield chunk

        if tool_calls is not None:
            self.handle_function_calling_response(tool_calls, messages)
            yield "\n"
            yield from self.get_streaming_response(
                messages=messages,
                tools=tools,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": result})

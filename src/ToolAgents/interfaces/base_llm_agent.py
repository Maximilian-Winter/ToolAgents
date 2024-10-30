import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Tuple

from ToolAgents import ToolRegistry


class BaseToolAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Tuple[Any, bool]:
        """
        Performs a single step of interaction with the chat API, returning the result
        and whether it contains tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            Tuple of (result, contains_tool_calls)
        """
        pass

    @abstractmethod
    def stream_step(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ):
        """
        Performs a single streaming step of interaction with the chat API,
        yielding chunks and whether they contain tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Tuples of (chunk, contains_tool_calls)
        """
        pass

    @abstractmethod
    def get_response(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> str:
        """
        Gets a complete response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            The final response string
        """
        pass

    @abstractmethod
    def get_streaming_response(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ):
        """
        Gets a streaming response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Response chunks
        """
        pass

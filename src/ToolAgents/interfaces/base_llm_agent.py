import json
from abc import ABC, abstractmethod

from typing import Optional, Dict, List, Any, Tuple, Generator

from ToolAgents import ToolRegistry
from ToolAgents.interfaces import SamplingSettings
from ToolAgents.interfaces.llm_provider import StreamingChatAPIResponse
from ToolAgents.messages.chat_message import ChatMessage


class BaseToolAgent(ABC):
    def __init__(self):
        self.last_messages_buffer = []

    @abstractmethod
    def get_default_settings(self) -> SamplingSettings:
        pass

    @abstractmethod
    def step(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> ChatMessage:
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
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[StreamingChatAPIResponse, None, None]:
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
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> ChatMessage:
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
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[StreamingChatAPIResponse, None, None]:
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


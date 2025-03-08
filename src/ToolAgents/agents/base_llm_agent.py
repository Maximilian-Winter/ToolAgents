import json
from abc import ABC, abstractmethod
from types import NoneType

from typing import Optional, List, Any, Generator, AsyncGenerator, Union

from pydantic import BaseModel, Field

from ToolAgents import ToolRegistry
from ToolAgents.data_models.messages import ToolCallContent, ToolCallResultContent
from ToolAgents.provider.llm_provider import ProviderSettings
from ToolAgents.provider.llm_provider import StreamingChatMessage
from ToolAgents.data_models.messages import ChatMessage
from data_models.responses import ChatResponse, ChatResponseChunk


class BaseToolAgent(ABC):
    def __init__(self):
        self.last_messages_buffer = []

    @abstractmethod
    def get_default_settings(self) -> ProviderSettings:
        pass

    @abstractmethod
    def step(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> ChatMessage:
        """
        Performs a single step of interaction with the agent, returning the chat message of the agent  .

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            Chat Message (ChatMessage)
        """
        pass

    @abstractmethod
    def stream_step(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> Generator[StreamingChatMessage, None, None]:
        """
        Performs a single streaming step of interaction with the agent, yielding chunks.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Chunks of chat api responses (StreamingChatAPIResponse)
        """
        pass

    @abstractmethod
    def get_response(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> ChatResponse:
        """
        Gets a complete response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            The final chat response from the agent.(ChatResponse)
        """
        pass

    @abstractmethod
    def get_streaming_response(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> Generator[ChatResponseChunk, None, None]:
        """
        Gets a streaming response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Response chunks(ChatResponseChunk)
        """
        pass

    @abstractmethod
    def get_last_response(self) -> ChatResponse:
        """
        Returns the last response from the agent.
        Returns:
            Last response(ChatResponse)
        """
        pass


class AsyncBaseToolAgent(ABC):
    def __init__(self):
        self.last_messages_buffer = []

    @abstractmethod
    def get_default_settings(self) -> ProviderSettings:
        pass

    @abstractmethod
    async def step(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> ChatMessage:
        """
        Performs a single step of interaction with the agent, returning the chat message of the agent  .

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            Chat Message (ChatMessage)
        """
        pass

    @abstractmethod
    async def stream_step(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> Generator[StreamingChatMessage, None, None]:
        """
        Performs a single streaming step of interaction with the agent, yielding chunks.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Chunks of chat api responses (StreamingChatAPIResponse)
        """
        pass

    @abstractmethod
    async def get_response(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> ChatResponse:
        """
        Gets a complete response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            The final chat response from the agent.(ChatResponse)
        """
        pass

    @abstractmethod
    async def get_streaming_response(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[ProviderSettings] = None,
        reset_last_messages_buffer: bool = True,
    ) -> AsyncGenerator[ChatResponseChunk, None]:
        """
        Gets a streaming response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Response chunks(ChatResponseChunk)
        """
        pass

    @abstractmethod
    def get_last_response(self) -> ChatResponse:
        """
        Returns the last response from the agent.
        Returns:
            Last response(ChatResponse)
        """
        pass


class AgentObservabilityHandler(ABC):

    @abstractmethod
    def on_request(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry,
        settings: Optional[ProviderSettings],
        reset_last_messages_buffer: bool,
        result_chat_message: ChatMessage,
    ):
        pass

    @abstractmethod
    def on_streaming_request(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry,
        settings: Optional[ProviderSettings],
        reset_last_messages_buffer: bool,
        result_chat_message: ChatMessage,
    ):
        pass

    @abstractmethod
    def on_tool_call(
        self, tool_call: ToolCallContent, tool_call_result: ToolCallResultContent
    ):
        pass

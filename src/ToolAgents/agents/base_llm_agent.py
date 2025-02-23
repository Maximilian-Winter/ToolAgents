from abc import ABC, abstractmethod

from typing import Optional, List, Any, Generator

from pydantic import BaseModel, Field

from ToolAgents import ToolRegistry
from ToolAgents.provider.llm_provider import ProviderSettings
from ToolAgents.provider.llm_provider import StreamingChatMessage
from ToolAgents.messages.chat_message import ChatMessage


class ChatResponse(BaseModel):
    """
    Represents an agent chat response.
    """
    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages.")
    response: str = Field(default_factory=str, description="Final response from the agent.")


class ChatResponseChunk(BaseModel):
    """
    Represents an agent chat response chunk.
    """
    chunk: str = Field("", description="Response chunk from the agent.")

    finished: bool = Field(default_factory=bool, description="Whether the response has been completed.")
    finished_response: ChatResponse = Field(default_factory=ChatResponse, description="Finished response object from the agent.")


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
            settings: Optional[Any] = None,
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
            settings: Optional[Any] = None,
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
            settings: Optional[Any] = None,
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
            settings: Optional[Any] = None,
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
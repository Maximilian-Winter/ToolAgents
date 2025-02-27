import json
from typing import List, Optional, Any, Generator

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.agents.base_llm_agent import ChatResponse, ChatResponseChunk
from ToolAgents.messages import ChatMessage
from ToolAgents.provider import ChatAPIProvider, StreamingChatMessage


class StructuredOutputAgent(ChatToolAgent):
    def __init__(self, chat_api: ChatAPIProvider):
        super().__init__(chat_api)
        self.last_received_response: str = ""

    def send_response(self, response: str):
        """
        Sends a response to the user.
        :arg response: The response to send.
        :return:
        """
        self.last_received_response = response

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
        if tool_registry is None:
            return super().step(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
                reset_last_messages_buffer=reset_last_messages_buffer,
            )
        if settings is None:
            settings = self.chat_api.get_default_settings()

        tool_registry.add_tool(FunctionTool(self.send_response))
        schema = tool_registry.get_guided_sampling_json_schema()
        settings.set_response_format({"type": "json_object", "schema": schema})
        tool_registry.remove("send_response")

        result = super().step(
            messages=messages,
            settings=settings,
            tool_registry=None,
            reset_last_messages_buffer=reset_last_messages_buffer,
        )

        return result

    def stream_step(
        self,
        messages: List[ChatMessage],
        tool_registry: ToolRegistry = None,
        settings: Optional[Any] = None,
        reset_last_messages_buffer: bool = True,
    ) -> Generator[StreamingChatMessage, None, None]:
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
        if tool_registry is None:
            return super().stream_step(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
                reset_last_messages_buffer=reset_last_messages_buffer,
            )
        if settings is None:
            settings = self.chat_api.get_default_settings()

        tool_registry.add_tool(FunctionTool(self.send_response))
        schema = tool_registry.get_guided_sampling_json_schema()
        settings.set_response_format({"type": "json_object", "schema": schema})
        tool_registry.remove("send_response")

        result = super().stream_step(
            messages=messages,
            settings=settings,
            tool_registry=None,
            reset_last_messages_buffer=reset_last_messages_buffer,
        )

        return result

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
            The final response string
        """
        if tool_registry is None:
            return super().get_response(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
                reset_last_messages_buffer=reset_last_messages_buffer,
            )
        if settings is None:
            settings = self.chat_api.get_default_settings()

        result = self.step(
            messages=messages,
            settings=settings,
            tool_registry=None,
            reset_last_messages_buffer=reset_last_messages_buffer,
        )

        json_string = result.get_as_text()
        json_object = json.loads(json_string)

        tool_name = json_object["tool"]
        if tool_name == "send_response":
            return ChatResponse(
                messages=self.last_messages_buffer, response=self.last_received_response
            )
        tool = self.tool_registry.get_tool(tool_name)
        if tool is not None:
            result = tool.execute(json_object["parameters"])

        return result

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
            Response chunks
        """
        if tool_registry is None:
            return super().get_streaming_response(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
                reset_last_messages_buffer=reset_last_messages_buffer,
            )
        if settings is None:
            settings = self.chat_api.get_default_settings()

        schema = tool_registry.get_guided_sampling_json_schema()
        settings.set_response_format({"type": "json_object", "schema": schema})

        result = super().get_streaming_response(
            messages=messages,
            settings=settings,
            tool_registry=None,
            reset_last_messages_buffer=reset_last_messages_buffer,
        )
        return result

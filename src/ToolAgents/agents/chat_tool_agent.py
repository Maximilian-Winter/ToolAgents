import datetime
import json
import uuid
from typing import Optional, Dict, List, Any, Tuple
from ToolAgents import ToolRegistry
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent
from ToolAgents.interfaces.llm_tool_call import generate_id
from ToolAgents.interfaces.llm_provider import ChatAPIProvider
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, ToolCallResultContent


class ChatToolAgent(BaseToolAgent):

    def __init__(self, chat_api: ChatAPIProvider, debug_output: bool = False):
        super().__init__()
        self.chat_api = chat_api
        self.debug_output = debug_output
        self.last_messages_buffer = []
        self.tool_registry = ToolRegistry()
        self.last_response_has_tool_calls = False

    def get_default_settings(self):
        return self.chat_api.get_default_settings()

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
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", '\n'.join([msg.model_dump_json(indent=4) for msg in messages]))

        result = self.chat_api.get_response(self.chat_api.convert_chat_messages(messages), settings=settings, tools=tools)

        return result


    def stream_step(
            self,
            messages: List[ChatMessage],
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
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(messages, indent=2))

        for chunk in self.chat_api.get_streaming_response(self.chat_api.convert_chat_messages(messages), settings=settings, tools=tools):
            yield chunk

    def handle_function_calling_response(
            self,
            chat_message: ChatMessage,
            current_messages: List[ChatMessage],
    ) -> None:
        """
        Handles the response containing function calls by executing tools and updating messages.

        Args:
            chat_message: chat message
            current_messages: List of current conversation messages
        """
        tool_calls = chat_message.get_tool_calls()
        content = []
        for tool_call in tool_calls:
            tool_name = tool_call.tool_call_name
            tool = self.tool_registry.get_tool(tool_name)

            if tool:
                call_parameters = tool_call.tool_call_arguments
                output = tool.execute(call_parameters)
                content.append(ToolCallResultContent(tool_call_result_id=str(uuid.uuid4()), tool_call_id=tool_call.tool_call_id, tool_call_name=tool_call.tool_call_name, tool_call_result=str(output)))
        tool_message = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Tool, content=content, created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        self.last_messages_buffer.append(chat_message)
        current_messages.append(chat_message)
        self.last_messages_buffer.append(tool_message)
        current_messages.append(tool_message)

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
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False
        result = self.step(messages, tool_registry, settings, reset_last_messages_buffer)

        if result.contains_tool_call():
            self.last_response_has_tool_calls = True
            self.handle_function_calling_response(result, messages)
            return self.get_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append(result)
            return result

    def get_streaming_response(
            self,
            messages: List[ChatMessage],
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
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False

        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        finished_message = None

        for chunk in self.stream_step(messages=messages, tool_registry=tool_registry, settings=settings, reset_last_messages_buffer=True):
            if chunk.get_finished():
                finished_message = chunk.get_finished_chat_message()
            yield chunk
        if finished_message.contains_tool_call():
            self.last_response_has_tool_calls = True
            self.handle_function_calling_response(finished_message, messages)
            yield "\n"
            yield from self.get_streaming_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append(finished_message)


import datetime
import uuid
from typing import Optional, List, Any, Generator, AsyncGenerator
from ToolAgents import ToolRegistry
from ToolAgents.agents.base_llm_agent import BaseToolAgent, ChatResponse, ChatResponseChunk, AsyncBaseToolAgent

from ToolAgents.provider.llm_provider import ChatAPIProvider, StreamingChatMessage, AsyncChatAPIProvider
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, ToolCallResultContent


class ChatToolAgent(BaseToolAgent):

    def __init__(self, chat_api: ChatAPIProvider, debug_output: bool = False):
        super().__init__()
        self.chat_api = chat_api
        self.debug_output = debug_output
        self.last_messages_buffer: list[ChatMessage] = []
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

        result = self.chat_api.get_response(messages, settings=settings, tools=tools)

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
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", '\n'.join([msg.model_dump_json(indent=4, exclude_none=True) for msg in messages]))

        for chunk in self.chat_api.get_streaming_response(messages, settings=settings, tools=tools):
            yield chunk

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
            return ChatResponse(messages=self.last_messages_buffer, response=result.get_as_text())

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
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False

        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        has_tool_call = False
        tool_call = None
        finished_message = None
        for chunk in self.stream_step(messages=messages, tool_registry=tool_registry, settings=settings,
                                      reset_last_messages_buffer=reset_last_messages_buffer):
            if chunk.get_finished():
                finished_message = chunk.get_finished_chat_message()
            yield ChatResponseChunk(chunk=chunk.chunk, has_tool_call=chunk.is_tool_call, tool_call=chunk.tool_call, finished=False)
            has_tool_call = chunk.is_tool_call
            tool_call = chunk.tool_call

        if finished_message.contains_tool_call():
            self.last_response_has_tool_calls = True
            self.handle_function_calling_response(finished_message, messages)
            tool_calls = messages[-2].get_tool_calls()
            tool_call_results = messages[-1].get_tool_call_results()
            for tool_call, tool_call_result in zip(tool_calls, tool_call_results):
                yield ChatResponseChunk(has_tool_call=True, tool_call=tool_call.model_dump(),
                                        has_tool_call_result=True,
                                        tool_call_result=tool_call_result.model_dump())


            yield from self.get_streaming_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append(finished_message)
            yield ChatResponseChunk(chunk="", finished=True, has_tool_call=has_tool_call, tool_call=tool_call,
                                    finished_response=ChatResponse(messages=self.last_messages_buffer,
                                                                   response=finished_message.get_as_text()))

    def get_last_response(self) -> ChatResponse:
        return ChatResponse(messages=self.last_messages_buffer,
                            response=self.last_messages_buffer[-1].get_as_text())

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
                content.append(
                    ToolCallResultContent(tool_call_result_id=str(uuid.uuid4()), tool_call_id=tool_call.tool_call_id,
                                          tool_call_name=tool_call.tool_call_name, tool_call_result=str(output)))
        tool_message = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Tool, content=content,
                                   created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        self.last_messages_buffer.append(chat_message)
        current_messages.append(chat_message)
        self.last_messages_buffer.append(tool_message)
        current_messages.append(tool_message)


class AsyncChatToolAgent(AsyncBaseToolAgent):

    def __init__(self, chat_api: AsyncChatAPIProvider, debug_output: bool = False):
        super().__init__()
        self.chat_api = chat_api
        self.debug_output = debug_output
        self.last_messages_buffer: list[ChatMessage] = []
        self.tool_registry = ToolRegistry()
        self.last_response_has_tool_calls = False

    def get_default_settings(self):
        return self.chat_api.get_default_settings()

    async def step(
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

        result = await self.chat_api.get_response(messages, settings=settings, tools=tools)

        return result

    async def stream_step(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> AsyncGenerator[StreamingChatMessage, None]:
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
            print("Input messages:", '\n'.join([msg.model_dump_json(indent=4, exclude_none=True) for msg in messages]))

        async for chunk in await self.chat_api.get_streaming_response(messages, settings=settings, tools=tools):
            yield chunk

    async def get_response(
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
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False
        result = await self.step(messages, tool_registry, settings, reset_last_messages_buffer)

        if result.contains_tool_call():
            self.last_response_has_tool_calls = True
            self.handle_function_calling_response(result, messages)
            return await self.get_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append(result)
            return ChatResponse(messages=self.last_messages_buffer, response=result.get_as_text())

    async def get_streaming_response(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
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
            Response chunks
        """
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False

        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry

        finished_message = None
        has_tool_call = False
        tool_call = None
        async for chunk in self.stream_step(messages=messages, tool_registry=tool_registry, settings=settings,
                                            reset_last_messages_buffer=reset_last_messages_buffer):
            if chunk.get_finished():
                finished_message = chunk.get_finished_chat_message()
            yield ChatResponseChunk(chunk=chunk.chunk, has_tool_call=chunk.is_tool_call, tool_call=chunk.tool_call, finished=chunk.get_finished())
            tool_call = chunk.tool_call
            has_tool_call = chunk.is_tool_call
        if finished_message.contains_tool_call():
            self.last_response_has_tool_calls = True
            self.handle_function_calling_response(finished_message, messages)
            tool_calls = messages[-2].get_tool_calls()
            tool_call_results = messages[-1].get_tool_call_results()
            for tool_call, tool_call_result in zip(tool_calls, tool_call_results):
                yield ChatResponseChunk(has_tool_call=True, tool_call=tool_call.model_dump(), has_tool_call_result=True,
                                        tool_call_result=tool_call_result.model_dump())

            async for chunk in self.get_streaming_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            ):
                yield chunk
        else:
            self.last_messages_buffer.append(finished_message)
            yield ChatResponseChunk(chunk="", finished=True, has_tool_call=has_tool_call, tool_call=tool_call,
                                    finished_response=ChatResponse(messages=self.last_messages_buffer,
                                                                   response=finished_message.get_as_text()))

    def get_last_response(self) -> ChatResponse:
        return ChatResponse(messages=self.last_messages_buffer,
                            response=self.last_messages_buffer[-1].get_as_text())

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
                content.append(
                    ToolCallResultContent(tool_call_result_id=str(uuid.uuid4()), tool_call_id=tool_call.tool_call_id,
                                          tool_call_name=tool_call.tool_call_name, tool_call_result=str(output)))
        date = datetime.datetime.now()
        tool_message = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Tool, content=content,
                                   created_at=date, updated_at=date)
        self.last_messages_buffer.append(chat_message)
        current_messages.append(chat_message)
        self.last_messages_buffer.append(tool_message)
        current_messages.append(tool_message)

import copy
import datetime
import uuid
from typing import Any, List, Optional, Dict, Generator

from ToolAgents import FunctionTool
from ToolAgents.messages import ChatMessage, ChatMessageRole, TextContent
from ToolAgents.provider.llm_provider import ProviderSettings, ChatAPIProvider, StreamingChatMessage, \
    AsyncChatAPIProvider
from .default_implementations import TemplateToolCallHandler, MistralMessageConverterLlamaCpp, MistralTokenizer
from .completion_interfaces import LLMTokenizer, LLMToolCallHandler, CompletionEndpoint, AsyncCompletionEndpoint
from ToolAgents.provider.message_converter.message_converter import BaseMessageConverter



class CompletionProvider(ChatAPIProvider):

    def __init__(self, completion_endpoint: CompletionEndpoint, tokenizer: LLMTokenizer = MistralTokenizer(), message_converter: BaseMessageConverter = MistralMessageConverterLlamaCpp(), tool_call_handler: LLMToolCallHandler = TemplateToolCallHandler()):
        self.tokenizer = tokenizer
        self.message_converter = message_converter
        self.completion_endpoint = completion_endpoint
        self.tool_call_handler = tool_call_handler
        self.default_settings = completion_endpoint.get_default_settings()

    def get_default_settings(self) -> ProviderSettings:
        return copy.copy(self.default_settings)

    def set_default_settings(self, settings: ProviderSettings):
        self.default_settings = settings

    def get_response(self, messages: List[ChatMessage], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        if settings is None:
            settings = self.get_default_settings()

        msg = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=[], created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        messages = self.message_converter.prepare_request("llama.cpp", messages, settings, tools)
        prompt = self.tokenizer.apply_template(messages=messages['messages'], tools=[tool.to_openai_tool() for tool in tools])

        result = self.completion_endpoint.create_completion(prompt, settings)
        if self.tool_call_handler.contains_tool_calls(result):
            tool_calls = self.tool_call_handler.parse_tool_calls(result)
            msg.content.extend(tool_calls)
        else:
            msg.add_text(result.replace(self.tokenizer.get_eos_token_string(), ""))
        return msg

    def get_streaming_response(self, messages: List[ChatMessage], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatMessage, None, None]:
        messages = self.message_converter.prepare_request("llama.cpp", messages, settings, tools)
        prompt = self.tokenizer.apply_template(messages=messages['messages'], tools=[tool.to_openai_tool() for tool in tools])
        if settings is None:
            settings = self.get_default_settings()

        # Get the streaming generator
        token_stream = self.completion_endpoint.create_streaming_completion(prompt, settings)
        complete_response = ""
        buffer = ""  # Buffer for potential tool call tokens
        is_in_tool_call = False
        eos_token = self.tokenizer.get_eos_token_string()

        # Create a sliding window of tokens
        token_window = []

        for token in token_stream:
            current_token = token.replace(eos_token, "")
            token_window.append(current_token)

            # Keep only last 3 tokens (current + 2 look-ahead)
            if len(token_window) > 3:
                token_window.pop(0)

            # If we don't have enough tokens for look-ahead yet, just buffer
            if len(token_window) < 3 and not is_in_tool_call:
                buffer += current_token
                continue

            # Check if current buffer + window would form a tool call
            test_sequence = complete_response + buffer + "".join(token_window)

            if self.tool_call_handler.contains_partial_tool_calls(test_sequence):
                # Potential tool call detected
                is_in_tool_call = True
                buffer += current_token
            elif is_in_tool_call:
                # Already in tool call mode, keep buffering
                buffer += current_token
            else:
                # If we have buffered tokens and no tool call detected, stream them
                if buffer:
                    chunk = StreamingChatMessage(chunk=buffer)
                    complete_response += buffer
                    buffer = ""
                    yield chunk

                # Stream current token
                chunk = StreamingChatMessage(chunk=current_token)
                complete_response += current_token
                yield chunk

        # Process any remaining buffered tokens
        if buffer:
            complete_response += buffer

        final_chunk = StreamingChatMessage(chunk="", finished=True)

        if self.tool_call_handler.contains_tool_calls(complete_response):
            final_chunk.is_tool_call = True
            final_chunk.tool_call = {}
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.Assistant,
                content=self.tool_call_handler.parse_tool_calls(complete_response),
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
        else:
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.Assistant,
                content=[TextContent(content=complete_response.replace(self.tokenizer.get_eos_token_string(), ""))],
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )

        final_chunk.finished_chat_message = msg
        yield final_chunk

    def get_provider_identifier(self) -> str:
        return "completion"


import copy
import uuid
import datetime
from typing import List, Optional, AsyncGenerator


class AsyncCompletionProvider(AsyncChatAPIProvider):
    def __init__(self,
                 completion_endpoint: AsyncCompletionEndpoint,
                 tokenizer: LLMTokenizer = MistralTokenizer(),
                 message_converter: BaseMessageConverter = MistralMessageConverterLlamaCpp(),
                 tool_call_handler: LLMToolCallHandler = TemplateToolCallHandler()):
        self.tokenizer = tokenizer
        self.message_converter = message_converter
        self.completion_endpoint = completion_endpoint
        self.tool_call_handler = tool_call_handler
        self.default_settings = completion_endpoint.get_default_settings()

    def get_default_settings(self) -> ProviderSettings:
        return copy.copy(self.default_settings)

    def set_default_settings(self, settings: ProviderSettings) -> None:
        self.default_settings = settings

    async def get_response(self,
                           messages: List[ChatMessage],
                           settings: Optional[ProviderSettings] = None,
                           tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        if settings is None:
            settings = self.get_default_settings()

        msg = ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

        messages = self.message_converter.prepare_request(
            "llama.cpp",
            messages,
            settings,
            tools
        )

        prompt = self.tokenizer.apply_template(
            messages=messages['messages'],
            tools=[tool.to_openai_tool() for tool in tools] if tools else None
        )

        result = await self.completion_endpoint.create_completion(prompt, settings)

        if self.tool_call_handler.contains_tool_calls(result):
            tool_calls = self.tool_call_handler.parse_tool_calls(result)
            msg.content.extend(tool_calls)
        else:
            msg.add_text(result.replace(self.tokenizer.get_eos_token_string(), ""))

        return msg

    async def get_streaming_response(self,
                                     messages: List[ChatMessage],
                                     settings: Optional[ProviderSettings] = None,
                                     tools: Optional[List[FunctionTool]] = None) -> AsyncGenerator[
        StreamingChatMessage, None]:
        if settings is None:
            settings = self.get_default_settings()

        messages = self.message_converter.prepare_request(
            "llama.cpp",
            messages,
            settings,
            tools
        )

        prompt = self.tokenizer.apply_template(
            messages=messages['messages'],
            tools=[tool.to_openai_tool() for tool in tools] if tools else None
        )

        # Get the streaming generator
        token_stream = self.completion_endpoint.create_streaming_completion(prompt, settings)
        return self._yield_from_response_stream(token_stream)


    async def _yield_from_response_stream(self, token_stream) -> AsyncGenerator[StreamingChatMessage, None]:
        # Create a sliding window of tokens
        complete_response = ""
        eos_token = self.tokenizer.get_eos_token_string()
        token_buffer = []
        async for token in token_stream:
            current_token = token.replace(eos_token, "")
            token_buffer.append(current_token)
            test_sequence = complete_response + ''.join(token_buffer)
            if not self.tool_call_handler.contains_partial_tool_calls(test_sequence) and len(token_buffer) > 1:
                chunk = StreamingChatMessage(chunk=''.join(token_buffer))
                yield chunk
            if len(token_buffer) > 1:
                complete_response += ''.join(token_buffer)
                token_buffer.clear()
        if len(token_buffer) == 1:
            complete_response += token_buffer[0]
        final_chunk = StreamingChatMessage(chunk="", finished=True)

        if self.tool_call_handler.contains_tool_calls(complete_response):
            final_chunk.is_tool_call = True
            final_chunk.tool_call = {}
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.Assistant,
                content=self.tool_call_handler.parse_tool_calls(complete_response),
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
        else:
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.Assistant,
                content=[TextContent(content=complete_response.replace(self.tokenizer.get_eos_token_string(), ""))],
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )

        final_chunk.finished_chat_message = msg
        yield final_chunk

    def get_provider_identifier(self) -> str:
        return "completion"
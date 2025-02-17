import abc
import copy
import datetime
import json
import uuid
from dataclasses import dataclass
from typing import Any, List, Union, Optional, Dict, Generator

import requests

from ToolAgents import FunctionTool
from ToolAgents.messages import ChatMessage, ChatMessageRole, TextContent
from ToolAgents.provider.llm_provider import SamplingSettings, ChatAPIProvider, StreamingChatAPIResponse
from .default_implementations import TemplateToolCallHandler, MistralMessageConverterLlamaCpp, MistralTokenizer
from .completion_interfaces import LLMTokenizer, LLMToolCallHandler, CompletionEndpoint
from ToolAgents.messages.message_converter.message_converter import BaseMessageConverter



class CompletionProvider(ChatAPIProvider):

    def __init__(self, completion_endpoint: CompletionEndpoint, tokenizer: LLMTokenizer = MistralTokenizer(), message_converter: BaseMessageConverter = MistralMessageConverterLlamaCpp(), tool_call_handler: LLMToolCallHandler = TemplateToolCallHandler()):
        self.tokenizer = tokenizer
        self.message_converter = message_converter
        self.completion_endpoint = completion_endpoint
        self.tool_call_handler = tool_call_handler
        self.default_settings = completion_endpoint.get_default_settings()

    def get_default_settings(self) -> SamplingSettings:
        return copy.copy(self.default_settings)

    def set_default_settings(self, settings: SamplingSettings):
        self.default_settings = settings

    def get_response(self, messages: List[Dict[str, Any]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        msg = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=[], created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        prompt = self.tokenizer.apply_template(messages=messages, tools=[tool.to_openai_tool() for tool in tools])
        if settings is None:
            settings = self.get_default_settings()
        settings.stream = False
        result = self.completion_endpoint.create_completion(prompt, settings)
        if self.tool_call_handler.contains_tool_calls(result):
            tool_calls = self.tool_call_handler.parse_tool_calls(result)
            msg.content.extend(tool_calls)
        else:
            msg.add_text(result.replace(self.tokenizer.get_eos_token_string(), ""))
        return msg

    def get_streaming_response(self, messages: List[Dict[str, Any]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatAPIResponse, None, None]:

        prompt = self.tokenizer.apply_template(messages=messages, tools=[tool.to_openai_tool() for tool in tools])
        if settings is None:
            settings = self.get_default_settings()
        settings.stream = True

        # Get the streaming generator
        token_stream = self.completion_endpoint.create_completion(prompt, settings)
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
                    chunk = StreamingChatAPIResponse(chunk=buffer)
                    complete_response += buffer
                    buffer = ""
                    yield chunk

                # Stream current token
                chunk = StreamingChatAPIResponse(chunk=current_token)
                complete_response += current_token
                yield chunk

        # Process any remaining buffered tokens
        if buffer:
            complete_response += buffer

        final_chunk = StreamingChatAPIResponse(chunk="", finished=True)

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


    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        return self.message_converter.to_provider_format(messages)

    def get_provider_identifier(self) -> str:
        return "completion"
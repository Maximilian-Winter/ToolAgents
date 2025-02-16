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
        settings.stream = True  # Ensure streaming is enabled

        # Get the complete token stream
        token_stream = list(self.completion_endpoint.create_completion(prompt, settings))
        complete_response = ""
        buffer = ""  # Buffer to hold tokens for look-ahead
        is_in_tool_call = False
        eos_token = self.tokenizer.get_eos_token_string()

        i = 0
        while i < len(token_stream):
            current_token = token_stream[i].replace(eos_token, "")

            # Look ahead at the next 2 tokens
            next_tokens = ""
            for j in range(1, 3):
                if i + j < len(token_stream):
                    next_tokens += token_stream[i + j].replace(eos_token, "")

            # Check if current buffer + next tokens would form a tool call
            test_sequence = complete_response + current_token + next_tokens

            if self.tool_call_handler.contains_partial_tool_calls(test_sequence):
                # We've detected a potential tool call, enter tool call mode
                is_in_tool_call = True
                buffer += current_token
            elif is_in_tool_call:
                # We're already in a tool call, keep buffering
                buffer += current_token
            else:
                # No tool call detected, safe to stream the token
                chunk = StreamingChatAPIResponse(chunk=current_token)
                complete_response += current_token
                yield chunk

            i += 1

        # Process the final response
        if buffer:
            complete_response += buffer

        final_chunk = StreamingChatAPIResponse(chunk="", finished=True)

        # Check if the complete response contains tool calls
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

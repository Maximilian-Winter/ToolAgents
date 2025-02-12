import copy
import datetime
import json
import uuid
from json import JSONDecodeError
from typing import List, Dict, Optional, Any, Generator

from anthropic import Anthropic
from anthropic.types import ToolUseBlock, TextBlock

from ToolAgents import FunctionTool
from ToolAgents.messages.message_converter.anthropic_message_converter import AnthropicMessageConverter, \
    AnthropicResponseConverter
from ToolAgents.provider.llm_provider import ChatAPIProvider, SamplingSettings, StreamingChatAPIResponse
from ToolAgents.messages.chat_message import ChatMessage, ToolCallContent, TextContent, ChatMessageRole, BinaryContent, \
    BinaryStorageType, ToolCallResultContent
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages



class AnthropicSettings(SamplingSettings):

    def __init__(self):
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 0
        self.max_tokens = 1024
        self.stop_sequences = []
        self.request_kwargs = None
        self.extra_body = None
        self.tool_choice = None

    def save_to_file(self, settings_file: str):
        with open(settings_file, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

    def load_from_file(self, settings_file: str):
        with open(settings_file, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)

    def as_dict(self):
        return copy.copy(self.__dict__)

    def set_stop_tokens(self, tokens: List[str]):
        pass

    def set_max_new_tokens(self, max_new_tokens: int):
        self.max_tokens = max_new_tokens

    def set(self, setting_key: str, setting_value: str):
        if hasattr(self, setting_key):
            setattr(self, setting_key, setting_value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{setting_key}'")

    def neutralize_sampler(self, sampler_name: str):
        if sampler_name == "temperature":
            self.temperature = 1.0
        elif sampler_name == "top_k":
            self.top_k = 0
        elif sampler_name == "top_p":
            self.top_p = 1.0

        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_k = 0
        self.top_p = 1.0

    def set_response_format(self, response_format: dict[str, Any]):
        pass

    def set_extra_body(self, extra_body: dict[str, Any]):
        self.extra_body = extra_body

    def set_extra_request_kwargs(self, **kwargs):
        self.request_kwargs = kwargs

    def set_tool_choice(self, tool_choice: str):
        self.tool_choice = tool_choice

def prepare_messages(messages: List[Dict[str, str]]) -> tuple:
    system_message = None
    other_messages = []
    cleaned_messages = clean_history_messages(messages)
    for i, message in enumerate(cleaned_messages):
        if message['role'] == 'system':
            system_message = message['content']
        else:
            msg = {
                'role': message['role'],
                'content': message["content"],
            }

            other_messages.append(msg)
    return system_message, other_messages


class AnthropicChatAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.settings = AnthropicSettings()
        self.message_converter = AnthropicMessageConverter()
        self.response_converter = AnthropicResponseConverter()

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        system, other_messages = prepare_messages(messages)
        anthropic_tools = [tool.to_anthropic_tool() for tool in tools] if tools else None

        # Prepare the base kwargs
        request_kwargs = {
            "model": self.model,
            "system": system if system else [],
            "messages": other_messages,
            "temperature": self.settings.temperature if settings is None else settings.temperature,
            "top_p": self.settings.top_p if settings is None else settings.top_p,
            "top_k": self.settings.top_k if settings is None else settings.top_k,
            "max_tokens": self.settings.max_tokens if settings is None else settings.max_tokens,
            "stop_sequences": self.settings.stop_sequences if settings is None else settings.stop_sequences,
            "tools": anthropic_tools if anthropic_tools else [],
        }

        # Add extra_body if present
        if (settings is None and self.settings.extra_body) or (settings and settings.extra_body):
            extra_body = self.settings.extra_body if settings is None else settings.extra_body
            request_kwargs["extra_body"] = extra_body

        # Add extra request kwargs if present
        if (settings is None and self.settings.request_kwargs) or (settings and settings.request_kwargs):
            extra_kwargs = self.settings.request_kwargs if settings is None else settings.request_kwargs
            request_kwargs.update(extra_kwargs)

        response = self.client.messages.create(**request_kwargs)

        return self.response_converter.from_provider_response(response)

    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatAPIResponse, None, None]:
        system, other_messages = prepare_messages(messages)
        anthropic_tools = [tool.to_anthropic_tool() for tool in tools] if tools else None

        # Prepare the base kwargs
        request_kwargs = {
            "model": self.model,
            "system": system if system else [],
            "messages": other_messages,
            "stream": True,
            "temperature": self.settings.temperature if settings is None else settings.temperature,
            "top_p": self.settings.top_p if settings is None else settings.top_p,
            "max_tokens": self.settings.max_tokens if settings is None else settings.max_tokens,
            "tools": anthropic_tools if anthropic_tools else [],
        }

        # Add extra_body if present
        if (settings is None and self.settings.extra_body) or (settings and settings.extra_body):
            extra_body = self.settings.extra_body if settings is None else settings.extra_body
            request_kwargs.update(extra_body)

        # Add extra request kwargs if present
        if (settings is None and self.settings.request_kwargs) or (settings and settings.request_kwargs):
            extra_kwargs = self.settings.request_kwargs if settings is None else settings.request_kwargs
            request_kwargs.update(extra_kwargs)

        stream = self.client.messages.create(**request_kwargs)
        yield from self.response_converter.yield_from_provider(stream)


    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        return self.message_converter.to_provider_format(messages)


    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings
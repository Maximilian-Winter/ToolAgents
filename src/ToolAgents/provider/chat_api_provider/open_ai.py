import datetime
import json
import uuid
from copy import copy
from typing import List, Dict, Optional, Any, Generator

from openai import OpenAI

from ToolAgents import FunctionTool
from ToolAgents.messages.message_converter.open_ai_message_converter import OpenAIMessageConverter, \
    OpenAIResponseConverter
from ToolAgents.provider.llm_provider import SamplingSettings
from ToolAgents.provider.llm_provider import ChatAPIProvider, StreamingChatAPIResponse
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent, ToolCallContent, ToolCallResultContent, \
    BinaryContent, BinaryStorageType
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages


class OpenAISettings(SamplingSettings):
    def __init__(self):
        self.temperature = 0.4
        self.top_p = 1
        self.max_tokens = 4096
        self.response_format = None
        self.request_kwargs = None
        self.extra_body = None
        self.tool_choice = "auto"
        self.debug_mode = False

    def save_to_file(self, settings_file: str):
        with open(settings_file, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

    def load_from_file(self, settings_file: str):
        with open(settings_file, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)

    def as_dict(self):
        return copy(self.__dict__)

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
        elif sampler_name == "top_p":
            self.top_p = 1.0

        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_p = 1.0

    def set_response_format(self, response_format: dict[str, Any]):
        self.response_format = response_format

    def set_extra_body(self, extra_body: dict[str, Any]):
        self.extra_body = extra_body

    def set_extra_request_kwargs(self, **kwargs):
        self.request_kwargs = kwargs

    def set_tool_choice(self, tool_choice: str):
        self.tool_choice = tool_choice

class OpenAIChatAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str , base_url: str = "https://api.openai.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.settings = OpenAISettings()
        self.message_converter = OpenAIMessageConverter()
        self.response_converter = OpenAIResponseConverter()

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        # Prepare base request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": clean_history_messages(messages),
            "max_tokens": self.settings.max_tokens if settings is None else settings.max_tokens,
            "temperature": self.settings.temperature if settings is None else settings.temperature,
            "top_p": self.settings.top_p if settings is None else settings.top_p,
        }

        # Add tools if present
        if openai_tools:
            request_kwargs.update({
                "tools": openai_tools,
                "tool_choice": "auto"
            })

        # Add response format if present
        if (settings is None and self.settings.response_format) or (settings and settings.response_format):
            response_format = self.settings.response_format if settings is None else settings.response_format
            request_kwargs["response_format"] = response_format

        # Add extra_body if present
        if (settings is None and self.settings.extra_body) or (settings and settings.extra_body):
            extra_body = self.settings.extra_body if settings is None else settings.extra_body
            request_kwargs["extra_body"] = extra_body

        # Add extra request kwargs if present
        if (settings is None and self.settings.request_kwargs) or (settings and settings.request_kwargs):
            extra_kwargs = self.settings.request_kwargs if settings is None else settings.request_kwargs
            request_kwargs.update(extra_kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        return self.response_converter.from_provider_response(response)


    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatAPIResponse, None, None]:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        # Prepare base request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": clean_history_messages(messages),
            "max_tokens": self.settings.max_tokens if settings is None else settings.max_tokens,
            "stream": True,
            "temperature": self.settings.temperature if settings is None else settings.temperature,
            "top_p": self.settings.top_p if settings is None else settings.top_p,
        }

        # Add tools if present
        if openai_tools:
            request_kwargs.update({
                "tools": openai_tools,
                "tool_choice": self.settings.tool_choice if settings is None else settings.tool_choice,
            })

        # Add response format if present
        if (settings is None and self.settings.response_format) or (settings and settings.response_format):
            response_format = self.settings.response_format if settings is None else settings.response_format
            request_kwargs["response_format"] = response_format

        # Add extra_body if present
        if (settings is None and self.settings.extra_body) or (settings and settings.extra_body):
            extra_body = self.settings.extra_body if settings is None else settings.extra_body
            request_kwargs["extra_body"] = extra_body

        # Add extra request kwargs if present
        if (settings is None and self.settings.request_kwargs) or (settings and settings.request_kwargs):
            extra_kwargs = self.settings.request_kwargs if settings is None else settings.request_kwargs
            request_kwargs.update(extra_kwargs)

        stream = self.client.chat.completions.create(**request_kwargs)
        yield from self.response_converter.yield_from_provider(stream)

    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        return self.message_converter.to_provider_format(messages)

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings
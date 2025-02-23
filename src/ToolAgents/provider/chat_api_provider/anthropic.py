import copy
import json
from typing import List, Dict, Optional, Any, Generator

from anthropic import Anthropic

from ToolAgents import FunctionTool
from ToolAgents.messages.message_converter.anthropic_message_converter import AnthropicMessageConverter, \
    AnthropicResponseConverter
from ToolAgents.provider.llm_provider import ChatAPIProvider, ProviderSettings, StreamingChatMessage
from ToolAgents.messages.chat_message import ChatMessage

class AnthropicSettings(ProviderSettings):

    def __init__(self):
        super().__init__({"type": "auto"})
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 0

    def neutralize_sampler(self, sampler_name: str):
        if sampler_name == "temperature":
            self.temperature = 0.0
        elif sampler_name == "top_k":
            self.top_k = 0
        elif sampler_name == "top_p":
            self.top_p = 1.0

        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 0.0
        self.top_k = 0
        self.top_p = 1.0

def prepare_messages(messages: List[Dict[str, str]]) -> tuple:
    system_message = None
    other_messages = []
    cleaned_messages = messages
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

    def get_response(self, messages: List[ChatMessage], settings: AnthropicSettings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = False
        response = self.client.messages.create(**request_kwargs)

        return self.response_converter.from_provider_response(response)

    def get_streaming_response(self, messages: List[ChatMessage], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatMessage, None, None]:
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = True
        response = self.client.messages.create(**request_kwargs)
        yield from self.response_converter.yield_from_provider(response)

    def get_default_settings(self):
        return copy.copy(self.settings)

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return 'anthropic'

    def _prepare_request(self, messages: List[ChatMessage], settings: AnthropicSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings

        system, other_messages = prepare_messages(self.message_converter.to_provider_format(messages))
        anthropic_tools = [tool.to_anthropic_tool() for tool in tools] if tools else None

        request_kwargs = settings.to_dict()
        request_kwargs["model"] = self.model
        request_kwargs['system'] = system
        request_kwargs['messages'] = other_messages
        if anthropic_tools and len(anthropic_tools) > 0:
            request_kwargs['tools'] = anthropic_tools
        else:
            request_kwargs.pop('tool_choice')
        return request_kwargs
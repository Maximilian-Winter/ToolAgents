import copy
from typing import List, Dict, Optional, Any, Generator, AsyncGenerator

from anthropic import Anthropic, AsyncAnthropic

from ToolAgents import FunctionTool
from ToolAgents.provider.message_converter.anthropic_message_converter import AnthropicMessageConverter, \
    AnthropicResponseConverter
from ToolAgents.provider.llm_provider import ChatAPIProvider, ProviderSettings, StreamingChatMessage, SamplerSetting, \
    AsyncChatAPIProvider
from ToolAgents.messages.chat_message import ChatMessage



class AnthropicChatAPI(ChatAPIProvider):

    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.settings = ProviderSettings({"type": "auto"}, [SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                                                            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0),
                                                            SamplerSetting.create_sampler_setting("top_k", 0, 0)])
        self.message_converter = AnthropicMessageConverter()
        self.response_converter = AnthropicResponseConverter()

    def get_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = False
        response = self.client.messages.create(**request_kwargs)

        return self.response_converter.from_provider_response(response)

    def get_streaming_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
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

    def _prepare_request(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings
        request_kwargs = self.message_converter.prepare_request(self.model, messages, settings, tools)
        return request_kwargs

class AsyncAnthropicChatAPI(AsyncChatAPIProvider):

    def __init__(self, api_key: str, model: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.settings = ProviderSettings({"type": "auto"}, [SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                                                            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0),
                                                            SamplerSetting.create_sampler_setting("top_k", 0, 0)])
        self.message_converter = AnthropicMessageConverter()
        self.response_converter = AnthropicResponseConverter()

    async def get_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = False
        response = await self.client.messages.create(**request_kwargs)

        return self.response_converter.from_provider_response(response)

    async def get_streaming_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
                               tools: Optional[List[FunctionTool]] = None) -> AsyncGenerator[
        StreamingChatMessage, None]:
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = True
        response = self.client.messages.create(**request_kwargs)
        return self.response_converter.async_yield_from_provider(response)

    def get_default_settings(self):
        return copy.copy(self.settings)

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return 'anthropic'

    def _prepare_request(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings
        request_kwargs = self.message_converter.prepare_request(self.model, messages, settings, tools)
        return request_kwargs
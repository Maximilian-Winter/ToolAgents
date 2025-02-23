import copy
from typing import List, Dict, Optional, Any, Generator, AsyncGenerator

from mistralai import Mistral

from ToolAgents import FunctionTool
from ToolAgents.provider.message_converter.mistral_message_converter import MistralMessageConverter, \
    MistralResponseConverter
from ToolAgents.provider.llm_provider import ProviderSettings, SamplerSetting, AsyncChatAPIProvider
from ToolAgents.provider.llm_provider import ChatAPIProvider, StreamingChatMessage
from ToolAgents.messages.chat_message import ChatMessage


class MistralChatAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.settings = ProviderSettings("auto", [SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                                                            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0)])
        self.message_converter = MistralMessageConverter()
        self.response_converter = MistralResponseConverter()

    def get_response(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)

        response = self.client.chat.complete(**request_kwargs)
        return self.response_converter.from_provider_response(response)

    def get_streaming_response(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatMessage, None, None]:
        request_kwargs = self._prepare_request(messages, settings, tools)

        stream = self.client.chat.stream(**request_kwargs)

        yield from self.response_converter.yield_from_provider(stream)

    def get_default_settings(self):
        return copy.copy(self.settings)

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return 'mistral'

    def _prepare_request(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings

        request_kwargs = self.message_converter.prepare_request(self.model, messages, settings, tools)
        return request_kwargs

class AsyncMistralChatAPI(AsyncChatAPIProvider):
    def __init__(self, api_key: str, model: str):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.settings = ProviderSettings("auto", [SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                                                            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0)])
        self.message_converter = MistralMessageConverter()
        self.response_converter = MistralResponseConverter()

    async def get_response(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)

        response = await self.client.chat.complete_async(**request_kwargs)
        return self.response_converter.from_provider_response(response)

    async def get_streaming_response(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                               tools: Optional[List[FunctionTool]] = None) -> AsyncGenerator[
        StreamingChatMessage, None]:
        request_kwargs = self._prepare_request(messages, settings, tools)

        stream = self.client.chat.stream_async(**request_kwargs)

        return self.response_converter.async_yield_from_provider(stream)

    def get_default_settings(self):
        return copy.copy(self.settings)

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return 'mistral'

    def _prepare_request(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings

        request_kwargs = self.message_converter.prepare_request(self.model, messages, settings, tools)
        return request_kwargs
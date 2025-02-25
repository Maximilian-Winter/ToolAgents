from copy import copy
from typing import List, Dict, Optional, Any, Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI

from ToolAgents import FunctionTool
from ToolAgents.provider.message_converter.message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.provider.message_converter.open_ai_message_converter import OpenAIMessageConverter, \
    OpenAIResponseConverter
from ToolAgents.provider.llm_provider import ProviderSettings, SamplerSetting, AsyncChatAPIProvider
from ToolAgents.provider.llm_provider import ChatAPIProvider, StreamingChatMessage
from ToolAgents.messages.chat_message import ChatMessage


class OpenAIChatAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str , base_url: str = "https://api.openai.com/v1", provider_identifier:str = "openai", message_converter: BaseMessageConverter = OpenAIMessageConverter(), response_converter: BaseResponseConverter = OpenAIResponseConverter(), debug_mode: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.settings = ProviderSettings("auto", [SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                                                            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0)])
        self.message_converter = message_converter
        self.response_converter = response_converter
        self.debug_mode = debug_mode
        self.provider_identifier = provider_identifier

    def get_response(self, messages: List[ChatMessage], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)

        request_kwargs["stream"] = False

        response = self.client.chat.completions.create(**request_kwargs)
        return self.response_converter.from_provider_response(response)

    def get_streaming_response(self, messages: List[ChatMessage], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatMessage, None, None]:
        request_kwargs = self._prepare_request(messages, settings, tools)

        request_kwargs["stream"] = True

        stream = self.client.chat.completions.create(**request_kwargs)

        yield from self.response_converter.yield_from_provider(stream)

    def get_default_settings(self):
        return copy(self.settings)

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return self.provider_identifier

    def _prepare_request(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings

        request_kwargs = self.message_converter.prepare_request(self.model, messages, settings, tools)
        return request_kwargs

class AsyncOpenAIChatAPI(AsyncChatAPIProvider):
    def __init__(self, api_key: str, model: str , base_url: str = "https://api.openai.com/v1", provider_identifier:str = "openai", message_converter: BaseMessageConverter = OpenAIMessageConverter(), response_converter: BaseResponseConverter = OpenAIResponseConverter(), debug_mode: bool = False):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.settings = ProviderSettings("auto", [SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                                                            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0)])
        self.message_converter = message_converter
        self.response_converter = response_converter
        self.debug_mode = debug_mode
        self.provider_identifier = provider_identifier

    async def get_response(self, messages: List[ChatMessage], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        request_kwargs = self._prepare_request(messages, settings, tools)

        request_kwargs["stream"] = False

        response = await self.client.chat.completions.create(**request_kwargs)
        return self.response_converter.from_provider_response(response)

    async def get_streaming_response(self, messages: List[ChatMessage], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> AsyncGenerator[
        StreamingChatMessage, None]:
        request_kwargs = self._prepare_request(messages, settings, tools)

        request_kwargs["stream"] = True

        stream = self.client.chat.completions.create(**request_kwargs)
        return self.response_converter.async_yield_from_provider(stream)

    def get_default_settings(self):
        return copy(self.settings)

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return self.provider_identifier

    def _prepare_request(self, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        if settings is None:
            settings = self.settings

        request_kwargs = self.message_converter.prepare_request(self.model, messages, settings, tools)
        return request_kwargs
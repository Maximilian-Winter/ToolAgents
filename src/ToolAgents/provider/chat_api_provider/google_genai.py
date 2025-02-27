# src/ToolAgents/provider/chat_api_provider/google_genai.py
import copy
from typing import List, Dict, Optional, Any, Generator, AsyncGenerator
import uuid
import datetime
import json

import google.generativeai as genai
from google.generativeai.types import (
    AsyncGenerateContentResponse,
    GenerateContentResponse,
)

from ToolAgents import FunctionTool
from ToolAgents.messages import ChatMessageRole
from ToolAgents.provider.message_converter.google_genai_message_converter import (
    GoogleGenAIMessageConverter,
    GoogleGenAIResponseConverter,
)
from ToolAgents.provider.llm_provider import (
    ChatAPIProvider,
    ProviderSettings,
    StreamingChatMessage,
    SamplerSetting,
    AsyncChatAPIProvider,
)
from ToolAgents.messages.chat_message import ChatMessage


class GoogleGenAIChatAPI(ChatAPIProvider):
    """
    Google GenAI Chat API Provider for ToolAgents framework
    """

    def __init__(self, api_key: str, model: str):
        # Initialize the Google GenAI client
        genai.configure(api_key=api_key)
        self.model = model

        # Configure default settings
        self.settings = ProviderSettings(
            {"type": "auto"},
            [
                SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0),
                SamplerSetting.create_sampler_setting("top_k", 32, 0),
                SamplerSetting.create_sampler_setting("max_output_tokens", 2048, 2048),
            ],
        )
        self.message_converter = GoogleGenAIMessageConverter()
        self.response_converter = GoogleGenAIResponseConverter()

    def get_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> ChatMessage:
        """Get a response from the Google GenAI model"""
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = False
        model = genai.GenerativeModel(
            self.model, system_instruction=request_kwargs.pop("system_instruction")
        )
        response = model.generate_content(**request_kwargs)
        return self.response_converter.from_provider_response(response)

    def get_streaming_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Generator[StreamingChatMessage, None, None]:
        """Get a streaming response from the Google GenAI model"""

        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = True
        model = genai.GenerativeModel(
            self.model, system_instruction=request_kwargs.pop("system_instruction")
        )
        response = model.generate_content(**request_kwargs)
        yield from self.response_converter.yield_from_provider(response)

    def get_default_settings(self):
        """Return a copy of the default settings"""
        return copy.copy(self.settings)

    def set_default_settings(self, settings) -> None:
        """Set the default settings"""
        self.settings = settings

    def get_provider_identifier(self) -> str:
        """Return the provider identifier"""
        return "google_genai"

    def _prepare_request(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Dict[str, Any]:
        """Prepare the request for the Google GenAI API"""
        if settings is None:
            settings = self.settings

        request_kwargs = self.message_converter.prepare_request(
            self.model, messages, settings, tools
        )
        return request_kwargs


class AsyncGoogleGenAIChatAPI(AsyncChatAPIProvider):
    """
    Async Google GenAI Chat API Provider for ToolAgents framework
    """

    def __init__(self, api_key: str, model: str):
        # Initialize the Google GenAI client
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

        # Configure default settings
        self.settings = ProviderSettings(
            {"type": "auto"},
            [
                SamplerSetting.create_sampler_setting("temperature", 1.0, 0.0),
                SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0),
                SamplerSetting.create_sampler_setting("top_k", 32, 0),
                SamplerSetting.create_sampler_setting("max_output_tokens", 2048, 2048),
            ],
        )
        self.message_converter = GoogleGenAIMessageConverter()
        self.response_converter = GoogleGenAIResponseConverter()

    async def get_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> ChatMessage:
        """Get an async response from the Google GenAI model"""
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = False

        response = await self.model.generate_content_async(**request_kwargs)
        return self.response_converter.from_provider_response(response)

    async def get_streaming_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> AsyncGenerator[StreamingChatMessage, None]:
        """Get an async streaming response from the Google GenAI model"""
        request_kwargs = self._prepare_request(messages, settings, tools)
        request_kwargs["stream"] = True

        response = await self.model.generate_content_async(**request_kwargs)
        async_generator = self.response_converter.async_yield_from_provider(response)

        async for chunk in async_generator:
            yield chunk

    def get_default_settings(self):
        """Return a copy of the default settings"""
        return copy.copy(self.settings)

    def set_default_settings(self, settings) -> None:
        """Set the default settings"""
        self.settings = settings

    def get_provider_identifier(self) -> str:
        """Return the provider identifier"""
        return "google_genai"

    def _prepare_request(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Dict[str, Any]:
        """Prepare the request for the Google GenAI API"""
        if settings is None:
            settings = self.settings

        request_kwargs = self.message_converter.prepare_request(
            self.model, messages, settings, tools
        )
        return request_kwargs

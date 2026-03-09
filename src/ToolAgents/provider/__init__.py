import copy

from ToolAgents.provider.chat_api_provider import (
    AnthropicChatAPI,
    GroqChatAPI,
    MistralChatAPI,
    OpenAIChatAPI,
)
from ToolAgents.provider.llm_provider import (
    ChatAPIProvider,
    ProviderSettings,
    StreamingChatMessage,
    create_anthropic_settings,
    create_openai_settings,
    create_standard_settings,
)
from ToolAgents.provider.completion_provider.completion_provider import CompletionProvider
from ToolAgents.provider.completion_provider.default_implementations import LlamaCppServer


class OpenAISettings(ProviderSettings):
    def __init__(self):
        base_settings = create_openai_settings()
        super().__init__([copy.deepcopy(setting) for setting in base_settings._settings.values()])


class AnthropicSettings(ProviderSettings):
    def __init__(self):
        base_settings = create_anthropic_settings()
        super().__init__([copy.deepcopy(setting) for setting in base_settings._settings.values()])


class GroqSettings(ProviderSettings):
    def __init__(self):
        base_settings = create_standard_settings()
        super().__init__([copy.deepcopy(setting) for setting in base_settings._settings.values()])


class MistralSettings(ProviderSettings):
    def __init__(self):
        base_settings = create_standard_settings()
        super().__init__([copy.deepcopy(setting) for setting in base_settings._settings.values()])


__all__ = [
    'AnthropicChatAPI',
    'AnthropicSettings',
    'ChatAPIProvider',
    'CompletionProvider',
    'GroqChatAPI',
    'GroqSettings',
    'LlamaCppServer',
    'MistralChatAPI',
    'MistralSettings',
    'OpenAIChatAPI',
    'OpenAISettings',
    'ProviderSettings',
    'StreamingChatMessage',
]

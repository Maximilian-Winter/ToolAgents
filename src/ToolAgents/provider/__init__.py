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

__all__ = [
    'AnthropicChatAPI',
    'ChatAPIProvider',
    'CompletionProvider',
    'GroqChatAPI',
    'LlamaCppServer',
    'MistralChatAPI',
    'OpenAIChatAPI',
    'ProviderSettings',
    'StreamingChatMessage',
    'create_anthropic_settings',
    'create_openai_settings',
    'create_standard_settings',
]


def __getattr__(name: str):
    if name == 'AnthropicChatAPI':
        from ToolAgents.provider.chat_api_provider.anthropic import AnthropicChatAPI

        return AnthropicChatAPI

    if name == 'GroqChatAPI':
        from ToolAgents.provider.chat_api_provider.groq import GroqChatAPI

        return GroqChatAPI

    if name == 'MistralChatAPI':
        from ToolAgents.provider.chat_api_provider.mistral import MistralChatAPI

        return MistralChatAPI

    if name == 'OpenAIChatAPI':
        from ToolAgents.provider.chat_api_provider.open_ai import OpenAIChatAPI

        return OpenAIChatAPI

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

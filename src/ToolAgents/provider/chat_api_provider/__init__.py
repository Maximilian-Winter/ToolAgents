__all__ = [
    "AnthropicChatAPI",
    "AsyncAnthropicChatAPI",
    "GroqChatAPI",
    "AsyncGroqChatAPI",
    "MistralChatAPI",
    "AsyncMistralChatAPI",
    "OpenAIChatAPI",
    "AsyncOpenAIChatAPI",
]


def __getattr__(name: str):
    if name in {"AnthropicChatAPI", "AsyncAnthropicChatAPI"}:
        from ToolAgents.provider.chat_api_provider.anthropic import (
            AnthropicChatAPI,
            AsyncAnthropicChatAPI,
        )

        return {
            "AnthropicChatAPI": AnthropicChatAPI,
            "AsyncAnthropicChatAPI": AsyncAnthropicChatAPI,
        }[name]

    if name in {"GroqChatAPI", "AsyncGroqChatAPI"}:
        from ToolAgents.provider.chat_api_provider.groq import (
            GroqChatAPI,
            AsyncGroqChatAPI,
        )

        return {
            "GroqChatAPI": GroqChatAPI,
            "AsyncGroqChatAPI": AsyncGroqChatAPI,
        }[name]

    if name in {"MistralChatAPI", "AsyncMistralChatAPI"}:
        from ToolAgents.provider.chat_api_provider.mistral import (
            MistralChatAPI,
            AsyncMistralChatAPI,
        )

        return {
            "MistralChatAPI": MistralChatAPI,
            "AsyncMistralChatAPI": AsyncMistralChatAPI,
        }[name]

    if name in {"OpenAIChatAPI", "AsyncOpenAIChatAPI"}:
        from ToolAgents.provider.chat_api_provider.open_ai import (
            OpenAIChatAPI,
            AsyncOpenAIChatAPI,
        )

        return {
            "OpenAIChatAPI": OpenAIChatAPI,
            "AsyncOpenAIChatAPI": AsyncOpenAIChatAPI,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


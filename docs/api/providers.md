---
title: Providers API
---

# Providers API

A provider wraps one LLM API. Agents talk to providers, so switching model or
vendor means swapping the provider, not rewriting the agent.

## Choosing an endpoint

Every provider takes the same three core arguments:

```python
Provider(api_key: str, model: str, base_url: str = None)
```

`base_url` points a provider at a different address for the same API shape — a
gateway, a proxy, or a self-hosted server. Omit it to use the vendor's own
endpoint. The value is retained as `api.base_url`, alongside `api.model`.

```python
# OpenAI's own endpoint
api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")

# Any OpenAI-compatible server: OpenRouter, vLLM, Ollama, llama-cpp-server
api = OpenAIChatAPI(
    api_key="your-api-key",
    model="qwen/qwen3.5-9b",
    base_url="https://openrouter.ai/api/v1",
)

# Anthropic behind an internal gateway
api = AnthropicChatAPI(
    api_key="your-api-key",
    model="claude-sonnet-4-20250514",
    base_url="https://anthropic-gateway.internal/v1",
)
```

`MistralChatAPI` forwards `base_url` to the Mistral SDK's `server_url`
argument, so the name is the same across every provider.

Async providers are **not** re-exported from `ToolAgents.provider`; import them
from their own modules:

```python
from ToolAgents.provider.chat_api_provider.open_ai import AsyncOpenAIChatAPI
from ToolAgents.provider.chat_api_provider.anthropic import AsyncAnthropicChatAPI
from ToolAgents.provider.chat_api_provider.groq import AsyncGroqChatAPI
from ToolAgents.provider.chat_api_provider.mistral import AsyncMistralChatAPI
```

## Settings

!!! warning "Assigning an undeclared setting name does nothing"

    Attribute assignment only routes to `set_value` for a name the provider
    already declares. `settings.max_tokens = 4096` on a provider without a
    `max_tokens` setting silently creates a dead attribute — no error, and
    nothing is sent. Providers declare different sets:

    | Provider | Declared settings |
    | --- | --- |
    | OpenAI | `temperature`, `top_p`, `tool_choice`, `extra_body`, `response_format` |
    | Anthropic | `temperature`, `top_p`, `top_k`, `max_tokens` |
    | Groq | `temperature`, `top_p` |
    | Mistral | `temperature`, `top_p` |

    Check `settings.setting_names()` first, or use
    `add_request_setting(name, value)` to add one deliberately.

`to_dict()` emits the **API request** shape (`{"PROVIDER_SETTINGS": ...,
"REQUEST_SETTINGS": ..., "METADATA": ...}`), not a round-trippable
configuration; there is no `from_dict`. To describe a provider in a
serializable form, see the pipelines
[`ProviderConfig`](pipelines.md#providerconfig).

::: ToolAgents.provider.llm_provider.SettingLevel

::: ToolAgents.provider.llm_provider.LLMSetting

::: ToolAgents.provider.llm_provider.ProviderSettings

## Provider interfaces

::: ToolAgents.provider.llm_provider.ChatAPIProvider

::: ToolAgents.provider.llm_provider.AsyncChatAPIProvider

## Chat API providers

::: ToolAgents.provider.chat_api_provider.open_ai.OpenAIChatAPI

::: ToolAgents.provider.chat_api_provider.open_ai.AsyncOpenAIChatAPI

::: ToolAgents.provider.chat_api_provider.anthropic.AnthropicChatAPI

::: ToolAgents.provider.chat_api_provider.anthropic.AsyncAnthropicChatAPI

::: ToolAgents.provider.chat_api_provider.groq.GroqChatAPI

::: ToolAgents.provider.chat_api_provider.groq.AsyncGroqChatAPI

::: ToolAgents.provider.chat_api_provider.mistral.MistralChatAPI

::: ToolAgents.provider.chat_api_provider.mistral.AsyncMistralChatAPI

## Streaming

::: ToolAgents.provider.llm_provider.StreamingChatMessage

## Completion providers

For traditional completion-based models rather than chat APIs.

::: ToolAgents.provider.completion_provider.completion_provider.CompletionProvider

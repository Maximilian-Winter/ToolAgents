---
title: Providers API
---

# Providers API

## ProviderSettings

```python
from ToolAgents.provider.llm_provider import ProviderSettings
```

`ProviderSettings` stores named settings and can separate them into provider-level, request-level, and metadata fields.

!!! warning "Assigning an unknown setting name does nothing"

    Attribute assignment only routes to `set_value` for a name the provider
    already declares. `settings.max_tokens = 4096` on a provider without a
    `max_tokens` setting silently creates a dead attribute instead of raising —
    and providers declare different sets (Mistral has only `temperature` and
    `top_p`). Check `setting_names()` first, or use `add_request_setting(name,
    value)` to add one deliberately.

Note that `to_dict()` emits the **API request** shape
(`{"PROVIDER_SETTINGS": ..., "REQUEST_SETTINGS": ..., "METADATA": ...}`), not a
round-trippable config; there is no `from_dict`.

Common methods:

- `add_setting(setting)`
- `add_request_setting(name, value)`
- `add_provider_setting(name, value)`
- `remove_setting(name)`
- `get_setting(name)`
- `setting_names()`
- `get_value(name)`
- `set_value(name, value)`
- `update(**kwargs)`
- `reset(name)`
- `reset_all()`
- `neutralize(name)`
- `neutralize_all()`
- `to_dict(include=None, exclude=None, include_neutral=True, param_mapping=None)`
- `copy()`

The top-level provider helpers exported from `ToolAgents.provider` are:

- `create_openai_settings()`
- `create_anthropic_settings()`
- `create_standard_settings()`

## ChatAPIProvider

```python
from ToolAgents.provider.llm_provider import ChatAPIProvider
```

Synchronous provider interface:

- `get_response(messages, settings=None, tools=None)`
- `get_streaming_response(messages, settings=None, tools=None)`
- `get_default_settings()`
- `set_default_settings(settings)`
- `get_provider_identifier()`

## AsyncChatAPIProvider

```python
from ToolAgents.provider.llm_provider import AsyncChatAPIProvider
```

Async provider interface:

- `get_response(messages, settings=None, tools=None)`
- `get_streaming_response(messages, settings=None, tools=None)`
- `get_default_settings()`
- `set_default_settings(settings)`
- `get_provider_identifier()`

## Built-in Chat API Providers

```python
from ToolAgents.provider import (
    AnthropicChatAPI,
    GroqChatAPI,
    MistralChatAPI,
    OpenAIChatAPI,
)
```

Every provider takes the same three core arguments:

```python
Provider(api_key: str, model: str, base_url: str = None)
```

`base_url` points the provider at a different address for the same API shape —
a gateway, a proxy, or a self-hosted server. Omit it to use the provider's own
endpoint. The value is retained as `api.base_url`.

```python
# OpenAI's own endpoint
api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")

# Any OpenAI-compatible server
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

settings = api.get_default_settings()
```

`OpenAIChatAPI` additionally accepts `provider_identifier`, `message_converter`,
`response_converter` and `debug_mode`. `MistralChatAPI` forwards `base_url` to
the Mistral SDK's `server_url` argument, so the name is the same across
providers.

Async counterparts take the same arguments, but are **not** re-exported from
`ToolAgents.provider`; import them from their own modules:

```python
from ToolAgents.provider.chat_api_provider.open_ai import AsyncOpenAIChatAPI
from ToolAgents.provider.chat_api_provider.anthropic import AsyncAnthropicChatAPI
from ToolAgents.provider.chat_api_provider.groq import AsyncGroqChatAPI
from ToolAgents.provider.chat_api_provider.mistral import AsyncMistralChatAPI

api = AsyncOpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
agent = AsyncChatToolAgent(chat_api=api)
```

Provider instances expose the endpoint they were built for:

```python
api.model      # "gpt-4o-mini"
api.base_url   # "https://api.openai.com/v1", or your override
```

Default settings differ per provider: OpenAI declares `temperature`, `top_p`,
`tool_choice`, `extra_body` and `response_format`; Anthropic declares
`temperature`, `top_p`, `top_k` and `max_tokens`; Groq and Mistral declare
`temperature` and `top_p`.

## Completion Providers

```python
from ToolAgents.provider import CompletionProvider, LlamaCppServer
```

`CompletionProvider` is the abstract completion interface. `LlamaCppServer` is the built-in exported implementation helper for llama.cpp server usage.

## StreamingChatMessage

```python
from ToolAgents.provider.llm_provider import StreamingChatMessage
```

Streaming providers yield `StreamingChatMessage` instances, which expose chunk-level state and the finished chat message when streaming completes.

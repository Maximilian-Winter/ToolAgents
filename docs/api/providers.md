---
title: Providers API
---

# Providers API

## ProviderSettings

```python
from ToolAgents.provider.llm_provider import ProviderSettings
```

`ProviderSettings` stores named settings and can separate them into provider-level, request-level, and metadata fields.

Common methods:

- `add_setting(setting)`
- `add_request_setting(name, value)`
- `add_provider_setting(name, value)`
- `remove_setting(name)`
- `get_setting(name)`
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

Example:

```python
api = OpenAIChatAPI(
    api_key="your-api-key",
    model="gpt-4o-mini",
    base_url=None,
)
settings = api.get_default_settings()
```

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

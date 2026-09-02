---
title: Providers
---

# Providers

Providers are the components in ToolAgents that handle communication with language model APIs. They abstract away the differences between various LLM APIs, providing a consistent interface for agents to work with.

## Provider Types

ToolAgents supports a variety of LLM providers:

### Chat API Providers

Chat API providers handle communication with chat-based LLM APIs:

1. **OpenAIChatAPI**: For OpenAI's GPT models and compatible APIs
2. **AnthropicChatAPI**: For Anthropic's Claude models
3. **MistralChatAPI**: For Mistral AI's models
4. **GroqChatAPI**: For Groq's models
Ollama, OpenRouter, vLLM, Together and similar services speak the OpenAI API,
so they are `OpenAIChatAPI` with a `base_url` rather than classes of their own.
Every provider accepts `base_url`, so an Anthropic-, Groq- or Mistral-shaped API
can likewise be reached at a custom address.

### Completion Providers

For providers that use a completion-style API rather than a chat API:

1. **CompletionProvider**: Base class for completion providers
2. **TransformersCompletionEndpoint**: For using Hugging Face models
3. **LlamaCppPythonEndpoint**: For using llama.cpp models

## Provider Usage

### OpenAI API

```python
from ToolAgents.provider import OpenAIChatAPI

# Create the provider
api = OpenAIChatAPI(
    api_key="your-api-key",
    model="gpt-4o-mini"  # Specify the model
)

# Optionally, specify a base URL for compatible APIs
api = OpenAIChatAPI(
    api_key="your-api-key",
    model="your-model",
    base_url="https://api.example.com/v1"  # For compatible APIs
)
```

### Anthropic API

```python
from ToolAgents.provider import AnthropicChatAPI

# Create the provider
api = AnthropicChatAPI(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022"
)

# Optionally, point it at a gateway or proxy speaking the Anthropic API
api = AnthropicChatAPI(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022",
    base_url="https://anthropic-gateway.internal/v1"
)
```

### Mistral API

```python
from ToolAgents.provider import MistralChatAPI

# Create the provider
api = MistralChatAPI(
    api_key="your-mistral-key",
    model="mistral-small-latest",
    # base_url is optional; it is forwarded to the SDK's server_url
)
```

### Groq API

```python
from ToolAgents.provider import GroqChatAPI

# Create the provider
api = GroqChatAPI(
    api_key="your-groq-key",
    model="llama-3.3-70b-versatile",
    # base_url is optional
)
```

### OpenAI-compatible endpoints

There is no separate class for OpenRouter, Ollama, vLLM, Together or the rest.
They speak the OpenAI API, so they are `OpenAIChatAPI` with a different
`base_url`.

```python
from ToolAgents.provider import OpenAIChatAPI

# OpenRouter
api = OpenAIChatAPI(
    api_key="your-openrouter-key",
    model="google/gemini-2.0-pro-exp-02-05:free",
    base_url="https://openrouter.ai/api/v1"
)

# A local Ollama server. Note the /v1 suffix: that is the
# OpenAI-compatible endpoint, not Ollama's native one. A local
# server needs no real key, but the SDK requires the argument.
api = OpenAIChatAPI(
    api_key="not-required",
    model="llama3",
    base_url="http://localhost:11434/v1"
)

# A local vLLM server
api = OpenAIChatAPI(
    api_key="not-required",
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1"
)
```

`provider_identifier` can be set to label which service a provider is really
talking to:

```python
api = OpenAIChatAPI(
    api_key="your-openrouter-key",
    model="google/gemini-2.0-pro-exp-02-05:free",
    base_url="https://openrouter.ai/api/v1",
    provider_identifier="openrouter",
)
```

## Provider Configuration

### API Settings

Each provider has its own settings class for configuring API requests:

```python
# Get default settings for a provider
settings = api.get_default_settings()

# See what this provider actually declares before assigning
print(settings.setting_names())

# Configure settings
settings.temperature = 0.7
settings.top_p = 1.0

# Use settings when making requests
response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)
```

!!! warning "Assigning an undeclared setting does nothing"

    Attribute assignment only takes effect for a name the provider already
    declares. `settings.max_tokens = 1000` on an `OpenAIChatAPI` creates a dead
    attribute and is never sent — silently, with no error. Providers declare
    different sets, so check `settings.setting_names()` first.

    To send a parameter a provider does not declare but the endpoint
    understands, add it deliberately:

    ```python
    settings.add_request_setting("seed", 42)
    ```

### What each provider declares

Only `temperature` and `top_p` are common to all four. The rest differ, which
is why assignment should be checked against `setting_names()`:

| Provider | Declared settings |
| --- | --- |
| OpenAI | `temperature`, `top_p`, `tool_choice`, `extra_body`, `response_format` |
| Anthropic | `temperature`, `top_p`, `top_k`, `max_tokens` |
| Groq | `temperature`, `top_p` |
| Mistral | `temperature`, `top_p` |

Anything else an endpoint accepts — `seed`, `frequency_penalty`,
`presence_penalty`, `min_p` — is sent with `add_request_setting(name, value)`,
or through `extra_body` on OpenAI-compatible endpoints.

## Provider-Specific Features

### OpenAI

```python
from ToolAgents.provider import OpenAIChatAPI

api = OpenAIChatAPI(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

settings = api.get_default_settings()
settings.temperature = 0.7
settings.top_p = 1.0
settings.frequency_penalty = 0.0
settings.presence_penalty = 0.0
settings.response_format = {"type": "json_object"}  # Force JSON response
```

### Anthropic

```python
from ToolAgents.provider import AnthropicChatAPI

api = AnthropicChatAPI(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022"
)

settings = api.get_default_settings()
settings.temperature = 0.7
settings.top_p = 0.9
settings.top_k = 40
settings.max_tokens = 1000
```

### Mistral

```python
from ToolAgents.provider import MistralChatAPI

api = MistralChatAPI(
    api_key="your-mistral-key",
    model="mistral-small-latest"
)

settings = api.get_default_settings()
settings.temperature = 0.7
settings.top_p = 1.0
settings.random_seed = 42  # Set a seed for reproducibility
```

## Message Converters

ToolAgents uses message converters to translate between its unified format and provider-specific formats:

```python
from ToolAgents.provider.message_converter import (
    OpenAIMessageConverter,
    AnthropicMessageConverter,
    MistralMessageConverter
)

# These are used internally by the providers
converter = OpenAIMessageConverter()
# Convert from ToolAgents format to provider format
provider_messages = converter.convert_to_provider_messages(toolagents_messages)
# Convert from provider format to ToolAgents format
toolagents_messages = converter.convert_from_provider_messages(provider_messages)
```

## Working with Multiple Providers

You can create multiple providers and switch between them:

```python
# Create providers
openai_api = OpenAIChatAPI(api_key="openai-key", model="gpt-4o-mini")
anthropic_api = AnthropicChatAPI(api_key="anthropic-key", model="claude-3-5-sonnet-20241022")

# Create agents with different providers
openai_agent = ChatToolAgent(chat_api=openai_api)
anthropic_agent = ChatToolAgent(chat_api=anthropic_api)

# Use the same messages and tools with different providers
response_openai = openai_agent.get_response(
    messages=messages,
    settings=openai_api.get_default_settings(),
    tool_registry=tool_registry
)

response_anthropic = anthropic_agent.get_response(
    messages=messages,
    settings=anthropic_api.get_default_settings(),
    tool_registry=tool_registry
)
```

## Completion Providers

For traditional completion-based models:

```python
from ToolAgents.provider.completion_provider import CompletionProvider
from ToolAgents.provider.completion_provider.implementations import (
    TransformersCompletionEndpoint,
    LlamaCppPythonEndpoint,
)

# Use Hugging Face models
hf_provider = TransformersCompletionEndpoint(
    model_name="mistralai/Mistral-7B-Instruct-v0.1"
)

# Use llama.cpp models
llama_provider = LlamaCppPythonEndpoint(
    model_path="/path/to/model.gguf"
)
```

## Best Practices

1. **API Key Management**: Store API keys securely using environment variables. A
   [pipeline](../guides/pipelines.md#declaring-agents-and-endpoints) declares the
   *name* of the variable holding a key, never the key itself, so a workflow file
   can be committed safely.
2. **Error Handling**: Implement retry logic for API failures
3. **Model Selection**: Choose appropriate models for your use case
4. **Rate Limiting**: Be aware of API rate limits and implement throttling
5. **Fallback Providers**: Implement fallback mechanisms between providers
6. **Settings Optimization**: Tune settings based on your specific needs
7. **Cost Management**: Monitor usage to control costs with commercial APIs

## Next Steps

- [Learn about different agent types](agents.md)
- [Explore tool options](tools.md)
- [Understand message handling](messages.md)
- [Declare providers in a pipeline file](../guides/pipelines.md#declaring-agents-and-endpoints)
- [See provider usage examples](../examples/basic-agents.md)

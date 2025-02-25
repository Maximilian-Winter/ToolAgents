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
5. **OllamaChatAPI**: For local Ollama models
6. **OpenRouterChatAPI**: For accessing multiple models through OpenRouter

### Completion Providers

For providers that use a completion-style API rather than a chat API:

1. **CompletionProvider**: Base class for completion providers
2. **HuggingFaceTransformersProvider**: For using Hugging Face models
3. **LlamaCppPythonProvider**: For using llama.cpp models

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
```

### Mistral API

```python
from ToolAgents.provider import MistralChatAPI

# Create the provider
api = MistralChatAPI(
    api_key="your-mistral-key",
    model="mistral-small-latest"
)
```

### Groq API

```python
from ToolAgents.provider import GroqChatAPI

# Create the provider
api = GroqChatAPI(
    api_key="your-groq-key",
    model="llama-3.3-70b-versatile"
)
```

### Ollama API

```python
from ToolAgents.provider import OllamaChatAPI

# Create the provider (local Ollama server)
api = OllamaChatAPI(
    base_url="http://localhost:11434",  # Default Ollama URL
    model="llama3"
)
```

### OpenRouter API

```python
from ToolAgents.provider import OpenRouterChatAPI

# Create the provider
api = OpenRouterChatAPI(
    api_key="your-openrouter-key",
    model="google/gemini-2.0-pro-exp-02-05:free",
    base_url="https://openrouter.ai/api/v1"
)
```

## Provider Configuration

### API Settings

Each provider has its own settings class for configuring API requests:

```python
# Get default settings for a provider
settings = api.get_default_settings()

# Configure settings
settings.temperature = 0.7
settings.max_tokens = 1000
settings.top_p = 1.0

# Use settings when making requests
response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)
```

### Common Settings

While each provider has its own settings, there are common parameters across most providers:

1. **temperature**: Controls randomness (0.0 to 1.0)
2. **max_tokens**: Maximum number of tokens to generate
3. **top_p**: Nucleus sampling parameter (0.0 to 1.0)
4. **top_k**: Limits token selection to top K options (when supported)
5. **frequency_penalty**: Reduces repetition of tokens (when supported)
6. **presence_penalty**: Reduces repetition of topics (when supported)

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
    HuggingFaceTransformersProvider,
    LlamaCppPythonProvider
)

# Use Hugging Face models
hf_provider = HuggingFaceTransformersProvider(
    model_name="mistralai/Mistral-7B-Instruct-v0.1"
)

# Use llama.cpp models
llama_provider = LlamaCppPythonProvider(
    model_path="/path/to/model.gguf"
)
```

## Best Practices

1. **API Key Management**: Store API keys securely using environment variables
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
- [See provider usage examples](../examples/basic-agents.md)
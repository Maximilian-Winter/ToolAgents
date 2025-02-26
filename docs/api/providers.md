---
title: Providers API
---

# Providers API

The Providers module handles communication with various language model APIs, providing a consistent interface regardless of which LLM service you're using.

## Provider Settings

### ProviderSettings

Base class for LLM API request settings.

```python
from ToolAgents.provider.llm_provider import ProviderSettings
```

#### Properties

- `max_tokens` (int): Maximum tokens to generate
- `stop_sequences` (List[str]): Sequences that stop generation
- `samplers` (dict): Configurable generation parameters (temperature, top_p, etc.)

#### Methods

##### `set(setting_key, setting_value)`

Sets a specific setting.

**Parameters:**
- `setting_key` (str): The setting name
- `setting_value`: The setting value

##### `set_tool_choice(tool_choice)`

Sets tool choice behavior.

**Parameters:**
- `tool_choice`: Tool choice configuration (auto, required, specific tool)

##### `to_dict(include=None, filter_out=None)`

Converts settings to a dictionary.

**Parameters:**
- `include` (List[str], optional): Keys to include
- `filter_out` (List[str], optional): Keys to exclude

**Returns:**
- `dict`: Dictionary of settings

## Chat API Providers

### ChatAPIProvider

Abstract interface for chat API providers.

```python
from ToolAgents.provider.llm_provider import ChatAPIProvider
```

#### Methods

##### `get_response(messages, settings=None, tools=None)`

Gets a complete response from the LLM.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `settings` (ProviderSettings, optional): Provider settings
- `tools` (List[FunctionTool], optional): Available tools

**Returns:**
- `ChatMessage`: Response message

##### `get_streaming_response(messages, settings=None, tools=None)`

Gets a streaming response from the LLM.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `settings` (ProviderSettings, optional): Provider settings
- `tools` (List[FunctionTool], optional): Available tools

**Returns:**
- Generator yielding `StreamingChatMessage` objects

##### `get_default_settings()`

Gets default settings for the provider.

**Returns:**
- `ProviderSettings`: Default settings

### OpenAIChatAPI

Implementation of ChatAPIProvider for OpenAI's API.

```python
from ToolAgents.provider import OpenAIChatAPI

api = OpenAIChatAPI(
    api_key="your-api-key",
    model="gpt-4o-mini",
    base_url=None  # Optional for compatible APIs
)
```

#### Constructor Parameters

- `api_key` (str): The API key
- `model` (str): Model identifier
- `base_url` (str, optional): API endpoint for compatible APIs

#### Methods

Implements all methods from `ChatAPIProvider`.

### AnthropicChatAPI

Implementation of ChatAPIProvider for Anthropic's API.

```python
from ToolAgents.provider import AnthropicChatAPI

api = AnthropicChatAPI(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022"
)
```

#### Constructor Parameters

- `api_key` (str): The API key
- `model` (str): Model identifier
- `base_url` (str, optional): API endpoint for compatible APIs

#### Methods

Implements all methods from `ChatAPIProvider`.

### MistralChatAPI

Implementation of ChatAPIProvider for Mistral's API.

```python
from ToolAgents.provider import MistralChatAPI

api = MistralChatAPI(
    api_key="your-mistral-key",
    model="mistral-small-latest"
)
```

#### Constructor Parameters

- `api_key` (str): The API key
- `model` (str): Model identifier
- `base_url` (str, optional): API endpoint for compatible APIs

#### Methods

Implements all methods from `ChatAPIProvider`.

### GroqChatAPI

Implementation of ChatAPIProvider for Groq's API.

```python
from ToolAgents.provider import GroqChatAPI

api = GroqChatAPI(
    api_key="your-groq-key",
    model="llama-3.3-70b-versatile"
)
```

#### Constructor Parameters

- `api_key` (str): The API key
- `model` (str): Model identifier
- `base_url` (str, optional): API endpoint for compatible APIs

#### Methods

Implements all methods from `ChatAPIProvider`.



## Async Chat API Providers

### AsyncChatAPIProvider

Asynchronous version of ChatAPIProvider.

```python
from ToolAgents.provider.llm_provider import AsyncChatAPIProvider
```

#### Methods

##### `get_response_async(messages, settings=None, tools=None)`

Asynchronous version of `get_response`.

**Returns:**
- `Awaitable[ChatMessage]`: Response message

##### `get_streaming_response_async(messages, settings=None, tools=None)`

Asynchronous version of `get_streaming_response`.

**Returns:**
- Async generator yielding `StreamingChatMessage` objects

## Completion Providers

### CompletionProvider

Interface for traditional completion-based models.

```python
from ToolAgents.provider.completion_provider import CompletionProvider
```

#### Methods

##### `get_completion(prompt, settings=None)`

Gets a completion from the model.

**Parameters:**
- `prompt` (str): The text prompt
- `settings` (dict, optional): Completion settings

**Returns:**
- `str`: Generated text

##### `get_streaming_completion(prompt, settings=None)`

Gets a streaming completion from the model.

**Parameters:**
- `prompt` (str): The text prompt
- `settings` (dict, optional): Completion settings

**Returns:**
- Generator yielding text chunks

### CompletionProviderImplementations

#### HuggingFaceTransformersProvider

Implementation for Hugging Face models.

```python
from ToolAgents.provider.completion_provider.implementations import HuggingFaceTransformersProvider

provider = HuggingFaceTransformersProvider(
    model_name="mistralai/Mistral-7B-Instruct-v0.1"
)
```

#### LlamaCppPythonProvider

Implementation for llama.cpp models.

```python
from ToolAgents.provider.completion_provider.implementations import LlamaCppPythonProvider

provider = LlamaCppPythonProvider(
    model_path="/path/to/model.gguf"
)
```

## Message Converters

Message converters translate between ToolAgents' unified message format and provider-specific formats.

### MessageConverter

Abstract base class for message converters.

```python
from ToolAgents.provider.message_converter import MessageConverter
```

#### Methods

##### `convert_to_provider_messages(messages)`

Converts from ToolAgents format to provider format.

**Parameters:**
- `messages` (List[ChatMessage]): ToolAgents messages

**Returns:**
- Provider-specific messages

##### `convert_from_provider_messages(provider_messages)`

Converts from provider format to ToolAgents format.

**Parameters:**
- `provider_messages`: Provider-specific messages

**Returns:**
- `List[ChatMessage]`: ToolAgents messages

### Specific Converters

- `OpenAIMessageConverter`
- `AnthropicMessageConverter`
- `MistralMessageConverter`
- `GroqMessageConverter`

```python
from ToolAgents.provider.message_converter import OpenAIMessageConverter

converter = OpenAIMessageConverter()
openai_messages = converter.convert_to_provider_messages(toolagents_messages)
```

## Response Classes

### StreamingChatMessage

Represents a chunk in a streaming response.

```python
class StreamingChatMessage:
    chunk: str                   # Text chunk
    is_tool_call: bool           # Whether this chunk contains a tool call
    tool_call: Optional[dict]    # Tool call details if present
    finished: bool               # Whether this is the final chunk
    finished_chat_message: Optional[ChatMessage]  # Complete message when finished
```
---
title: Agents API
---

# Agents API

The Agents module contains the core agent classes that orchestrate interactions between language models and tools.

## BaseToolAgent

`BaseToolAgent` is an abstract base class that defines the interface for tool-capable agents.

```python
from ToolAgents.agents.base_llm_agent import BaseToolAgent
```

### Methods

#### `get_default_settings()`

Returns default settings for the agent.

**Returns:**
- Settings object specific to the provider being used

#### `step(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`

Executes a single interaction step with the LLM.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `tool_registry` (ToolRegistry, optional): Registry of available tools
- `settings` (ProviderSettings, optional): Provider settings
- `reset_last_messages_buffer` (bool): Whether to reset the messages buffer

**Returns:**
- `ChatResponse`: Response object containing messages and text

#### `stream_step(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`

Streaming version of the step method.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `tool_registry` (ToolRegistry, optional): Registry of available tools
- `settings` (ProviderSettings, optional): Provider settings
- `reset_last_messages_buffer` (bool): Whether to reset the messages buffer

**Returns:**
- Generator yielding `ChatResponseChunk` objects

#### `get_response(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`

Get a complete response with automatic tool handling.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `tool_registry` (ToolRegistry, optional): Registry of available tools
- `settings` (ProviderSettings, optional): Provider settings
- `reset_last_messages_buffer` (bool): Whether to reset the messages buffer

**Returns:**
- `ChatResponse`: Complete response including tool call results

#### `get_streaming_response(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`

Get a streaming response with automatic tool handling.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `tool_registry` (ToolRegistry, optional): Registry of available tools
- `settings` (ProviderSettings, optional): Provider settings
- `reset_last_messages_buffer` (bool): Whether to reset the messages buffer

**Returns:**
- Generator yielding `ChatResponseChunk` objects

#### `get_last_response()`

Returns the last response from the agent.

**Returns:**
- `ChatResponse`: The last response object

## ChatToolAgent

`ChatToolAgent` implements `BaseToolAgent` using LLM chat APIs.

```python
from ToolAgents.agents import ChatToolAgent

agent = ChatToolAgent(chat_api=api_provider, debug_output=False)
```

### Constructor Parameters

- `chat_api` (ChatAPIProvider): The chat API provider instance
- `debug_output` (bool, optional): Whether to output debugging information. Defaults to False.

### Methods

In addition to implementing all methods from `BaseToolAgent`, `ChatToolAgent` provides:

#### `handle_function_calling_response(chat_message, current_messages)`

Handles the execution of function calls in responses.

**Parameters:**
- `chat_message` (ChatMessage): The message containing function calls
- `current_messages` (List[ChatMessage]): Current conversation messages

**Returns:**
- `List[ChatMessage]`: Updated messages including tool responses

## AsyncBaseToolAgent

Asynchronous version of `BaseToolAgent`.

```python
from ToolAgents.agents.base_llm_agent import AsyncBaseToolAgent
```

### Methods

#### `get_response_async(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`

Asynchronous version of `get_response`.

**Returns:**
- `Awaitable[ChatResponse]`: Complete response including tool call results

#### `get_streaming_response_async(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`

Asynchronous version of `get_streaming_response`.

**Returns:**
- Async generator yielding `ChatResponseChunk` objects

## AsyncChatToolAgent

Asynchronous version of `ChatToolAgent`.

```python
from ToolAgents.agents import ChatToolAgent

agent = ChatToolAgent(chat_api=async_api_provider, debug_output=False)
```

## AdvancedAgent

`AdvancedAgent` is an enhanced agent with stateful context and advanced management features.

```python
from ToolAgents.agents import AdvancedAgent

agent = AdvancedAgent(
    chat_api=api_provider,
    context_app_state=app_state
)
```

### Constructor Parameters

- `chat_api` (ChatAPIProvider): The chat API provider instance
- `context_app_state` (ContextAppState): State management instance
- `debug_output` (bool, optional): Whether to output debugging information

### Methods

In addition to methods from `ChatToolAgent`, `AdvancedAgent` provides:

#### `get_response(messages, settings=None, include_relevant_context=True, stream_context=False)`

Get a response with context management.

**Parameters:**
- `messages` (List[ChatMessage]): The conversation messages
- `settings` (ProviderSettings, optional): Provider settings
- `include_relevant_context` (bool): Whether to include relevant context from memory
- `stream_context` (bool): Whether to stream context augmentation 

**Returns:**
- `ChatResponse`: Complete response

## Response Classes

### ChatResponse

Represents a complete response from the agent.

```python
class ChatResponse:
    messages: List[ChatMessage]  # All messages including tool calls and responses
    response: str                # Final text response
```

### ChatResponseChunk

Represents a chunk of a streaming response.

```python
class ChatResponseChunk:
    chunk: str                   # Text chunk
    has_tool_call: bool          # Whether this chunk contains a tool call
    tool_call: Optional[dict]    # Tool call data if present
    finished: bool               # Whether this is the final chunk
    finished_response: Optional[ChatResponse]  # Complete response in final chunk
```
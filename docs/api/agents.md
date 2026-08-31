---
title: Agents API
---

# Agents API

## BaseToolAgent

```python
from ToolAgents.agents.base_llm_agent import BaseToolAgent
```

Main synchronous interface:

- `get_default_settings()`
- `step(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `stream_step(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `get_response(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `get_streaming_response(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `get_last_response()`

## AsyncBaseToolAgent

```python
from ToolAgents.agents.base_llm_agent import AsyncBaseToolAgent
```

Main async interface:

- `get_default_settings()`
- `step(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `stream_step(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `get_response(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `get_streaming_response(messages, tool_registry=None, settings=None, reset_last_messages_buffer=True)`
- `get_last_response()`

## ChatToolAgent

```python
from ToolAgents.agents import ChatToolAgent

agent = ChatToolAgent(chat_api=api_provider, log_output=False, log_to_file=False)
```

Constructor parameters:

- `chat_api`: `ChatAPIProvider`
- `log_output`: enable request/response logging
- `log_to_file`: write logs to a timestamped file when logging is enabled

Additional method:

- `handle_function_calling_response(chat_message, current_messages)`

## AsyncChatToolAgent

```python
from ToolAgents.agents import AsyncChatToolAgent

agent = AsyncChatToolAgent(chat_api=async_api_provider)
```

Current constructor parameter:

- `chat_api`: `AsyncChatAPIProvider`

## Response Types

### ChatResponse

```python
from ToolAgents.data_models.responses import ChatResponse
```

Fields:

- `messages`: full message list accumulated by the agent
- `response`: final assistant text

### ChatResponseChunk

```python
from ToolAgents.data_models.responses import ChatResponseChunk
```

Fields:

- `chunk`
- `has_tool_call`
- `tool_call`
- `has_tool_call_result`
- `tool_call_result`
- `finished`
- `finished_response`

Helper methods:

- `get_tool_name()`
- `get_tool_arguments()`
- `get_tool_results()`

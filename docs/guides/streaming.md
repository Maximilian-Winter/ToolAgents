---
title: Streaming Responses
---

# Streaming Responses

Use `get_streaming_response(...)` when you want incremental output instead of waiting for the full response.

## Basic Streaming

```python
from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
tool_registry = ToolRegistry()

messages = [
    ChatMessage.create_system_message("You are a helpful assistant."),
    ChatMessage.create_user_message("Tell me about quantum computing."),
]

for chunk in agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry,
):
    print(chunk.chunk, end="", flush=True)
```

## Chunk Fields

`ChatResponseChunk` exposes:

- `chunk`
- `has_tool_call`
- `tool_call`
- `has_tool_call_result`
- `tool_call_result`
- `finished`
- `finished_response`

Example:

```python
for chunk in agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry,
):
    print(chunk.chunk, end="", flush=True)

    if chunk.has_tool_call:
        print("\nTool call:", chunk.get_tool_name())

    if chunk.has_tool_call_result:
        print("\nTool result:", chunk.get_tool_results())

    if chunk.finished:
        final_response = chunk.finished_response
```

## Streaming with Chat History

```python
from ToolAgents.data_models.chat_history import ChatHistory

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant.")
chat_history.add_user_message("What is 42 * 8?")

final_response = None
for chunk in agent.get_streaming_response(
    messages=chat_history.get_messages(),
    settings=settings,
    tool_registry=tool_registry,
):
    print(chunk.chunk, end="", flush=True)
    if chunk.finished:
        final_response = chunk.finished_response

if final_response is not None:
    chat_history.add_messages(final_response.messages)
```

## Async Streaming

Use `AsyncChatToolAgent` for async applications:

```python
import asyncio

from ToolAgents.agents import AsyncChatToolAgent

async def main(agent, messages, settings, tool_registry):
    async for chunk in agent.get_streaming_response(
        messages=messages,
        settings=settings,
        tool_registry=tool_registry,
    ):
        print(chunk.chunk, end="", flush=True)

asyncio.run(main(agent, messages, settings, tool_registry))
```

## Best Practices

1. Read `chunk.chunk` for user-visible text.
2. Use `finished_response` as the canonical final result.
3. Append `finished_response.messages` back into chat history after the stream completes.
4. Watch `has_tool_call` and `has_tool_call_result` if your UI surfaces tool activity.

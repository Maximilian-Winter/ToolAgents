---
title: Agents
---

# Agents

Agents connect provider backends, messages, and tools.

## ChatToolAgent

`ChatToolAgent` is the main synchronous agent:

```python
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api, log_output=False)
```

Constructor parameters:

- `chat_api`: a `ChatAPIProvider` implementation
- `log_output`: enable request/response logging
- `log_to_file`: optionally write logs to a timestamped file

## AsyncChatToolAgent

Use `AsyncChatToolAgent` with an async provider:

```python
from ToolAgents.agents import AsyncChatToolAgent

agent = AsyncChatToolAgent(chat_api=async_api_provider)
```

## Common Usage Pattern

```python
from ToolAgents import ToolRegistry
from ToolAgents.data_models.messages import ChatMessage

settings = api.get_default_settings()
tool_registry = ToolRegistry()

messages = [
    ChatMessage.create_system_message("You are helpful."),
    ChatMessage.create_user_message("What is the capital of France?"),
]

response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry,
)

print(response.response)
```

## Streaming

```python
stream = agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry,
)

for chunk in stream:
    print(chunk.chunk, end="", flush=True)
```

## Async Streaming

```python
async for chunk in agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry,
):
    print(chunk.chunk, end="", flush=True)
```

## AgentHarness

For longer-running assistants, use `AgentHarness` instead of writing the turn
loop yourself. It wraps `ChatToolAgent` with prompt composition, message
lifecycle helpers, context trimming, token tracking, tools, and event hooks.

```python
from ToolAgents.agent_harness import create_harness

harness = create_harness(
    provider=api,
    system_prompt="You are helpful.",
    max_context_tokens=128000,
)

harness.prompt_composer.add_module(
    "runtime_context",
    position=10,
    content_fn=lambda: current_state_as_text(),
)

print(harness.chat("Hello"))
```

Use `ContextManager` directly only when you need a fully custom agent loop.

## Choosing an Agent

- Use `ChatToolAgent` for most applications.
- Use `AsyncChatToolAgent` when the surrounding app is async.
- Use `AgentHarness` or `AsyncAgentHarness` when you want a managed turn loop.

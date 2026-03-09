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

## AdvancedAgent

`AdvancedAgent` wraps a base tool agent and adds app-state and memory-oriented workflows.

```python
from ToolAgents import ToolRegistry
from ToolAgents.agents import AdvancedAgent, ChatToolAgent
from ToolAgents.agents.advanced_agent import AgentConfig

base_agent = ChatToolAgent(chat_api=api)
tool_registry = ToolRegistry()
agent_config = AgentConfig()

advanced_agent = AdvancedAgent(
    agent=base_agent,
    tool_registry=tool_registry,
    agent_config=agent_config,
)
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

## Choosing an Agent

- Use `ChatToolAgent` for most applications.
- Use `AsyncChatToolAgent` when the surrounding app is async.
- Use `AdvancedAgent` when you specifically need the higher-level state/memory workflow.

---
title: Agents API
---

# Agents API

An agent drives the conversation loop: it sends messages to a
[provider](providers.md), executes any tool calls the model returns, feeds the
results back, and repeats until the model answers without calling a tool.

```python
from ToolAgents.agents import ChatToolAgent, AsyncChatToolAgent
```

## ChatToolAgent

::: ToolAgents.agents.chat_tool_agent.ChatToolAgent

## AsyncChatToolAgent

::: ToolAgents.agents.chat_tool_agent.AsyncChatToolAgent

## Base classes

Implement these to write an agent of your own.

::: ToolAgents.agents.base_llm_agent.BaseToolAgent

::: ToolAgents.agents.base_llm_agent.AsyncBaseToolAgent

## Observability

::: ToolAgents.agents.base_llm_agent.AgentObservabilityHandler

## Responses

::: ToolAgents.data_models.responses.ChatResponse

::: ToolAgents.data_models.responses.ChatResponseChunk

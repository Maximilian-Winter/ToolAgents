---
title: Tools API
---

# Tools API

A tool is a Python callable the model can invoke. `FunctionTool` wraps a
function, a Pydantic model, or an OpenAI-style specification into one shape,
and `ToolRegistry` is the set of tools offered for a given request.

```python
from ToolAgents import FunctionTool, ToolRegistry
```

See the [custom tools guide](../guides/custom-tools.md) for how to write one,
and [tool adapters](../components/tool-adapters.md) for exposing tools over a
CLI or MCP.

## FunctionTool

::: ToolAgents.function_tool.FunctionTool

::: ToolAgents.function_tool.AsyncFunctionTool

## ToolRegistry

::: ToolAgents.function_tool.ToolRegistry

## Pre- and post-processors

Processors run either side of a tool call: use them to normalize arguments
before execution or to reshape results before they return to the model.

::: ToolAgents.function_tool.BaseProcessor

::: ToolAgents.function_tool.PreProcessor

::: ToolAgents.function_tool.PostProcessor

## Execution context and confirmation

::: ToolAgents.function_tool.ToolExecutionContext

::: ToolAgents.function_tool.ConfirmationRequest

::: ToolAgents.function_tool.ConfirmationState

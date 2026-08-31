---
title: Tool Adapters
---

# Tool Adapters

Tool adapters let you reuse `FunctionTool` and `ToolRegistry` collections outside
the normal agent loop. The same tools can be called from Python, exposed through a
CLI, loaded from an MCP server, or served as an MCP server.

## Execute Tools By Name

```python
from ToolAgents import FunctionTool
from ToolAgents.tool_adapters import execute_tool_by_name


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


tools = [FunctionTool(add)]
result = execute_tool_by_name(tools, "add", {"a": 2, "b": 3})
print(result)
```

## Use Tools From The CLI

Create a module that returns tools:

```python
# my_tools.py
from ToolAgents import FunctionTool


def greet(name: str) -> str:
    """Greet a person.

    Args:
        name: Person to greet.
    """
    return f"Hello {name}"


def create_tools():
    return [FunctionTool(greet)]
```

Then list, inspect, and call the tools:

```bash
toolagents-tools list --module my_tools:create_tools
toolagents-tools schema greet --module my_tools:create_tools
toolagents-tools call greet --module my_tools:create_tools --json "{\"name\":\"Ada\"}"
```

The CLI accepts any zero-argument factory that returns a list of `FunctionTool`
instances or a `ToolRegistry`.

## Load MCP Tools Into ToolAgents

Install MCP support first:

```bash
pip install ToolAgents[mcp]
```

Then load tools from an MCP server:

```python
import asyncio

from ToolAgents.tool_adapters import load_mcp_tools_from_http


tools = asyncio.run(
    load_mcp_tools_from_http({"url": "http://127.0.0.1:8042/mcp"})
)
harness.add_tools(tools)
```

## Serve ToolAgents Tools As MCP

Any ToolAgents tool collection can be exposed as an MCP server:

```python
from ToolAgents.tool_adapters import serve_tools_as_mcp

from my_tools import create_tools


serve_tools_as_mcp(
    create_tools(),
    name="my-toolagents-tools",
    transport="streamable-http",
    port=8042,
)
```

For `NavigableMemory`, keep the memory implementation independent from MCP and
adapt the generated tools:

```python
from ToolAgents import FunctionTool
from ToolAgents.agent_memory.navigable_memory import JSONBackend, NavigableMemory
from ToolAgents.tool_adapters import serve_tools_as_mcp


memory = NavigableMemory(JSONBackend("memory.json"))
tools = [FunctionTool(tool) for tool in memory.create_tools()]

serve_tools_as_mcp(tools, name="navigable-memory", port=8042)
```

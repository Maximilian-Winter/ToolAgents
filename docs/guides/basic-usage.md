---
title: Basic Usage
---

# Basic Usage

This guide covers the fundamental concepts and usage patterns of the ToolAgents framework.

## Core Components

ToolAgents consists of several key components:

1. **Agents**: Classes that orchestrate communication with LLMs and tool execution
2. **Tools**: Functions or classes that provide capabilities to agents
3. **Providers**: Interfaces to various LLM APIs (OpenAI, Anthropic, etc.)
4. **Messages**: Structures for chat history and communication
5. **ToolRegistry**: Container for registering and managing tools

## Basic Agent Usage

The most common agent type is the `ChatToolAgent`, which facilitates communication with chat-based LLM APIs and handles function/tool calling:

```python
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

# Initialize the API provider
api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")

# Create the agent
agent = ChatToolAgent(chat_api=api)

# Configure settings
settings = api.get_default_settings()
settings.temperature = 0.7
```

## Working with Tools

Tools are the capabilities you provide to your agent. ToolAgents supports several ways to define tools:

```python
from ToolAgents import FunctionTool, ToolRegistry

# Define a simple tool function
def hello_world(name: str) -> str:
    """
    Return a greeting message.
    
    Args:
        name: The name to greet
    
    Returns:
        A greeting message
    """
    return f"Hello, {name}!"

# Create a FunctionTool
hello_tool = FunctionTool(hello_world)

# Register the tool
tool_registry = ToolRegistry()
tool_registry.add_tool(hello_tool)

# Or register multiple tools at once
tool_registry.add_tools([hello_tool, another_tool, third_tool])
```

## Sending Messages

To interact with the agent, you need to construct messages:

```python
from ToolAgents.messages.chat_message import ChatMessage

# Create messages
system_message = ChatMessage.create_system_message(
    "You are a helpful assistant that can greet users."
)
user_message = ChatMessage.create_user_message("Can you greet me?")

# Combine into a list
messages = [system_message, user_message]
```

## Getting Responses

Now you can get responses from the agent:

```python
# Get a standard response
response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

print(response.response)

# Access the messages that were generated
for message in response.messages:
    print(f"{message.role}: {message.content}")
```

## Handling Tool Calls

When the agent calls a tool, ToolAgents automatically handles the execution and returns the result:

```python
user_message = ChatMessage.create_user_message("Please greet John")
messages = [system_message, user_message]

response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

# The agent will call the hello_world tool with name="John"
print(response.response)  # Will include "Hello, John!"
```

## Working with Different Providers

ToolAgents makes it easy to switch between different LLM providers:

```python
from ToolAgents.provider import AnthropicChatAPI, MistralChatAPI, GroqChatAPI

# Using Anthropic
anthropic_api = AnthropicChatAPI(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022"
)
anthropic_agent = ChatToolAgent(chat_api=anthropic_api)

# Using Mistral
mistral_api = MistralChatAPI(
    api_key="your-mistral-key",
    model="mistral-small-latest"
)
mistral_agent = ChatToolAgent(chat_api=mistral_api)

# Using Groq
groq_api = GroqChatAPI(
    api_key="your-groq-key",
    model="llama-3.3-70b-versatile"
)
groq_agent = ChatToolAgent(chat_api=groq_api)
```

## Error Handling

When working with agents and tools, it's important to handle potential errors:

```python
try:
    response = agent.get_response(
        messages=messages,
        settings=settings,
        tool_registry=tool_registry
    )
    print(response.response)
except Exception as e:
    print(f"Error: {e}")
```

## Advanced Configuration

You can configure various aspects of the agent's behavior:

```python
# Configure OpenAI API settings
settings = api.get_default_settings()
settings.temperature = 0.7
settings.top_p = 1.0
settings.max_tokens = 1000
settings.presence_penalty = 0.0
settings.frequency_penalty = 0.0

# Configure Anthropic API settings
anthropic_settings = anthropic_api.get_default_settings()
anthropic_settings.temperature = 0.7
anthropic_settings.max_tokens = 1000
```

## Next Steps

Now that you understand the basics of ToolAgents, you can move on to more advanced topics:

- [Creating custom tools](custom-tools.md)
- [Managing chat history](chat-history.md)
- [Working with streaming responses](streaming.md)
- [Exploring the components](../components/agents.md)
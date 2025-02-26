---
title: Agents
---

# Agents

Agents are the core components of the ToolAgents framework that orchestrate the interaction between language models and tools. They handle message processing, tool execution, and response generation.

## Agent Types

ToolAgents provides several agent types for different use cases:

### ChatToolAgent

The most commonly used agent type, `ChatToolAgent` facilitates interaction with chat-based LLM APIs and manages function/tool calling:

```python
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

# Initialize API
api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")

# Create the agent
agent = ChatToolAgent(chat_api=api)
```

### AdvancedAgent

`AdvancedAgent` provides enhanced capabilities for complex workflows, contextual awareness, and state management:

```python
from ToolAgents.agents import AdvancedAgent

# Create advanced agent with context
agent = AdvancedAgent(
    chat_api=api,
    context_app_state=your_app_state  # Context and application state
)
```

### BaseLLMAgent

The foundation for all agent types in ToolAgents, `BaseLLMAgent` provides common functionality and interfaces:

```python
from ToolAgents.agents import BaseLLMAgent

# This is typically extended by other agent classes
class CustomAgent(BaseLLMAgent):
    def __init__(self, chat_api, **kwargs):
        super().__init__(chat_api, **kwargs)
        # Custom initialization
    
    # Override methods for custom behavior
```

## Agent Usage Patterns

### Basic Response Generation

```python
from ToolAgents.messages.chat_message import ChatMessage

# Define messages
messages = [
    ChatMessage.create_system_message("You are a helpful assistant."),
    ChatMessage.create_user_message("What is the capital of France?")
]

# Get a response
response = agent.get_response(
    messages=messages,
    settings=settings  # API settings
)

print(response.response)
```

### Tool Integration

```python
from ToolAgents import ToolRegistry, FunctionTool

# Define tools
def get_weather(location):
    """Get weather for a location"""
    # Implementation...
    return f"The weather in {location} is sunny."

weather_tool = FunctionTool(get_weather)

# Create tool registry
tool_registry = ToolRegistry()
tool_registry.add_tool(weather_tool)

# Get response with tools
response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)
```

### Streaming Responses

```python
# Get a streaming response
stream = agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

# Process the stream
for chunk in stream:
    print(chunk.chunk, end='', flush=True)
    if chunk.finished:
        final_response = chunk.finished_response
```

### Async Responses

```python
import asyncio

async def get_agent_response():
    # Get an async response
    response = await agent.get_response_async(
        messages=messages,
        settings=settings,
        tool_registry=tool_registry
    )
    return response

# Or with streaming
async def get_streaming_response():
    async for chunk in agent.get_streaming_response_async(
        messages=messages,
        settings=settings,
        tool_registry=tool_registry
    ):
        print(chunk.chunk, end='', flush=True)

asyncio.run(get_agent_response())
```

## Advanced Agent Features

### Chat History Integration

```python
from ToolAgents.messages import ChatHistory

# Create chat history
chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant.")
chat_history.add_user_message("Hello!")

# Get response using chat history
response = agent.get_response(
    messages=chat_history.get_messages(),
    settings=settings,
    tool_registry=tool_registry
)

# Update chat history
chat_history.add_messages(response.messages)
```

### Multiple Providers

Agents can work with various LLM providers:

```python
from ToolAgents.provider import AnthropicChatAPI, MistralChatAPI

# Create agents with different providers
anthropic_api = AnthropicChatAPI(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022"
)
anthropic_agent = ChatToolAgent(chat_api=anthropic_api)

mistral_api = MistralChatAPI(
    api_key="your-mistral-key",
    model="mistral-small-latest"
)
mistral_agent = ChatToolAgent(chat_api=mistral_api)
```

### Memory Management

Advanced agents can incorporate semantic memory:

```python
from ToolAgents.agent_memory.semantic_memory import HierarchicalMemory

# Create memory
memory = HierarchicalMemory()
memory.add_memory("User is interested in quantum physics.")

# Retrieve relevant memories
relevant_memories = memory.search_memory(
    "What topics should I discuss?",
    top_k=3
)

# Use memories in system message
system_message = f"""
You are a helpful assistant.
Relevant context: {' '.join(relevant_memories)}
"""

# Create chat history with memory-enhanced system message
chat_history = ChatHistory()
chat_history.add_system_message(system_message)
```

## Best Practices

1. **Select the right agent type** for your use case:
   - `ChatToolAgent` for most applications
   - `AdvancedAgent` for complex workflows with state management

2. **Configure API settings** appropriately:
   - Lower temperature (0.0-0.5) for deterministic, factual responses
   - Higher temperature (0.6-1.0) for creative, varied responses

3. **Provide clear system messages** to guide agent behavior:
   - Define the agent's role and capabilities
   - Include any constraints or special instructions

4. **Structure tools effectively**:
   - Group related functionality into coherent tools
   - Provide detailed descriptions and type hints

5. **Handle errors gracefully**:
   - Implement error handling for API failures
   - Validate tool inputs and outputs

6. **Manage context effectively**:
   - Keep track of message token counts
   - Summarize or truncate long conversations

## Next Steps

- [Learn about ToolAgents tools](tools.md)
- [Explore provider options](providers.md)
- [Understand message handling](messages.md)
- [See more advanced examples](../examples/advanced-agents.md)
---
title: Streaming Responses
---

# Streaming Responses

Streaming responses provide a better user experience by showing the agent's response as it's being generated, rather than waiting for the complete response. This guide covers how to use streaming with ToolAgents.

## Basic Streaming

To get streaming responses from a ChatToolAgent, use the `get_streaming_response` method instead of `get_response`:

```python
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents import ToolRegistry

# Set up API and agent
api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()

# Set up tool registry
tool_registry = ToolRegistry()
tool_registry.add_tools([your_tool1, your_tool2])

# Create messages
messages = [
    ChatMessage.create_system_message("You are a helpful assistant."),
    ChatMessage.create_user_message("Tell me about quantum computing.")
]

# Get a streaming response
stream = agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

# Process the stream
for chunk in stream:
    print(chunk.chunk, end='', flush=True)
```

## Stream Chunk Properties

Each chunk in the stream has several properties:

```python
for chunk in stream:
    # The text content of this chunk
    print(chunk.chunk)
    
    # Whether this is the final chunk
    if chunk.finished:
        # The complete response object (only available in the final chunk)
        final_response = chunk.finished_response
        
        # All messages generated in the response
        for message in final_response.messages:
            print(f"{message.role}: {message.content}")
```

## Streaming with Tool Calls

When streaming responses that include tool calls, the process is handled automatically:

```python
for chunk in stream:
    # Regular content chunks
    print(chunk.chunk, end='', flush=True)
    
    # Check if this chunk indicates a tool call is about to happen
    if chunk.tool_call_start:
        print("\nStarting tool call...")
    
    # Check if this chunk has tool call results
    if chunk.tool_call_result:
        print(f"\nTool call result: {chunk.tool_call_result}")
```

## Complete Streaming Example

Here's a complete example of streaming with tool calls and chat history:

```python
import os
from dotenv import load_dotenv

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables
load_dotenv()

# Define a simple calculator tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression, {"__builtins__": {}}, {"sqrt": lambda x: x**0.5})

calculator_tool = FunctionTool(calculator)

# Set up API and agent
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.7

# Set up tool registry
tool_registry = ToolRegistry()
tool_registry.add_tool(calculator_tool)

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful assistant with tool calling capabilities."
)

# User interaction loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
        
    # Add user message to history
    chat_history.add_user_message(user_input)
    
    # Get streaming response
    stream = agent.get_streaming_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tool_registry
    )
    
    # Process the stream
    print("Assistant: ", end='')
    final_response = None
    
    for chunk in stream:
        print(chunk.chunk, end='', flush=True)
        if chunk.finished:
            final_response = chunk.finished_response
    
    print()  # New line after response
    
    # Add response messages to history
    if final_response:
        chat_history.add_messages(final_response.messages)
    else:
        print("Error: No final response received")
```

## Async Streaming

ToolAgents also supports async streaming for applications that use asynchronous programming:

```python
import asyncio
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

async def main():
    # Set up API and agent
    api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    
    # Set up messages and tools
    # ...
    
    # Get an async streaming response
    async for chunk in agent.get_streaming_response_async(
        messages=messages,
        settings=settings,
        tool_registry=tool_registry
    ):
        print(chunk.chunk, end='', flush=True)
        if chunk.finished:
            final_response = chunk.finished_response

# Run the async function
asyncio.run(main())
```

## Streaming in Web Applications

For web applications, you can use streaming to provide a responsive chat interface:

```python
from flask import Flask, Response, request, jsonify
import json

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # Set up agent, tools, and chat history
    # ...
    
    # Add user message to chat history
    chat_history.add_user_message(user_message)
    
    def generate():
        stream = agent.get_streaming_response(
            messages=chat_history.get_messages(),
            settings=settings,
            tool_registry=tool_registry
        )
        
        for chunk in stream:
            yield f"data: {json.dumps({'chunk': chunk.chunk})}\n\n"
            
            if chunk.finished:
                # Add response to chat history
                chat_history.add_messages(chunk.finished_response.messages)
                yield f"data: {json.dumps({'done': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

## Best Practices

When using streaming responses:

1. **Handle all chunk types**: Process text chunks, tool calls, and the final response
2. **Manage UI updates**: Update the UI smoothly as chunks arrive
3. **Use flush**: When printing to console, use `flush=True` to ensure immediate output
4. **Error handling**: Include error handling for interrupted streams
5. **Save chat history**: Only update chat history when you receive the final chunk

## Next Steps

Now that you know how to use streaming responses:

- [Learn about different agent types](../components/agents.md)
- [Explore more complex examples](../examples/advanced-agents.md)
- [Build web interfaces](../examples/web-research.md)
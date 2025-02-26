---
title: Quick Start
---

# Quick Start Guide

This guide will help you get up and running with ToolAgents quickly. We'll create a simple agent that can perform calculations, check the current date and time, and get weather information.

## Installation

First, make sure you have ToolAgents installed:

```bash
pip install ToolAgents
```

## Setting Up Your Environment

Create a new Python file and start by importing the necessary modules and setting up your environment:

```python
import os
from dotenv import load_dotenv

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables from a .env file (optional)
load_dotenv()

# Set up the API with your API key
api = OpenAIChatAPI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
```

## Defining Tools

Let's define a few simple tools for our agent:

```python
from ToolAgents import FunctionTool
import datetime
import math

# A calculator tool using a simple function
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression to evaluate.
            Example: "2 + 2", "5 * 10", "sqrt(16)"
    
    Returns:
        The result of the evaluated expression as a float.
    """
    # For safety in a real app, you would want to validate and sandbox this
    return eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos})

# A date/time tool
def current_datetime(output_format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Get the current date and time in the specified format.
    
    Args:
        output_format: The format string for the date and time output.
            Default is '%Y-%m-%d %H:%M:%S'.
    
    Returns:
        The current date and time as a formatted string.
    """
    return datetime.datetime.now().strftime(output_format)

# A weather tool (mock implementation)
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country for the weather report.
            Example: "London, UK", "New York, USA"
        unit: The temperature unit, either "celsius" or "fahrenheit".
            Default is "celsius".
    
    Returns:
        A string with the weather information.
    """
    # In a real app, this would call a weather API
    weather_data = {
        "London, UK": {"temperature": 18, "condition": "Cloudy"},
        "New York, USA": {"temperature": 25, "condition": "Sunny"},
        "Tokyo, Japan": {"temperature": 22, "condition": "Rainy"},
    }
    
    # Default weather for locations not in our mock data
    default_weather = {"temperature": 20, "condition": "Clear"}
    
    weather = weather_data.get(location, default_weather)
    temp = weather["temperature"]
    
    if unit.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return f"The weather in {location} is {temp}°{'F' if unit.lower() == 'fahrenheit' else 'C'} and {weather['condition'].lower()}."

# Create FunctionTool instances
calculator_tool = FunctionTool(calculator)
datetime_tool = FunctionTool(current_datetime)
weather_tool = FunctionTool(get_weather)
```

## Creating the Agent

Now, let's set up our agent with the tools we've defined:

```python
# Create the agent with the OpenAI API
agent = ChatToolAgent(chat_api=api)

# Get default settings and adjust as needed
settings = api.get_default_settings()
settings.temperature = 0.45
settings.top_p = 1.0

# Create a tool registry and add our tools
tool_registry = ToolRegistry()
tool_registry.add_tools([calculator_tool, datetime_tool, weather_tool])
```

## Using the Agent

Let's use our agent to answer a question that requires the use of our tools:

```python
# Define messages for the agent
messages = [
    ChatMessage.create_system_message(
        "You are a helpful assistant with tool calling capabilities. "
        "Use the tools provided to answer the user's questions accurately."
    ),
    ChatMessage.create_user_message(
        "What's the weather in London and New York? Also, calculate 42 * 8 and tell me the current date."
    )
]

# Get a response from the agent
response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

# Print the response
print(response.response)
```

## Using Streaming Responses

For a more responsive experience, you can use streaming responses:

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
        chat_response = chunk.finished_response
```

## Complete Example

Here's the complete example:

```python
import os
import math
import datetime
from dotenv import load_dotenv

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables
load_dotenv()

# Define tools
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos})

def current_datetime(output_format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Get the current date and time in the specified format."""
    return datetime.datetime.now().strftime(output_format)

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    weather_data = {
        "London, UK": {"temperature": 18, "condition": "Cloudy"},
        "New York, USA": {"temperature": 25, "condition": "Sunny"},
    }
    default_weather = {"temperature": 20, "condition": "Clear"}
    weather = weather_data.get(location, default_weather)
    temp = weather["temperature"]
    if unit.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
    return f"The weather in {location} is {temp}°{'F' if unit.lower() == 'fahrenheit' else 'C'} and {weather['condition'].lower()}."

# Create tool instances
calculator_tool = FunctionTool(calculator)
datetime_tool = FunctionTool(current_datetime)
weather_tool = FunctionTool(get_weather)

# Set up API and agent
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.45

# Create tool registry
tool_registry = ToolRegistry()
tool_registry.add_tools([calculator_tool, datetime_tool, weather_tool])

# Define messages
messages = [
    ChatMessage.create_system_message(
        "You are a helpful assistant with tool calling capabilities. "
        "Use the tools provided to answer the user's questions accurately."
    ),
    ChatMessage.create_user_message(
        "What's the weather in London and New York? Also, calculate 42 * 8 and tell me the current date."
    )
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

## Next Steps

Now that you've created a basic agent with ToolAgents, you can:

- Learn how to [create custom tools](../guides/custom-tools.md)
- Set up [chat history](../guides/chat-history.md) for ongoing conversations
- Explore [other LLM providers](../components/providers.md)
- Check out [advanced agent examples](../examples/advanced-agents.md)
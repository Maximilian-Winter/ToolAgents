---
title: Basic Agents
---

# Basic Agent Examples

This page provides practical examples of creating and using basic agents with ToolAgents.

## Simple Question-Answering Agent

This example shows how to create a basic agent that can answer questions without using tools:

```python
import os
from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables
load_dotenv()

# Initialize the API
api = OpenAIChatAPI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

# Create the agent
agent = ChatToolAgent(chat_api=api)

# Configure settings
settings = api.get_default_settings()
settings.temperature = 0.7

# Create messages
messages = [
    ChatMessage.create_system_message(
        "You are a helpful, concise assistant. Always provide accurate information."
    ),
    ChatMessage.create_user_message(
        "What are the key features of Python programming language?"
    )
]

# Get response
response = agent.get_response(
    messages=messages,
    settings=settings
)

# Print the response
print(response.response)
```

## Calculator Agent

This example creates an agent with a calculator tool:

```python
import os
from dotenv import load_dotenv

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables
load_dotenv()

# Define calculator tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression to evaluate.
            Example: "2 + 2", "5 * 10", "sqrt(16)"
    
    Returns:
        The result of the evaluated expression as a float.
    """
    import math
    
    # Define allowed functions
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    
    # For safety in a real app, you would want to validate expressions
    return eval(expression, {"__builtins__": {}}, safe_dict)

calculator_tool = FunctionTool(calculator)

# Set up API and agent
api = OpenAIChatAPI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()

# Register tool
tool_registry = ToolRegistry()
tool_registry.add_tool(calculator_tool)

# Create messages
messages = [
    ChatMessage.create_system_message(
        "You are a helpful math assistant. Use the calculator tool to solve math problems."
    ),
    ChatMessage.create_user_message(
        "Calculate the area of a circle with radius 5, and then multiply the result by 2."
    )
]

# Get response
response = agent.get_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

# Print the response
print(response.response)
```

## Weather and Time Agent

This example creates an agent with weather and time tools:

```python
import os
import datetime
from dotenv import load_dotenv

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables
load_dotenv()

# Define time tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time in the specified timezone.
    
    Args:
        timezone: The timezone to get the time for. Default is "UTC".
            Examples: "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"
    
    Returns:
        The current time as a formatted string.
    """
    from datetime import datetime
    import pytz
    
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}. Please use a valid timezone name."

# Define weather tool (mock)
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country for the weather report.
            Example: "London, UK", "New York, USA", "Tokyo, Japan"
        unit: The temperature unit, either "celsius" or "fahrenheit".
            Default is "celsius".
    
    Returns:
        A string with the weather information.
    """
    # This is a mock implementation - in a real app, you would call a weather API
    weather_data = {
        "London, UK": {"temperature": 18, "condition": "Cloudy", "humidity": 70},
        "New York, USA": {"temperature": 25, "condition": "Sunny", "humidity": 50},
        "Tokyo, Japan": {"temperature": 22, "condition": "Rainy", "humidity": 85},
        "Paris, France": {"temperature": 20, "condition": "Partly Cloudy", "humidity": 60},
        "Sydney, Australia": {"temperature": 28, "condition": "Clear", "humidity": 45},
    }
    
    # Default weather for locations not in our mock data
    default_weather = {"temperature": 20, "condition": "Clear", "humidity": 60}
    
    weather = weather_data.get(location, default_weather)
    temp = weather["temperature"]
    
    if unit.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return f"The weather in {location} is {temp}°{'F' if unit.lower() == 'fahrenheit' else 'C'}, " \
           f"{weather['condition'].lower()} with {weather['humidity']}% humidity."

# Create tools
time_tool = FunctionTool(get_current_time)
weather_tool = FunctionTool(get_weather)

# Set up API and agent
api = OpenAIChatAPI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.7

# Register tools
tool_registry = ToolRegistry()
tool_registry.add_tools([time_tool, weather_tool])

# Create chat history
chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful assistant that can provide weather and time information. "
    "Use the tools available to you to answer questions accurately."
)

# Interactive chat loop
print("Chat with the Weather and Time Assistant (type 'exit' to quit)")
print("-----------------------------------------------------------")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    
    # Add user message to history
    chat_history.add_user_message(user_input)
    
    # Get response
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tool_registry
    )
    
    # Print the response
    print(f"\nAssistant: {response.response}")
    
    # Add response messages to history
    chat_history.add_messages(response.messages)
```

## Multi-Provider Agent

This example shows how to use the same agent with different providers:

```python
import os
from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import (
    OpenAIChatAPI,
    AnthropicChatAPI,
    MistralChatAPI
)

# Load environment variables
load_dotenv()

# Define a function to test different providers
def test_provider(api, provider_name):
    # Create the agent
    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = 0.7
    
    # Create messages
    messages = [
        ChatMessage.create_system_message(
            "You are a helpful, concise assistant."
        ),
        ChatMessage.create_user_message(
            "Explain the concept of machine learning in one paragraph."
        )
    ]
    
    # Get response
    print(f"\n--- {provider_name} Response ---")
    response = agent.get_response(
        messages=messages,
        settings=settings
    )
    
    print(response.response)

# Test OpenAI
openai_api = OpenAIChatAPI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
test_provider(openai_api, "OpenAI GPT-4o Mini")

# Test Anthropic (if key is available)
if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_api = AnthropicChatAPI(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-sonnet-20241022"
    )
    test_provider(anthropic_api, "Anthropic Claude 3.5 Sonnet")

# Test Mistral (if key is available)
if os.getenv("MISTRAL_API_KEY"):
    mistral_api = MistralChatAPI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-small-latest"
    )
    test_provider(mistral_api, "Mistral Small")
```

## Streaming Response Example

This example demonstrates streaming responses:

```python
import os
from dotenv import load_dotenv

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

# Load environment variables
load_dotenv()

# Define a simple calculator tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    import math
    safe_dict = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi}
    return eval(expression, {"__builtins__": {}}, safe_dict)

calculator_tool = FunctionTool(calculator)

# Set up API and agent
api = OpenAIChatAPI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()

# Register tool
tool_registry = ToolRegistry()
tool_registry.add_tool(calculator_tool)

# Create messages
messages = [
    ChatMessage.create_system_message(
        "You are a helpful math assistant. Use the calculator tool to solve problems."
    ),
    ChatMessage.create_user_message(
        "What is the area of a circle with radius 5? Then calculate the volume of a sphere with the same radius."
    )
]

# Get streaming response
print("Streaming response:")
stream = agent.get_streaming_response(
    messages=messages,
    settings=settings,
    tool_registry=tool_registry
)

for chunk in stream:
    print(chunk.chunk, end='', flush=True)
    
    # If this is the final chunk, we can access the full response
    if chunk.finished:
        final_response = chunk.finished_response
        print("\n\nFinal response complete!")
```

## Next Steps

- [Advanced agent examples](advanced-agents.md)
- [Web research agents](web-research.md)
- [Code interpreter examples](code-interpreter.md)
- [Memory examples](memory.md)
# Configuration Guide

This guide covers the configuration options for ToolAgents, including setting up different agent types, configuring providers, and creating custom tools.

## Agent Types

ToolAgents supports several agent types, each designed for specific LLM providers:

1. MistralAgent
2. LlamaAgent
3. ChatAPIAgent
4. OllamaAgent

### MistralAgent and LlamaAgent

These agents are used with llama.cpp, TGI, and vLLM servers.

```python
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider, LlamaCppSamplingSettings

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = MistralAgent(llm_provider=provider, debug_output=False,
                     system_prompt="You are a helpful assistant.")

settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.max_tokens = 4096
```

### ChatAPIAgent

This agent is used with OpenAI and Anthropic APIs.

```python
from ToolAgents.agents import ChatAPIAgent
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings

api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-sonnet-20240229")
agent = ChatAPIAgent(chat_api=api, system_prompt="You are a helpful assistant.")

settings = AnthropicSettings()
settings.temperature = 0.45
settings.top_p = 0.85
```

### OllamaAgent

This agent is used with Ollama.

```python
from ToolAgents.agents import OllamaAgent

agent = OllamaAgent(model='mistral-nemo', system_prompt="You are a helpful assistant.", debug_output=False)
```

## Provider Settings

Each provider has its own settings class:

- LlamaCppSamplingSettings
- VLLMServerSamplingSettings
- OpenAISettings
- AnthropicSettings

These classes allow you to configure provider-specific parameters like temperature, top_p, and max_tokens.

## Creating Custom Tools

ToolAgents supports three main ways to create custom tools:

### 1. Pydantic Model-based Tools

```python
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class Calculator(BaseModel):
    """Perform a math operation on two numbers."""
    number_one: float = Field(..., description="First number.")
    operation: str = Field(..., description="Math operation to perform.")
    number_two: float = Field(..., description="Second number.")

    def run(self):
        # Implementation...

calculator_tool = FunctionTool(Calculator)
```

### 2. Function-based Tools

```python
from ToolAgents import FunctionTool

def get_current_datetime(output_format: str = '%Y-%m-%d %H:%M:%S'):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
    """
    # Implementation...

current_datetime_tool = FunctionTool(get_current_datetime)
```

### 3. OpenAI-style Function Specifications

```python
from ToolAgents import FunctionTool

def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    # Implementation...

open_ai_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
}

weather_tool = FunctionTool.from_openai_tool(open_ai_tool_spec, get_current_weather)
```

## Using Tools with Agents

Once you've created your tools, you can pass them to your agent when making a request:

```python
tools = [calculator_tool, current_datetime_tool, weather_tool]

result = agent.get_response(
    "What's the weather like in New York and what's 42 times 42?",
    tools=tools,
    settings=settings
)
```

Remember to provide clear and comprehensive docstrings and descriptions for your tools to ensure optimal performance.

## Next Steps

- Explore the [API Reference](../api/index.md) for detailed information on each module and component.
- Check out the [Examples](../examples/index.md) for more advanced usage scenarios.
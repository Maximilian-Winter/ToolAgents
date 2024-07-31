# FunkyFlow

FunkyFlow is a lightweight, versatile framework for function calling agents. It supports various LLM providers including llama.cpp servers, TGI, vllm, Mistral's Nemo Agent, paid Chat APIs, and Ollama. FunkyFlow provides an easy interface for passing functions, Pydantic models, and tools from other frameworks to the LLM, performing function calls, and returning results to the LLM for response generation.

## Features

- Support for multiple LLM providers
- Easy integration of custom tools and functions
- Flexible API for different agent types
- Pydantic model support for type safety
- Streaming responses (for supported providers)


## Quick Start

Here's a simple example to get you started with FunkyFlow:

```python
from funkyflow import MistralAgent
from funkyflow.provider import LlamaCppServerProvider
from funkyflow.utilities import calculator_function_tool, current_datetime_function_tool

# Setup the LLM provider
provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

# Create an agent
agent = MistralAgent(llm_provider=provider, debug_output=True)

# Define tools
tools = [calculator_function_tool, current_datetime_function_tool]

# Get a response
result = agent.get_response(
    "What is the result of 42 * 42, and what's the current date?",
    tools=tools
)

print(result)
```

## Defining Tools

FunkyFlow supports multiple ways to define tools:

### 1. Using Pydantic Models

```python
from pydantic import BaseModel, Field
from funkyflow import FunctionTool

class Calculator(BaseModel):
    number_one: float = Field(..., description="First number")
    operation: str = Field(..., description="Math operation to perform")
    number_two: float = Field(..., description="Second number")

    def run(self):
        if self.operation == "add":
            return self.number_one + self.number_two
        # ... other operations

calculator_tool = FunctionTool(Calculator)
```

### 2. Using Simple Python Functions

```python
from funkyflow import FunctionTool

def get_current_datetime(output_format: str = '%Y-%m-%d %H:%M:%S'):
    import datetime
    return datetime.datetime.now().strftime(output_format)

datetime_tool = FunctionTool(get_current_datetime)
```

### 3. Using OpenAI-style Function Definitions

```python
from funkyflow import FunctionTool

def get_weather(location: str, unit: str):
    # Implementation here
    pass

weather_tool_spec = {
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

weather_tool = FunctionTool.from_openai_tool(weather_tool_spec, get_weather)
```

## Using Different Providers

FunkyFlow supports various LLM providers. Here are examples for a few:

### LlamaCpp Server

```python
from funkyflow import MistralAgent
from funkyflow.provider import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = MistralAgent(llm_provider=provider)
```

### VLLM Server

```python
from funkyflow import MistralAgent
from funkyflow.provider import VLLMServerProvider

provider = VLLMServerProvider("http://127.0.0.1:8000/")
agent = MistralAgent(llm_provider=provider)
```

### Chat API (e.g., OpenAI, Anthropic)

```python
from funkyflow import ChatAPIAgent
from funkyflow.utilities import OpenAIChatAPI, OpenAISettings

api = OpenAIChatAPI(api_key="your-api-key", base_url="https://api.openai.com/v1", model="gpt-3.5-turbo")
settings = OpenAISettings()
agent = ChatAPIAgent(chat_api=api)
```

### Ollama

```python
from funkyflow import OllamaAgent

agent = OllamaAgent(model='mistral-nemo')
```

## Advanced Usage

### Streaming Responses

Some providers support streaming responses:

```python
async for chunk in agent.get_streaming_response(
    message="Your question here",
    tools=your_tools,
):
    print(chunk, end='', flush=True)
```

### Custom System Prompts

You can set custom system prompts for some agents:

```python
agent = MistralAgent(
    llm_provider=provider,
    system_prompt="Always answer as an old drunken pirate."
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

FunkyFlow is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

# ToolAgents

ToolAgents is a lightweight and flexible framework for creating function-calling agents with various language models and APIs. It provides a unified interface for integrating different LLM providers and executing function calls seamlessly.

## Features

- Support for multiple LLM providers:
    - llama.cpp servers
    - Hugging Face's Text Generation Interface (TGI) servers
    - vLLM servers
    - OpenAI API
    - Anthropic API
    - Ollama (with Tool calling support)
- Easy-to-use interface for passing functions, Pydantic models, and tools to LLMs
- Streamlined process for function calling and result handling
- Flexible agent types:
    - MistralAgent for llama.cpp, TGI, and vLLM servers
    - ChatAPIAgent for OpenAI and Anthropic APIs
    - OllamaAgent for Ollama integration

## Installation


```bash
pip install toolagents
```

## Usage

### MistralAgent with llama.cpp Server

```python
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider, LlamaCppSamplingSettings
from ToolAgents.tests.test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

# Initialize the provider and agent
provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = MistralAgent(llm_provider=provider, debug_output=False,
                     system_prompt="You are a helpful assistant.")

# Configure settings
settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.max_tokens = 4096

# Define tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

# Get a response
result = agent.get_streaming_response(
    "Perform the following tasks: Get the current weather in Celsius in London, New York, and at the North Pole. "
    "Solve these calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6, and 96/8.",
    sampling_settings=settings,
    tools=tools
)

for token in result:
    print(token, end="", flush=True)
print()
```

### ChatAPIAgent with Anthropic API

```python
import os
from dotenv import load_dotenv
from ToolAgents.agents import ChatAPIAgent
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings
from ToolAgents.tests.test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

load_dotenv()

# Initialize the API and agent
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-sonnet-20240229")
agent = ChatAPIAgent(chat_api=api, system_prompt="You are a helpful assistant.")

# Configure settings
settings = AnthropicSettings()
settings.temperature = 0.45
settings.top_p = 0.85

# Define tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

# Get a response
result = agent.get_response(
    "Perform the following tasks: Get the current weather in Celsius in London, New York, and at the North Pole. "
    "Solve these calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6, and 96/8.",
    tools=tools,
    settings=settings
)

print(result)
```

### OllamaAgent

```python
from ToolAgents.agents import OllamaAgent
from ToolAgents.tests.test_tools import get_flight_times_tool

def run():
    agent = OllamaAgent(model='mistral-nemo', system_prompt="You are a helpful assistant.", debug_output=False)

    tools = [get_flight_times_tool]

    response = agent.get_response(
        message="What is the flight time from New York (NYC) to Los Angeles (LAX)?",
        tools=tools,
    )

    print(response)

    print("\nStreaming response:")
    for chunk in agent.get_streaming_response(
            message="What is the flight time from London (LHR) to New York (JFK)?",
            tools=tools,
    ):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    run()
```

## Custom Tools

You can create custom tools using Pydantic models or function definitions. Here's an example of a custom calculator tool:

```python
from enum import Enum
from typing import Union

from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: Union[int, float] = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: Union[int, float] = Field(..., description="Second number.")

    def run(self):
        if self.operation == MathOperation.ADD:
            return self.number_one + self.number_two
        elif self.operation == MathOperation.SUBTRACT:
            return self.number_one - self.number_two
        elif self.operation == MathOperation.MULTIPLY:
            return self.number_one * self.number_two
        elif self.operation == MathOperation.DIVIDE:
            return self.number_one / self.number_two
        else:
            raise ValueError("Unknown operation.")

calculator_tool = FunctionTool(Calculator)
```

## Contributing

Contributions to ToolAgents are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License

MIT License
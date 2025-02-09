# ToolAgents

ToolAgents is a lightweight and flexible framework for creating function-calling agents with various language models and APIs. It provides a unified interface for integrating different LLM providers and executing function calls seamlessly.


## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
  - [MistralAgent with llama.cpp Server](#mistralagent-with-llamacpp-server)
  - [LlamaAgent with llama.cpp Server](#llamaagent-with-llamacpp-server)
  - [ChatAPIAgent with Anthropic API](#chatapiagent-with-anthropic-api)
  - [OllamaAgent](#ollamaagent)
4. [Custom Tools](#custom-tools)
  - [Pydantic Model-based Tools](#1-pydantic-model-based-tools)
  - [Function-based Tools](#2-function-based-tools)
  - [OpenAI-style Function Specifications](#3-openai-style-function-specifications)
  - [The Importance of Good Docstrings and Descriptions](#the-importance-of-good-docstrings-and-descriptions)
5. [Contributing](#contributing)
6. [License](#license)

## Features

- Support for multiple LLM providers:
  - llama.cpp servers
  - Hugging Face's Text Generation Interface (TGI) servers
  - vLLM servers
  - OpenAI API
  - Anthropic API
  - Mistral API
  - Ollama (with Tool calling support)
- Easy-to-use interface for passing functions, Pydantic models, and tools to LLMs
- Streamlined process for function calling and result handling
- Flexible agent types:
  - MistralAgent for llama.cpp, TGI, and vLLM servers
  - LlamaAgent for llama.cpp, TGI and vLLM servers
  - Customizable TemplateAgent for llama.cpp, TGI, and vLLM servers
  - ChatAPIAgent for OpenAI and Anthropic APIs
  - OllamaAgent for Ollama integration

## Installation

```bash
pip install ToolAgents
```

## Usage

### MistralAgent with llama.cpp Server

```python
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider, LlamaCppSamplingSettings
from ToolAgents.utilities import ChatHistory
from ToolAgents import ToolRegistry
from test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

# Initialize the provider and agent
provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = MistralAgent(provider=provider, debug_output=False)

# Configure settings
settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.max_tokens = 4096

# Define tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

# Create chat history and add system message and user message.
chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant.")
chat_history.add_user_message("Perform the following tasks: Get the current weather in Celsius in London, New York, and at the North Pole. Solve these calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6, and 96/8.")
# Get a response
result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings,
    tool_registry=tool_registry
)

for token in result:
    print(token, end="", flush=True)
print()

# Add the generated messages, including tool messages, to the chat history.
chat_history.add_list_of_dicts(agent.last_messages_buffer)

# Save chat history to file.
chat_history.save_history("./chat_history.json")
```

### LlamaAgent with llama.cpp Server

```python
from ToolAgents.agents import Llama31Agent
from ToolAgents.provider import LlamaCppServerProvider, LlamaCppSamplingSettings
from ToolAgents.utilities import ChatHistory
from ToolAgents import ToolRegistry
from test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

# Initialize the provider and agent
provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = Llama31Agent(provider=provider, debug_output=False)

# Configure settings
settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.max_tokens = 4096

# Define tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

# Create chat history and add system message and user message.
chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant.")
chat_history.add_user_message("Perform the following tasks: Get the current weather in Celsius in London, New York, and at the North Pole. Solve these calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6, and 96/8.")


# Get a response
result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings,
    tools=tools
)

for token in result:
    print(token, end="", flush=True)
print()

# Add the generated messages, including tool messages, to the chat history.
chat_history.add_list_of_dicts(agent.last_messages_buffer)

# Save chat history to file.
chat_history.save_history("./chat_history.json")
```
### ChatAPIAgent with Anthropic API

```python
import os
from dotenv import load_dotenv
from ToolAgents.agents import ChatAPIAgent
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings
from ToolAgents.utilities import ChatHistory
from ToolAgents import ToolRegistry
from test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

load_dotenv()

# Initialize the API and agent
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-sonnet-20240229")
agent = ChatAPIAgent(chat_api=api)

# Configure settings
settings = AnthropicSettings()
settings.temperature = 0.45
settings.top_p = 0.85

# Define tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

# Create chat history and add system message and user message.
chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant.")
chat_history.add_user_message("Perform the following tasks: Get the current weather in Celsius in London, New York, and at the North Pole. Solve these calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6, and 96/8.")


# Get a response
result = agent.get_response(
    messages=chat_history.to_list(),
    tools=tools,
    settings=settings
)

print(result)

# Add the generated messages, including tool messages, to the chat history.
chat_history.add_list_of_dicts(agent.last_messages_buffer)

# Save chat history to file.
chat_history.save_history("./chat_history.json")
```

### OllamaAgent

```python
from ToolAgents.agents import OllamaAgent
from ToolAgents.utilities import ChatHistory
from ToolAgents import ToolRegistry

from test_tools import get_flight_times_tool

def run():
    agent = OllamaAgent(model='mistral-nemo', debug_output=False)

    # Define tools
    tools = [get_flight_times_tool]
    
    tool_registry = ToolRegistry()
    
    tool_registry.add_tools(tools)

    # Create chat history and add system message and user message.
    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful assistant.")
    chat_history.add_user_message("What is the flight time from New York (NYC) to Los Angeles (LAX)?")

    response = agent.get_response(
            messages=chat_history.to_list(),
            tool_registry=tool_registry,
        )

    print(response)
    
    # Add the generated messages, including tool messages, to the chat history.
    chat_history.add_list_of_dicts(agent.last_messages_buffer)
    
    chat_history.add_user_message("What is the flight time from London (LHR) to New York (JFK)?")
    print("\nStreaming response:")
    for chunk in agent.get_streaming_response(
            messages=chat_history.to_list(),
            tools=tools,
    ):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    run()
```

## Custom Tools

ToolAgents supports various ways to create custom tools, allowing you to integrate specific functionalities into your agents. Here are different approaches to creating custom tools:

### 1. Pydantic Model-based Tools

You can create tools using Pydantic models, which provide strong typing and automatic validation. Here's an example of a calculator tool:

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

### 2. Function-based Tools

You can also create tools from simple Python functions. Here's an example of a datetime tool:

```python
import datetime
from ToolAgents import FunctionTool

def get_current_datetime(output_format: str = '%Y-%m-%d %H:%M:%S'):
    """
    Get the current date and time in the given format.

    Args:
        output_format: formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
    """
    return datetime.datetime.now().strftime(output_format)

current_datetime_tool = FunctionTool(get_current_datetime)
```

### 3. OpenAI-style Function Specifications

ToolAgents supports creating tools from OpenAI-style function specifications:

```python
from ToolAgents import FunctionTool

def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    # Implementation details...

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

### The Importance of Good Docstrings and Descriptions

When creating custom tools, it's crucial to provide clear and comprehensive docstrings and descriptions. Here's why they matter:

1. **AI Understanding**: The language model uses these descriptions to understand the purpose and functionality of each tool. Better descriptions lead to more accurate tool selection and usage.

2. **Parameter Clarity**: Detailed descriptions for each parameter help the AI understand what input is expected, reducing errors and improving the quality of the generated calls.

3. **Proper Usage**: Good docstrings guide the AI on how to use the tool correctly, including any specific formats or constraints for the input.

4. **Error Prevention**: By clearly stating the expected input types and any limitations, you can prevent many potential errors before they occur.

Here's an example of a well-documented tool:

```python
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class FlightTimes(BaseModel):
    """
    Retrieve flight information between two locations.

    This tool provides estimated flight times, including departure and arrival times,
    for flights between major airports. It uses airport codes for input.
    """

    departure: str = Field(
        ...,
        description="The departure airport code (e.g., 'NYC' for New York)",
        min_length=3,
        max_length=3
    )
    arrival: str = Field(
        ...,
        description="The arrival airport code (e.g., 'LAX' for Los Angeles)",
        min_length=3,
        max_length=3
    )

    def run(self) -> str:
        """
        Retrieve flight information for the given departure and arrival locations.

        Returns:
            str: A JSON string containing flight information including departure time,
                 arrival time, and flight duration. If no flight is found, returns an error message.
        """
        # Implementation details...

get_flight_times_tool = FunctionTool(FlightTimes)
```

In this example, the docstrings and field descriptions provide clear information about the tool's purpose, input requirements, and expected output, enabling both the AI and human developers to use the tool effectively.

## Contributing

Contributions to ToolAgents are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License

ToolAgents is released under the MIT License. See the [LICENSE](LICENSE) file for details.

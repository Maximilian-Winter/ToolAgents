# ToolAgents

ToolAgents is a lightweight and flexible framework for creating function-calling agents with various language models and APIs. It provides a unified interface for integrating different LLM providers and executing function calls seamlessly.


## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
  - [Simple ChatToolAgent Usage](#ChatToolAgent)
  - [Using different Providers](#Different-Providers)
  - [ChatToolAgent with User Loop and Chat History](#Use-ChatToolAgent-with-ChatHistory-class)
  - [Streaming ChatToolAgent with User Loop and Chat History](#Use-Streaming-ChatToolAgent-with-ChatHistory-class)
4. [Custom Tools](#custom-tools)
  - [Pydantic Model-based Tools](#1-pydantic-model-based-tools)
  - [Function-based Tools](#2-function-based-tools)
  - [OpenAI-style Function Specifications](#3-openai-style-function-specifications)
  - [The Importance of Good Docstrings and Descriptions](#the-importance-of-good-docstrings-and-descriptions)
5. [Contributing](#contributing)
6. [License](#license)

## Features

- Support for multiple LLM providers:
  - OpenAI API
  - Anthropic API
  - Mistral API
  - OpenAI like API, like OpenRouter, VLLM, llama-cpp-server
- Easy-to-use interface for passing functions, Pydantic models, and tools to LLMs
- Streamlined process for function calling and result handling
- Unified Message format, making switching of providers while keeping the same chat history easy.

## Installation

```bash
pip install ToolAgents
```

## Usage

### ChatToolAgent

```python
import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Official OpenAI API
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.45
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)
messages = [
    ChatMessage.create_system_message("You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."),
    ChatMessage.create_user_message("Get the weather in London and New York. Calculate 420 x 420 and retrieve the date and time in the format: %Y-%m-%d %H:%M:%S.")
]

result = agent.get_streaming_response(
    messages=messages,
    settings=settings, tool_registry=tool_registry)


for res in result:
    print(res.chunk, end='', flush=True)

```

### Different Providers
```python
# Import different providers
from ToolAgents.provider import AnthropicChatAPI, OpenAIChatAPI, GroqChatAPI, MistralChatAPI, CompletionProvider

# Official OpenAI API
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Local OpenAI like API, like vllm or llama-cpp-server
api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# Anthropic API
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")

# Groq API
api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")


# Mistral API
api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

```

### Use ChatToolAgent with ChatHistory class
```python
import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.provider import OpenAIChatAPI

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Openrouter API
api = OpenAIChatAPI(api_key=os.getenv("OPENROUTER_API_KEY"), model="google/gemini-2.0-pro-exp-02-05:free", base_url="https://openrouter.ai/api/v1")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question.")

while True:
    user_input = input("User input >")
    if user_input == "quit":
        break
    elif user_input == "save":
        chat_history.save_to_json("example_chat_history.json")
    elif user_input == "load":
        chat_history = ChatHistory.load_from_json("example_chat_history.json")
    else:
        chat_history.add_user_message(user_input)

        chat_response = agent.get_response(
            messages=chat_history.get_messages(),
            settings=settings, tool_registry=tool_registry)

        print(chat_response.response.strip())
        chat_history.add_messages(chat_response.messages)

```

### Use Streaming ChatToolAgent with ChatHistory class
```python
import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.provider import OpenAIChatAPI

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Openrouter API
api = OpenAIChatAPI(api_key=os.getenv("OPENROUTER_API_KEY"), model="google/gemini-2.0-pro-exp-02-05:free", base_url="https://openrouter.ai/api/v1")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question.")

while True:
    user_input = input("User input >")
    if user_input == "quit":
        break
    elif user_input == "save":
        chat_history.save_to_json("example_chat_history.json")
    elif user_input == "load":
        chat_history = ChatHistory.load_from_json("example_chat_history.json")
    else:
        chat_history.add_user_message(user_input)

        stream = agent.get_streaming_response(
            messages=chat_history.get_messages(),
            settings=settings, tool_registry=tool_registry)
        chat_response = None
        for res in stream:
            print(res.chunk, end='', flush=True)
            if res.finished:
              chat_response = res.finished_response
        if chat_response is not None:
            chat_history.add_messages(chat_response.messages)
        else:
          raise Exception("Error during response generation")
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

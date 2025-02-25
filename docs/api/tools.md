---
title: Tools API
---

# Tools API

The Tools module provides functionality for creating, managing, and executing tools that can be used by language models.

## FunctionTool

`FunctionTool` represents a tool that can be called by an LLM agent.

```python
from ToolAgents import FunctionTool

# Create from a function
function_tool = FunctionTool(your_function)

# Create from a Pydantic model
model_tool = FunctionTool(YourPydanticModel)
```

### Constructor Parameters

- `function_tool`: Can be:
  - A Python function with type hints and docstrings
  - A Pydantic model with a `run` method
  - A tuple of (OpenAI tool spec, function implementation)
- `pre_processors` (list, optional): Functions/classes to process parameters before execution
- `post_processors` (list, optional): Functions/classes to process results after execution
- `debug_mode` (bool, optional): Whether to output debugging information

### Methods

#### `execute(parameters)`

Executes the function with given parameters.

**Parameters:**
- `parameters` (dict): The parameters to pass to the function

**Returns:**
- The result of the function execution

#### `add_pre_processor(processor, position=None)`

Adds a parameter preprocessor.

**Parameters:**
- `processor`: The preprocessor function or class
- `position` (int, optional): Position in the preprocessing chain

#### `add_post_processor(processor, position=None)`

Adds a result postprocessor.

**Parameters:**
- `processor`: The postprocessor function or class
- `position` (int, optional): Position in the postprocessing chain

#### `to_mistral_tool()`

Converts the tool to Mistral API format.

**Returns:**
- Mistral tool specification dictionary

#### `to_openai_tool()`

Converts the tool to OpenAI API format.

**Returns:**
- OpenAI tool specification dictionary

#### `to_anthropic_tool()`

Converts the tool to Anthropic API format.

**Returns:**
- Anthropic tool specification dictionary

#### `get_json_schema()`

Gets the JSON schema for the tool.

**Returns:**
- JSON schema dictionary

#### `from_openai_tool(tool_spec, implementation_function)`

Creates a FunctionTool from an OpenAI tool specification.

**Parameters:**
- `tool_spec` (dict): OpenAI tool specification
- `implementation_function` (callable): Function that implements the tool

**Returns:**
- `FunctionTool`: A new function tool instance

## ToolRegistry

`ToolRegistry` manages a collection of tools available to agents.

```python
from ToolAgents import ToolRegistry

# Create a registry
tool_registry = ToolRegistry()

# Add tools
tool_registry.add_tool(your_tool)
tool_registry.add_tools([tool1, tool2, tool3])
```

### Methods

#### `add_tool(tool)`

Adds a tool to the registry.

**Parameters:**
- `tool` (FunctionTool): The tool to add

#### `add_tools(tools)`

Adds multiple tools to the registry.

**Parameters:**
- `tools` (list): List of FunctionTool instances

#### `get_tool(name)`

Gets a tool by name.

**Parameters:**
- `name` (str): The name of the tool

**Returns:**
- `FunctionTool`: The tool instance or None if not found

#### `has_tool(name)`

Checks if a tool exists in the registry.

**Parameters:**
- `name` (str): The name of the tool

**Returns:**
- `bool`: True if the tool exists

#### `get_tools()`

Gets all registered tools.

**Returns:**
- `list`: List of FunctionTool instances

#### `get_mistral_tools()`

Gets tools in Mistral API format.

**Returns:**
- `list`: List of Mistral tool specifications

#### `get_openai_tools()`

Gets tools in OpenAI API format.

**Returns:**
- `list`: List of OpenAI tool specifications

#### `get_anthropic_tools()`

Gets tools in Anthropic API format.

**Returns:**
- `list`: List of Anthropic tool specifications

## Processors

Processors allow modifying parameters before tool execution and results after execution.

### BaseProcessor

Abstract base class for processors.

```python
from ToolAgents.function_tool import BaseProcessor

class CustomProcessor(BaseProcessor):
    def process(self, data):
        # Transform data
        return transformed_data
```

### PreProcessor

Base class for parameter preprocessing.

```python
from ToolAgents.function_tool import PreProcessor

class CustomPreProcessor(PreProcessor):
    def process(self, parameters):
        # Transform parameters
        return transformed_parameters
```

### PostProcessor

Base class for result postprocessing.

```python
from ToolAgents.function_tool import PostProcessor

class CustomPostProcessor(PostProcessor):
    def process(self, result):
        # Transform result
        return transformed_result
```

## Function Tool Creation Examples

### From a Python Function

```python
from ToolAgents import FunctionTool

def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression.
            Example: "2 + 2", "5 * 10"
    
    Returns:
        The result of the evaluated expression.
    """
    return eval(expression, {"__builtins__": {}}, {})

calculator_tool = FunctionTool(calculator)
```

### From a Pydantic Model

```python
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: float = Field(..., description="First number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")
    number_two: float = Field(..., description="Second number")
    
    def run(self):
        """Execute the calculator operation."""
        if self.operation == "add":
            return self.number_one + self.number_two
        elif self.operation == "subtract":
            return self.number_one - self.number_two
        elif self.operation == "multiply":
            return self.number_one * self.number_two
        elif self.operation == "divide":
            return self.number_one / self.number_two
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

calculator_tool = FunctionTool(Calculator)
```

### From an OpenAI Tool Specification

```python
from ToolAgents import FunctionTool

def get_weather(location, unit):
    """Get the current weather in a given location"""
    # Implementation details...
    return f"Weather in {location}: 22°{unit[0].upper()}"

weather_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_weather",
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

### With Processors

```python
from ToolAgents import FunctionTool

def location_info(city: str, country: str) -> str:
    """Get information about a location"""
    return f"Information about {city}, {country}"

# Define a preprocessor
def normalize_location(parameters):
    parameters["city"] = parameters["city"].strip().title()
    parameters["country"] = parameters["country"].strip().upper()
    return parameters

# Define a postprocessor
def format_response(result):
    return f"LOCATION INFO: {result}"

# Create tool with processors
location_tool = FunctionTool(
    location_info,
    pre_processors=[normalize_location],
    post_processors=[format_response]
)
```
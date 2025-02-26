---
title: Custom Tools
---

# Creating Custom Tools

One of the most powerful features of ToolAgents is the ability to create custom tools that your agents can use. This guide explores different approaches to creating custom tools.

## Tool Creation Approaches

ToolAgents supports three main approaches to creating custom tools:

1. **Pydantic Model-based Tools**: Define tools using Pydantic models with strong typing and validation
2. **Function-based Tools**: Create tools from Python functions with type hints and docstrings
3. **OpenAI-style Function Specifications**: Define tools using OpenAI's function calling schema

## 1. Pydantic Model-based Tools

Pydantic models provide a robust way to define tools with rich typing, validation, and documentation:

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
    number_one: Union[int, float] = Field(..., description="First number for the operation.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: Union[int, float] = Field(..., description="Second number for the operation.")

    def run(self):
        """Execute the calculator operation."""
        if self.operation == MathOperation.ADD:
            return self.number_one + self.number_two
        elif self.operation == MathOperation.SUBTRACT:
            return self.number_one - self.number_two
        elif self.operation == MathOperation.MULTIPLY:
            return self.number_one * self.number_two
        elif self.operation == MathOperation.DIVIDE:
            if self.number_two == 0:
                return "Error: Division by zero"
            return self.number_one / self.number_two
        else:
            raise ValueError("Unknown operation.")

# Create the tool
calculator_tool = FunctionTool(Calculator)
```

### Key Elements of Pydantic Model Tools

1. **Class Docstring**: Provides a description of the tool's purpose
2. **Fields with Type Annotations**: Define the inputs with proper typing
3. **Field Descriptions**: Help the LLM understand parameter usage
4. **run() Method**: Implements the tool's functionality
5. **Validation**: Pydantic handles input validation automatically

## 2. Function-based Tools

You can also create tools from Python functions with type hints and docstrings:

```python
import datetime
from ToolAgents import FunctionTool

def get_current_datetime(output_format: str = '%Y-%m-%d %H:%M:%S'):
    """
    Get the current date and time in the given format.

    Args:
        output_format: The format string to use for the date and time output.
            Default is '%Y-%m-%d %H:%M:%S'.
            
    Returns:
        A string containing the formatted current date and time.
    """
    return datetime.datetime.now().strftime(output_format)

# Create the tool
datetime_tool = FunctionTool(get_current_datetime)
```

### Key Elements of Function-based Tools

1. **Function Docstring**: Describes the tool's purpose and usage
2. **Type Annotations**: Define the parameter types and return type
3. **Parameter Documentation**: Explain each parameter in the docstring
4. **Default Values**: Provide sensible defaults when appropriate

## 3. OpenAI-style Function Specifications

If you already have tools defined in OpenAI's function calling format, you can use them with ToolAgents:

```python
from ToolAgents import FunctionTool

def get_weather(location, unit):
    """Get the current weather in a given location"""
    # Implementation details...
    weather_data = {
        "New York": {"temp": 22, "condition": "sunny"},
        "London": {"temp": 15, "condition": "rainy"},
    }
    weather = weather_data.get(location, {"temp": 20, "condition": "unknown"})
    
    if unit == "fahrenheit":
        temp = (weather["temp"] * 9/5) + 32
    else:
        temp = weather["temp"]
        
    return f"{location}: {temp}°{'F' if unit == 'fahrenheit' else 'C'}, {weather['condition']}"

# OpenAI-style function specification
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
                    "description": "The city name, e.g. New York, London",
                },
                "unit": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                },
            },
            "required": ["location", "unit"],
        },
    },
}

# Create the tool from the OpenAI specification
weather_tool = FunctionTool.from_openai_tool(weather_tool_spec, get_weather)
```

## Pre-processors and Post-processors

To add additional functionality to your tools, you can use pre-processors and post-processors:

```python
from ToolAgents import FunctionTool

def log_query(country: str):
    """
    Get information about a country.
    
    Args:
        country: The name of the country
    """
    return f"Information about {country}"

# Define pre-processor and post-processor
def query_pre_processor(parameters):
    print(f"Query received for country: {parameters['country']}")
    parameters['country'] = parameters['country'].strip().title()
    return parameters

def query_post_processor(result):
    return f"{result} (processed at {datetime.datetime.now()})"

# Create tool with processors
country_info_tool = FunctionTool(
    log_query,
    pre_processor=query_pre_processor,
    post_processor=query_post_processor
)
```

## Tool Documentation Best Practices

The quality of your tool documentation directly impacts how effectively LLMs will use your tools:

1. **Clear Purpose**: Start with a clear description of what the tool does
2. **Parameter Details**: Describe each parameter's purpose, expected format, and constraints
3. **Examples**: Include examples of valid inputs in parameter descriptions
4. **Error Conditions**: Document potential error scenarios
5. **Return Value**: Explain what the tool returns and in what format

Here's an example of well-documented tool:

```python
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class WeatherQuery(BaseModel):
    """
    Retrieve current weather information for a specified location.
    
    This tool connects to a weather service API to get real-time
    weather data including temperature, conditions, humidity, and wind.
    """
    
    location: str = Field(
        ..., 
        description="The city and country/state for the weather lookup. "
                   "Examples: 'New York, NY', 'London, UK', 'Tokyo, Japan'."
    )
    
    units: str = Field(
        "metric",
        description="The unit system to use for temperature and other measurements. "
                   "Options: 'metric' (Celsius, km/h), 'imperial' (Fahrenheit, mph)."
    )
    
    include_forecast: bool = Field(
        False,
        description="Whether to include a 3-day forecast in the response. "
                   "Default is False (current conditions only)."
    )
    
    def run(self):
        """
        Execute the weather query and return formatted weather information.
        
        Returns:
            A string containing the current weather conditions and optional forecast.
            
        Raises:
            ValueError: If the location cannot be found or the API request fails.
        """
        # Implementation...
        
weather_query_tool = FunctionTool(WeatherQuery)
```

## Next Steps

Now that you know how to create custom tools, you can:

- [Manage chat history](chat-history.md) in your applications
- [Use streaming responses](streaming.md) for better user experience
- [Learn about different agent types](../components/agents.md)
- [Explore example applications](../examples/basic-agents.md)
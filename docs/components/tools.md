---
title: Tools
---

# Tools

Tools are the capabilities that you provide to your agents, allowing them to perform specific actions or access external services. ToolAgents offers a flexible system for creating and managing tools.

## Core Tool Concepts

### ToolRegistry

The `ToolRegistry` is a container for organizing and managing tools:

```python
from ToolAgents import ToolRegistry

# Create a registry
tool_registry = ToolRegistry()

# Add tools
tool_registry.add_tool(your_tool)
tool_registry.add_tools([tool1, tool2, tool3])

# Get a tool by name
calculator_tool = tool_registry.get_tool("calculator")

# Check if a tool exists
has_weather_tool = tool_registry.has_tool("get_weather")
```

### FunctionTool

`FunctionTool` is the primary way to create tools in ToolAgents. It can wrap Python functions, Pydantic models, or OpenAI-style function specifications:

```python
from ToolAgents import FunctionTool

# Create from a function
function_tool = FunctionTool(your_function)

# Create from a Pydantic model
model_tool = FunctionTool(YourPydanticModel)

# Create from an OpenAI tool spec
openai_tool = FunctionTool.from_openai_tool(openai_tool_spec, implementation_function)
```

## Creating Tools

### Function-Based Tools

Create tools from Python functions with type hints and docstrings:

```python
from ToolAgents import FunctionTool

def calculate_age(birth_year: int, current_year: int = 2024) -> int:
    """
    Calculate a person's age based on their birth year.
    
    Args:
        birth_year: The year the person was born
        current_year: The current year (defaults to 2024)
        
    Returns:
        The person's age in years
    """
    return current_year - birth_year

age_calculator = FunctionTool(calculate_age)
```

### Pydantic Model-Based Tools

Create more complex tools using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from ToolAgents import FunctionTool

class SearchTool(BaseModel):
    """
    Search for information on a given topic.
    """
    query: str = Field(..., description="The search query")
    num_results: int = Field(5, description="Number of results to return")
    include_snippets: bool = Field(True, description="Whether to include text snippets")
    
    def run(self):
        """Execute the search and return results."""
        # Implementation...
        results = [f"Result {i} for '{self.query}'" for i in range(self.num_results)]
        return results

search_tool = FunctionTool(SearchTool)
```

### OpenAI-Style Tools

Create tools using OpenAI's function calling format:

```python
from ToolAgents import FunctionTool

def translate_text(text, source_language, target_language):
    """Translate text between languages"""
    # Implementation...
    return f"Translated: {text} from {source_language} to {target_language}"

# OpenAI-style specification
translate_spec = {
    "type": "function",
    "function": {
        "name": "translate_text",
        "description": "Translate text from one language to another",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to translate"
                },
                "source_language": {
                    "type": "string",
                    "description": "The source language code (e.g., 'en', 'fr', 'es')"
                },
                "target_language": {
                    "type": "string",
                    "description": "The target language code (e.g., 'en', 'fr', 'es')"
                }
            },
            "required": ["text", "source_language", "target_language"]
        }
    }
}

translate_tool = FunctionTool.from_openai_tool(translate_spec, translate_text)
```

## Advanced Tool Features

### Pre-processors and Post-processors

Add processing steps before and after tool execution:

```python
from ToolAgents import FunctionTool

def database_query(query: str) -> list:
    """Execute a database query"""
    # Implementation...
    return [{"id": 1, "name": "Example"}]

# Define pre-processor to sanitize input
def sanitize_query(parameters):
    parameters["query"] = parameters["query"].strip().replace(";", "")
    return parameters

# Define post-processor to format output
def format_results(results):
    return {"count": len(results), "data": results}

# Create tool with processors
db_query_tool = FunctionTool(
    database_query,
    pre_processor=sanitize_query,
    post_processor=format_results
)
```

### Tool Documentation

The quality of your tool documentation directly impacts how effectively LLMs will use your tools:

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class SortOrder(Enum):
    ASCENDING = "asc"
    DESCENDING = "desc"

class SortItems(BaseModel):
    """
    Sort a list of items in ascending or descending order.
    
    This tool takes a list of comparable items (numbers or strings) and
    sorts them according to the specified order. It can handle numerical
    or alphabetical sorting.
    """
    
    items: List[str] = Field(
        ...,
        description="The items to sort. Can be numbers or strings. "
                   "Example: ['apple', 'banana', 'cherry'] or [10, 5, 8, 3]"
    )
    
    order: SortOrder = Field(
        SortOrder.ASCENDING,
        description="The sort order to use. 'asc' for ascending (smallest to largest), "
                   "'desc' for descending (largest to smallest)."
    )
    
    numeric: bool = Field(
        False,
        description="Whether to treat the items as numbers. If true, '10' will be sorted "
                   "after '2'. If false, standard string sorting applies."
    )
    
    def run(self):
        """
        Sort the items and return the sorted list.
        
        Returns:
            A list containing the sorted items.
        """
        if self.numeric:
            # Convert strings to numbers if numeric sorting is requested
            converted_items = [float(item) for item in self.items]
        else:
            converted_items = self.items
            
        # Determine if we should reverse the sort
        reverse = (self.order == SortOrder.DESCENDING)
        
        # Perform the sort
        sorted_items = sorted(converted_items, reverse=reverse)
        
        return sorted_items

sort_tool = FunctionTool(SortItems)
```

## Common Tool Categories

### Data Processing Tools

```python
def filter_data(data: list, condition: str) -> list:
    """
    Filter a list of items based on a condition.
    
    Args:
        data: List of dictionaries to filter
        condition: A Python expression like "item['age'] > 30"
        
    Returns:
        Filtered list of items
    """
    return [item for item in data if eval(condition, {"__builtins__": {}}, {"item": item})]

filter_tool = FunctionTool(filter_data)
```

### API Integration Tools

```python
class APIRequest(BaseModel):
    """
    Make an API request to an external service.
    """
    url: str = Field(..., description="The API endpoint URL")
    method: str = Field("GET", description="HTTP method: GET, POST, PUT, DELETE")
    headers: dict = Field({}, description="HTTP headers to include")
    body: Optional[dict] = Field(None, description="Request body for POST/PUT")
    
    def run(self):
        """Execute the API request."""
        import requests
        
        if self.method.upper() == "GET":
            response = requests.get(self.url, headers=self.headers)
        elif self.method.upper() == "POST":
            response = requests.post(self.url, headers=self.headers, json=self.body)
        # Add other methods as needed
        
        return {
            "status_code": response.status_code,
            "content": response.json() if response.headers.get("content-type") == "application/json" else response.text
        }

api_tool = FunctionTool(APIRequest)
```

### File Operation Tools

```python
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Confirmation message
    """
    with open(file_path, 'w') as file:
        file.write(content)
    return f"Successfully wrote to {file_path}"

read_tool = FunctionTool(read_file)
write_tool = FunctionTool(write_file)
```

## Best Practices

1. **Clear Documentation**: Provide detailed descriptions of what each tool does
2. **Proper Typing**: Use type hints to specify parameter types and return values
3. **Error Handling**: Implement robust error handling within tools
4. **Sensible Defaults**: Provide default values for optional parameters
5. **Validation**: Use Pydantic's validation capabilities to ensure correct inputs
6. **Focused Functionality**: Keep tools focused on specific tasks
7. **Consistent Naming**: Use clear, consistent naming conventions
8. **Security Considerations**: Validate inputs to prevent injection attacks
9. **Input Examples**: Include examples in parameter descriptions
10. **Output Documentation**: Describe the format and structure of tool outputs

## Next Steps

- [Learn about different agent types](agents.md)
- [Explore provider options](providers.md)
- [Understand message handling](messages.md)
- [See custom tool examples](../guides/custom-tools.md)
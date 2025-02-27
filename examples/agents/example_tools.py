import datetime
import os
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from ToolAgents import FunctionTool
from ToolAgents.function_tool import cli_confirmation_handler


# Simple tool for the agent, to get the current date and time in a specific format.
def get_current_datetime(output_format: str):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time
    """

    return datetime.datetime.now().strftime(output_format)


# Enum for the calculator tool.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# Simple pydantic calculator tool for the agent.
class calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """

    number_one: float = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: float = Field(..., description="Second number.")

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


# Example function based on an OpenAI example.
# llama-cpp-agent supports OpenAI like schemas for function definition.
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    if "London" in location:
        return f"Weather in {location}: {22}° {unit.value}"
    elif "New York" in location:
        return f"Weather in {location}: {24}° {unit.value}"
    elif "North Pole" in location:
        return f"Weather in {location}: {-42}° {unit.value}"
    else:
        return f"Weather in {location}: unknown"


# Here is a function definition in OpenAI style
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
                "unit": {
                    "type": "string",
                    "description": "The unit of measurement. Should be celsius or fahrenheit",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location", "unit"],
        },
    },
}


class ReadFileInput(BaseModel):
    """Input for reading a file."""

    filename: str = Field(..., description="The name of the file to read")


class WriteFileInput(BaseModel):
    """Input for writing to a file."""

    filename: str = Field(..., description="The name of the file to write to")
    content: str = Field(..., description="The content to write to the file")


class ListFilesInput(BaseModel):
    """Input for listing files in a directory."""

    directory: str = Field(
        ..., description="The directory to list files from (optional)"
    )


def read_file(input_data: ReadFileInput) -> str:
    """
    Read the contents of a file.
    Args:
        input_data: The input for the file read operation.
    Returns:
        The contents of the file.
    """
    full_path = input_data.filename
    try:
        with open(full_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File '{input_data.filename}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(input_data: WriteFileInput) -> str:
    """
    Write content to a file.
    Args:
        input_data: The input for the file write operation.
    Returns:
        A message indicating the result of the operation.
    """
    full_path = input_data.filename
    try:
        with open(full_path, "w") as file:
            file.write(input_data.content)
        return f"Successfully wrote to file '{input_data.filename}'."
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def list_files(input_data: ListFilesInput) -> List[str]:
    """
    List files in a directory.
    Args:
        input_data: The input for the list files operation.
    Returns:
        A list of filenames in the directory.
    """
    full_path = input_data.directory
    try:
        return os.listdir(full_path)
    except FileNotFoundError:
        return [f"Error: Directory '{input_data.directory}' not found."]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]


calculator_function_tool = FunctionTool(calculator)
current_datetime_function_tool = FunctionTool(get_current_datetime)

get_weather_function_tool = FunctionTool.from_openai_tool(
    open_ai_tool_spec, get_current_weather
)

read_file_tool = FunctionTool(read_file)
write_file_tool = FunctionTool(write_file)
list_files_tool = FunctionTool(list_files)

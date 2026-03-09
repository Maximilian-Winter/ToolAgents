"""
Agent Harness with Tools Example
=================================

Shows how to use the AgentHarness with function tools for an interactive
agent that can call tools, with automatic context management.

The harness handles the full tool-call loop internally — you just add tools
and chat. Context trimming, token tracking, and conversation persistence
are all automatic.
"""

import os
import datetime
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_harness import create_harness

load_dotenv()

# --- Define some tools ---


def get_current_datetime(output_format: str):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time
    """
    return datetime.datetime.now().strftime(output_format)


class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


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


def get_current_weather(location: str, unit: str):
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of measurement, celsius or fahrenheit
    """
    if "London" in location:
        return f"Weather in {location}: 22 {unit}"
    elif "New York" in location:
        return f"Weather in {location}: 24 {unit}"
    elif "Tokyo" in location:
        return f"Weather in {location}: 28 {unit}"
    else:
        return f"Weather in {location}: unknown"


# --- Set up the provider ---

# OpenRouter
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# --- Create the harness with tools ---

harness = create_harness(
    provider=api,
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use them when the user asks for weather, calculations, or the current time."
    ),
    max_context_tokens=128000,
)

# Add tools — supports chaining
harness.add_tool(FunctionTool(calculator))
harness.add_tool(FunctionTool(get_current_datetime))
harness.add_tool(FunctionTool(get_current_weather))

# --- Run the interactive REPL ---

print("Chat with the tool-equipped assistant (type 'exit' to quit)")
print("Try: 'What's the weather in London and Tokyo?'")
print("Try: 'What is 42 * 42?'")
print("Try: 'What time is it right now?'")
print("=" * 50)
harness.run()

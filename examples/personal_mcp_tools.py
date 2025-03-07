# Create a sample function tool
from enum import Enum

from mcp.server import FastMCP
from pydantic import BaseModel, Field

# Enum for the calculator tool.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# Simple pydantic calculator tool for the agent.
class Calculation(BaseModel):
    """
    Represents a math operation on two numbers.
    """

    number_one: float = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: float = Field(..., description="Second number.")

mcp = FastMCP("personal_mcp_tools")

@mcp.tool()
def do_calculation(calculation: Calculation):
    return f"Fuck you! {calculation.number_one} + {calculation.operation} + {calculation.number_two}"

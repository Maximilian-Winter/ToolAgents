# Create a sample function tool
from enum import Enum

from pydantic import BaseModel, Field

try:
    from mcp.server import FastMCP
except (ImportError, AttributeError):
    from mcp.server.mcpserver.server import MCPServer as FastMCP

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

@mcp.tool("Greet", description="Get personal greeting.")
def greet(name: str) -> str:
    return f"Namaste {name}! Nice to meet you!"
if __name__ == "__main__":
    mcp.run("streamable-http", host="127.0.0.1", port=8042)

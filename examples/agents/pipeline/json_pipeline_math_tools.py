from enum import Enum

from ToolAgents import FunctionTool


class MathOps(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


def math_operation(operation: MathOps, num1: float, num2: float) -> float:
    """
    Performs math operations on two numbers.

    Args:
        operation: Math operation to perform.
        num1: First number.
        num2: Second number.

    Returns:
        Result of the math operation.
    """
    if operation == MathOps.ADD:
        return num1 + num2
    if operation == MathOps.SUBTRACT:
        return num1 - num2
    if operation == MathOps.MULTIPLY:
        return num1 * num2
    if operation == MathOps.DIVIDE:
        return num1 / num2
    raise ValueError(f"Unsupported math operation: {operation}")


def create_tools():
    return [FunctionTool(math_operation)]

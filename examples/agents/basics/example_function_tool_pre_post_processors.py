from typing import Any

from ToolAgents import FunctionTool
from ToolAgents.function_tool import PreProcessor, PostProcessor


class ValidationPreProcessor(PreProcessor):
    """Example preprocessor that validates required fields."""

    def process(self, parameters: dict[str, Any]) -> dict[str, Any]:
        required_fields = ["name", "age"]  # Example required fields
        for field in required_fields:
            if field not in parameters:
                raise ValueError(f"Missing required field: {field}")
        return parameters


class LoggingPostProcessor(PostProcessor):
    """Example postprocessor that logs the result."""

    def process(self, result: Any) -> Any:
        print(f"Function returned: {result}")
        return result


def create_greeting(name: str, age: int) -> str:
    """
    Creates a personal greeting.
    Args:
        name (str): The name of the person to greet.
        age (int): The age of the person.
    Returns:
        A personal greeting(str)

    """
    return f"Hello {name}, you are {age} years old!"


# Create processors
validators = ValidationPreProcessor()
logger = LoggingPostProcessor()

# Create function tool with multiple processors
tool = FunctionTool(
    create_greeting,
    pre_processors=[validators],
    post_processors=[logger],
    debug_mode=True,
)

# Execute with processors
result = tool.execute({"name": "Alice", "age": 30})
print(f"Final result: {result}")

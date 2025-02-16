import json
import os
from enum import Enum

from dotenv import load_dotenv

from ToolAgents import FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.pipelines.pipeline import ProcessStep, Pipeline, SequentialProcess

from ToolAgents.provider.chat_api_provider.anthropic import AnthropicChatAPI, AnthropicSettings

load_dotenv()

api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api, debug_output=True)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.3

settings.set_max_new_tokens(4096)

api.set_default_settings(settings)

class MathOps(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


def math_operation(operation: MathOps, num1: float, num2: float) -> float:
    """
    Performs math operations on two numbers.

    Args:
        operation (MathOps): Math operation to perform
        num1 (float): first number
        num2 (float): second number
    Returns:
        float: result of math operation
    """
    if operation == MathOps.ADD:
        return num1 + num2
    elif operation == MathOps.SUBTRACT:
        return num1 - num2
    elif operation == MathOps.MULTIPLY:
        return num1 * num2
    elif operation == MathOps.DIVIDE:
        return num1 / num2


math_tool = FunctionTool(math_operation)

math_element = ProcessStep(
    step_name="math_result",
    system_message="You are a math assistant that performs mathematical operations.",
    prompt_template="Perform the following math operation: {operation} {num1} and {num2}",
    tools=[math_tool]
)

greeting_element = ProcessStep(
    step_name="greeting",
    system_message="You are a greeting assistant that generates personalized greetings.",
    prompt_template="Generate a personalized greeting for a person named {name} who just received the following math result: {math_result}"
)

chain = [math_element, greeting_element]

math_greeting = SequentialProcess(agent=agent)

math_greeting.add_steps(chain)

pipeline = Pipeline()

pipeline.add_process(math_greeting)
results = pipeline.run_pipeline(operation="multiply", num1=5, num2=3, name="Alex")
print(results["greeting"])



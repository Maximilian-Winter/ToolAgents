import datetime
import os

from ToolAgents import FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.function_tool import ConfirmationRequest, ToolRegistry
from ToolAgents.messages import ChatMessage
from ToolAgents.provider import MistralChatAPI

from dotenv import load_dotenv

load_dotenv()


# Simple tool for the agent, to get the current date and time in a specific format.
def get_current_datetime(output_format: str):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time
    """

    return datetime.datetime.now().strftime(output_format)


# Simple confirmation handler that will ask the user for confirmation before using a tool
def cli_confirmation_handler(request: ConfirmationRequest):
    """Simple CLI-based confirmation handler."""
    print(f"\n=== Confirmation Required for Tool Use: {request.function_name} ===")
    if request.description:
        print(f"Description: {request.description}")
    print("Parameters:")
    for key, value in request.parameters.items():
        print(f"{key}: {value}")

    response = input("\nApprove this execution? (y/n): ").strip().lower()

    if response == "y":
        request.approve()
    else:
        request.reject()


current_datetime_function_tool = FunctionTool(get_current_datetime)

current_datetime_function_tool.enable_confirmation("Get the current date and time")
current_datetime_function_tool.set_confirmation_handler(cli_confirmation_handler)


# Mistral API
api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0

tool_registry = ToolRegistry()

tool_registry.add_tool(current_datetime_function_tool)

messages = [
    ChatMessage.create_system_message(
        "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."
    ),
    ChatMessage.create_user_message(
        "Retrieve the date and time in the format: %Y-%m-%d %H:%M:%S."
    ),
]

chat_response = agent.get_response(
    messages=messages, settings=settings, tool_registry=tool_registry
)

print(chat_response.response)

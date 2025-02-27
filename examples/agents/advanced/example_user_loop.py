import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.provider import OpenAIChatAPI

from example_tools import (
    calculator_function_tool,
    current_datetime_function_tool,
    get_weather_function_tool,
)

from dotenv import load_dotenv

load_dotenv()

# Openrouter API
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-pro-exp-02-05:free",
    base_url="https://openrouter.ai/api/v1",
)

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0


chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."
)

while True:
    user_input = input("User input >")
    if user_input == "quit":
        break
    elif user_input == "save":
        chat_history.save_to_json("example_chat_history.json")
    elif user_input == "load":
        chat_history = ChatHistory.load_from_json("example_chat_history.json")
    else:
        chat_history.add_user_message(user_input)

        chat_response = agent.get_response(
            messages=chat_history.get_messages(),
            settings=settings
        )

        print(chat_response.response.strip())
        chat_history.add_messages(chat_response.messages)

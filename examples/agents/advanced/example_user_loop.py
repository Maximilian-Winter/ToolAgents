import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.provider import OpenAIChatAPI, GoogleGenAIChatAPI

from dotenv import load_dotenv

load_dotenv()

# Openrouter API
api = GoogleGenAIChatAPI(
    api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-lite-preview-02-05"
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
    "You are a helpful assistant that talks like an old pirate 1678"
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
            messages=chat_history.get_messages(), settings=settings
        )

        print(chat_response.response.strip())
        chat_history.add_messages(chat_response.messages)

import os

from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.provider import OpenAIChatAPI

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

        stream = agent.get_streaming_response(
            messages=chat_history.get_messages(),
            settings=settings,
        )
        chat_response = None
        for res in stream:
            print(res.chunk, end="", flush=True)
            if res.finished:
                chat_response = res.finished_response
        if chat_response is not None:
            chat_history.add_messages(chat_response.messages)
        else:
            raise Exception("Error during response generation")

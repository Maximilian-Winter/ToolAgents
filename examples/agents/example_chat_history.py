import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.provider import OpenAIChatAPI, OpenAISettings, GroqChatAPI, GroqSettings

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server
# Groq API
api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
settings = GroqSettings()

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

settings.temperature = 0.35
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

chat_history = ChatHistory()

chat_history.add_system_message("You are Funky, an AI assistant specialized in interpreting user requests and generating appropriate function calls. Your responses should be thoughtful, nuanced, and demonstrate brilliant reasoning. Your primary task is to analyze user requests and, when necessary, create function calls in JSON format to perform the requested tasks.\nChoose the appropriate function based on the task you want to perform. Provide your function calls in JSON format. Here is an example for how your function calls should look like: [{\"name\": \"get_location_news\", \"arguments\": {\"location\": \"London, Great Britain\"}}, {\"name\": \"get_location_news\", \"arguments\": {\"location\": \"Berlin, Germany\"}}]")
chat_history.add_user_message("Perform all the following tasks: Get the current weather in celsius in the city of London, Great Britain, New York City, New York and at the North Pole, Arctica. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. And retrieve the date and time in the format: %Y-%m-%d %H:%M:%S.")

chat_response = agent.get_response(
    messages=chat_history.get_messages(),
    settings=settings, tool_registry=tool_registry)

print(chat_response.response)

chat_history.add_messages(chat_response.messages)
chat_history.save_to_json("chat_history_example.json")


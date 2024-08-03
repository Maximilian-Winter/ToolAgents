import json
import os

from ToolAgents.agents import ChatAPIAgent
from ToolAgents.provider import OpenAIChatAPI, OpenAISettings, AnthropicChatAPI, AnthropicSettings
from ToolAgents.utilities import ChatHistory
from test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4-turbo")
# settings = OpenAISettings()

api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20240620")
settings = AnthropicSettings()

# Create the ChatAPIAgent
agent = ChatAPIAgent(chat_api=api, debug_output=False)

settings.temperature = 0.45
settings.top_p = 0.85

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Perform all the following tasks: Get the current weather in celsius in London, New York and at the North Pole. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. Retrieve the current date and time in the format: dd.mm.yyy hh:mm."}]

chat_history = ChatHistory()
chat_history.add_list_of_dicts(messages)
chat_history.save_history("./test_chat_history.json")
# Get a response
result = agent.get_response(
    messages=messages,
    tools=tools,
    settings=settings
)

print(result)
print(json.dumps(agent.last_messages_buffer, indent=2))
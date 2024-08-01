import os

from ToolAgents.agents import ChatAPIAgent
from ToolAgents.provider import OpenAIChatAPI, OpenAISettings, AnthropicChatAPI, AnthropicSettings
from test_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4-turbo")
# settings = OpenAISettings()

api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20240620")
settings = AnthropicSettings()

# Create the ChatAPIAgent
agent = ChatAPIAgent(chat_api=api, system_prompt="You are a helpful assistant.", debug_output=False)

settings.temperature = 0.45
settings.top_p = 0.85

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

# Get a response
result = agent.get_response(
    ("Perform all the following tasks: Get the current weather in celsius in London, New York and at the North Pole. "
     "Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6 and 96/8."),
    tools=tools,
    settings=settings
)

print(result)

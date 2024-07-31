import os

from FunkyFlow.agents.chat_api_agent import ChatAPIAgent
from FunkyFlow.utilities.chat_api_with_tools import OpenAIChatAPI, OpenAISettings, AnthropicChatAPI, AnthropicSettings
from FunkyFlow.utilities.testus_tool import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()


# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-3.5-turbo")
# settings = OpenAISettings()

api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20240620")
settings = AnthropicSettings()

# Create the ChatAPIAgent
agent = ChatAPIAgent(chat_api=api, debug_output=False)

settings.temperature = 0.3
settings.top_p = 1.0


# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

# Get a response
result = agent.get_response(
    "Perform all the following tasks: Get the current weather in celsius in London, New York and at the North Pole. "
    "Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6 and 96/8.",
    tools=tools,
    settings=settings
)

print(result)

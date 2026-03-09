import json
import os

from dotenv import load_dotenv

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.provider import AnthropicChatAPI, OpenAIChatAPI
from test_tools import (
    calculator_function_tool,
    current_datetime_function_tool,
    get_weather_function_tool,
)

load_dotenv()

# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4o")
api = AnthropicChatAPI(
    api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20240620"
)
settings = api.get_default_settings()

agent = ChatToolAgent(chat_api=api, log_output=True)
settings.temperature = 0.45
settings.top_p = 0.85

tool_registry = ToolRegistry()
tool_registry.add_tools(
    [
        calculator_function_tool,
        current_datetime_function_tool,
        get_weather_function_tool,
    ]
)

chat_history = ChatHistory()
with open("./test_tools_chat_history.json", "r", encoding="utf-8") as f:
    chat_history.add_messages_from_dictionaries(json.load(f))

result = agent.get_streaming_response(
    messages=chat_history.get_messages(), tool_registry=tool_registry, settings=settings
)

for res in result:
    print(res, end="", flush=True)
print()

chat_history.add_messages(agent.last_messages_buffer)
chat_history.save_to_json("./test_chat_history_after_chat_api.json")

import json

from ToolAgents.agents import Llama31Agent
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.provider import LlamaCppServerProvider

from test_tools import (
    calculator_function_tool,
    current_datetime_function_tool,
    get_weather_function_tool,
)

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = Llama31Agent(provider=provider, log_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.6
settings.set_max_new_tokens(4096)
settings.set_stop_tokens(["assistant"], None)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! How are you?"},
]

result = agent.get_response(messages=messages, sampling_settings=settings)
print(result)

tools = [
    calculator_function_tool,
    current_datetime_function_tool,
    get_weather_function_tool,
]

chat_history = ChatHistory()
with open("./test_tools_chat_history.json", "r", encoding="utf-8") as f:
    chat_history.add_messages_from_dictionaries(json.load(f))

result = agent.get_streaming_response(
    messages=chat_history.get_messages(), sampling_settings=settings, tools=tools
)
for tok in result:
    print(tok, end="", flush=True)
print()

chat_history.add_messages(agent.last_messages_buffer)
chat_history.save_to_json("./test_chat_history_after_llama.json")

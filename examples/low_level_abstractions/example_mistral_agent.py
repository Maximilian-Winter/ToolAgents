from ToolAgents import ToolRegistry
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider
from ToolAgents.utilities import ChatHistory

from example_tools import calculator_function_tool, \
    current_datetime_function_tool, get_weather_function_tool

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.85

settings.set_max_new_tokens(4096)

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}]

result = agent.get_response(messages=messages, settings=settings)
print(result)

tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

chat_history = ChatHistory()
chat_history.load_history("./test_tools_chat_history.json")

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings, tool_registry=tool_registry)
for tok in result:
    print(tok, end="", flush=True)
print()

chat_history.add_list_of_dicts(agent.last_messages_buffer)
chat_history.save_history("./test_chat_history_after_mistral.json")

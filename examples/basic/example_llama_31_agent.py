from ToolAgents.agents import LlamaAgent
from ToolAgents.provider import LlamaCppSamplingSettings, LlamaCppServerProvider
from ToolAgents.provider import VLLMServerSamplingSettings, \
    VLLMServerProvider
from ToolAgents.utilities import ChatHistory

from example_tools import calculator_function_tool, \
    current_datetime_function_tool, get_weather_function_tool

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = LlamaAgent(llm_provider=provider, debug_output=True)

settings = LlamaCppSamplingSettings()
settings.temperature = 10.0
settings.top_p = 1.0
settings.top_k = 0
settings.min_p = 0.0
settings.max_tokens = 4096

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}]

result = agent.get_response(messages=messages, sampling_settings=settings)
print(result)

tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

chat_history = ChatHistory()
chat_history.load_history("./test_tools_chat_history.json")

result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    sampling_settings=settings, tools=tools, add_tool_instructions_to_first_message=True)
for tok in result:
    print(tok, end="", flush=True)
print()

chat_history.add_list_of_dicts(agent.last_messages_buffer)
chat_history.save_history("./test_chat_history_after_llama.json")

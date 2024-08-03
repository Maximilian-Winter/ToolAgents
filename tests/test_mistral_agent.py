import json

from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppSamplingSettings, LlamaCppServerProvider
from ToolAgents.provider import VLLMServerSamplingSettings, \
    VLLMServerProvider

from test_tools import calculator_function_tool, \
    current_datetime_function_tool, get_weather_function_tool

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(llm_provider=provider, debug_output=True)

settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.top_k = 0
settings.min_p = 0.0
settings.max_tokens = 4096

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}]

result = agent.get_response(messages=messages, sampling_settings=settings)
print(result)

tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Perform all the following tasks: Get the current weather in celsius in London, New York and at the North Pole. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. Retrieve the current date and time in the format: dd.mm.yyy hh:mm."}]

result = agent.get_streaming_response(
    messages=messages,
    sampling_settings=settings, tools=tools)
for tok in result:
    print(tok, end="", flush=True)
print()
print(json.dumps(agent.last_messages_buffer, indent=2))

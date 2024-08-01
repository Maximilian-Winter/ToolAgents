from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppSamplingSettings, LlamaCppServerProvider
from ToolAgents.provider import VLLMServerSamplingSettings, \
    VLLMServerProvider

from test_tools import calculator_function_tool, \
    current_datetime_function_tool, get_weather_function_tool

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(llm_provider=provider, debug_output=False,
                     system_prompt="You are a helpful assistant."
                     )

settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.top_k = 0
settings.min_p = 0.0
settings.max_tokens = 4096


result = agent.get_response("Hello! How are you?", sampling_settings=settings)
print(result)

tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

result = agent.get_streaming_response("Perform all the following tasks: Get the current weather in celsius in London, New York and at the North Pole. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8.", sampling_settings=settings, tools=tools)
for tok in result:
    print(tok, end="", flush=True)
print()

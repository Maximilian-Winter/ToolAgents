from ToolAgents.agents import LlamaAgent
from ToolAgents.provider import LlamaCppSamplingSettings, LlamaCppServerProvider
from ToolAgents.provider import VLLMServerSamplingSettings, \
    VLLMServerProvider
from ToolAgents.utilities import ChatHistory
from example_tools import get_weather_function_tool, calculator_function_tool, current_datetime_function_tool
from code_executer import PythonCodeExecutor, system_message_code_agent

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = LlamaAgent(llm_provider=provider, debug_output=True)

settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.top_k = 0
settings.min_p = 0.0
settings.max_tokens = 4096
settings.stop = ["</s>", "<|eom_id|>", "<|eot_id|>", "assistant", "<|start_header_id|>assistant<|end_header_id|>"]

chat_history = ChatHistory()
chat_history.add_system_message(system_message_code_agent+ f"""\n\n## Predefined Functions

You have access to the following predefined functions in Python:

```python
{get_weather_function_tool.get_python_documentation()}

{calculator_function_tool.get_python_documentation()}

{current_datetime_function_tool.get_python_documentation()}
```

You can call these predefined functions in Python like this:

```python_interpreter
example_function(example_parameter=10.0)
```
""")

python_code_executor = PythonCodeExecutor([get_weather_function_tool, calculator_function_tool, current_datetime_function_tool])


def run_code_agent(user):
    chat_history.add_user_message(user)
    result_gen = agent.get_streaming_response(
        messages=chat_history.to_list(),
        sampling_settings=settings, tools=None)

    full_response = ""
    for tok in result_gen:
        print(tok, end="", flush=True)
        full_response += tok
    print()
    while True:
        chat_history.add_assistant_message(full_response)
        if "```python_interpreter" in full_response:
            chat_history.add_tool_message(python_code_executor.run(full_response))
            result_gen = agent.get_streaming_response(
                messages=chat_history.to_list(),
                sampling_settings=settings, tools=None)
            full_response = ""
            for tok in result_gen:
                print(tok, end="", flush=True)
                full_response += tok
            print()
        else:
            break


prom = "Get the current weather in New York in Celsius."
run_code_agent(prom)

prom = "Get the current weather in London in Celsius."
run_code_agent(prom)

prom = "Get the current weather on the North Pole in Celsius."
run_code_agent(prom)

prom2 = "Get the current date and time in the format: dd-mm-yyyy hh:mm"
prom3 = "Calculate 5215 * 6987"
prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""


run_code_agent(prom2)
run_code_agent(prom3)
run_code_agent(prompt)
run_code_agent(prompt2)
run_code_agent(prompt3)

chat_history.save_history("./test_chat_history_after_llama.json")

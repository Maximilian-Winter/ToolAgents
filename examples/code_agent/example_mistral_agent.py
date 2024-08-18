from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider
from ToolAgents.utilities import ChatHistory
from example_tools import get_weather_function_tool, calculator_function_tool, current_datetime_function_tool
from code_executer import PythonCodeExecutor, system_message_code_agent, run_code_agent

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.1
settings.set_stop_tokens(["</s>"], None)
settings.set_max_new_tokens(4096)


python_code_executor = PythonCodeExecutor(
    [get_weather_function_tool, calculator_function_tool, current_datetime_function_tool])

chat_history = ChatHistory()
chat_history.add_system_message(python_code_executor.get_python_interpreter_system_message())

prompt_function_calling = "Get the current weather in New York in Celsius. Get the current weather in London in Celsius. Get the current weather on the North Pole in Celsius. Calculate 5215 * 6987. Get the current date and time in the format: dd-mm-yyyy hh:mm"

prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""

run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt_function_calling,
               python_code_executor=python_code_executor)

run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt,
               python_code_executor=python_code_executor)
run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt2,
               python_code_executor=python_code_executor)
run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt3,
               python_code_executor=python_code_executor)

chat_history.save_history("./test_chat_history_after_mistral.json")

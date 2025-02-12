import os

from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings

from ToolAgents.messages import ChatHistory
from example_tools import get_weather_function_tool, calculator_function_tool, current_datetime_function_tool, unit, \
    MathOperation
from new_code_executor import PythonExecutor, run_llm_code_agent

from dotenv import load_dotenv

load_dotenv()

# Anthropic API
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
settings = AnthropicSettings()
agent = ChatToolAgent(chat_api=api, debug_output=True)

settings.neutralize_all_samplers()
settings.temperature = 0.1
settings.set_max_new_tokens(4096)

python_code_executor = PythonExecutor(predefined_types=[unit, MathOperation],
                                          predefined_functions=[get_weather_function_tool, calculator_function_tool,
                                                                current_datetime_function_tool])

chat_history = ChatHistory()
chat_history.add_system_message(python_code_executor.get_system_message())

prompt_function_calling = "Get the current weather in New York in Celsius. Get the current weather in London in Celsius. Get the current weather on the North Pole in Celsius. Calculate 5215 * 6987. Get the current date and time in the format: dd-mm-yyyy hh:mm"

prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""

run_llm_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt_function_calling,
               executor=python_code_executor)

run_llm_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt,
               executor=python_code_executor)
run_llm_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt2,
               executor=python_code_executor)
run_llm_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt3,
               executor=python_code_executor)

chat_history.save_to_json("./test_chat_history_after_llama.json")

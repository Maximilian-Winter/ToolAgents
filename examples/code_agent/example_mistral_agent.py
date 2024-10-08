from ToolAgents.agents import MistralAgent
from ToolAgents.agents.hosted_tool_agents import TemplateAgent
from ToolAgents.provider import LlamaCppServerProvider
from ToolAgents.utilities import ChatHistory
from ToolAgents.utilities.chat_history import AdvancedChatFormatter
from example_tools import get_weather_function_tool, calculator_function_tool, current_datetime_function_tool, unit, \
    MathOperation
from code_executer import PythonCodeExecutor, run_code_agent

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

system_template = "<system_instructions>\n{content}\n</system_instructions>\n\n"
assistant_template = "{content}</s>"
user_template = "[INST] {content} [/INST]"

advanced_chat_formatter = AdvancedChatFormatter({
    "system": system_template,
    "user": user_template,
    "assistant": assistant_template,
}, include_system_message_in_first_user_message=True)

# agent = MistralAgent(provider=provider, debug_output=True)
agent = TemplateAgent(provider, advanced_chat_formatter=advanced_chat_formatter, generation_prompt=None, debug_output=False)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.1
settings.set_stop_tokens(["</s>"], None)
settings.set_max_new_tokens(4096)

python_code_executor = PythonCodeExecutor(predefined_types=[unit, MathOperation],
                                          predefined_functions=[get_weather_function_tool, calculator_function_tool,
                                                                current_datetime_function_tool])

chat_history = ChatHistory()
chat_history.add_system_message(python_code_executor.get_python_interpreter_system_message())

prompt_function_calling = "Get the current weather in New York in Celsius. Get the current weather in London in Celsius. Get the current weather on the North Pole in Celsius. Calculate 5215 * 6987. Get the current date and time in the format: dd-mm-yyyy hh:mm"

prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""

run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt_function_calling,
               python_code_executor=python_code_executor)

# run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt,
#                python_code_executor=python_code_executor)
# run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt2,
#                python_code_executor=python_code_executor)
# run_code_agent(agent=agent, settings=settings, chat_history=chat_history, user_input=prompt3,
#                python_code_executor=python_code_executor)

chat_history.save_history("./test_chat_history_after_mistral.json")

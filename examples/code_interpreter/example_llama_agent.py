from ToolAgents import ToolRegistry
from ToolAgents.agents import Llama31Agent
from ToolAgents.provider import LlamaCppSamplingSettings, LlamaCppServerProvider
from ToolAgents.provider import VLLMServerSamplingSettings, \
    VLLMServerProvider
from ToolAgents.utilities import ChatHistory
from code_interpreter import CodeInterpreter

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = Llama31Agent(provider=provider, debug_output=True)

settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.top_k = 0
settings.min_p = 0.0
settings.max_tokens = 4096
settings.stop = ["</s>", "<|eom_id|>", "<|eot_id|>", "assistant", "<|start_header_id|>assistant<|end_header_id|>"]
code_interpreter = CodeInterpreter(venv_path="./.venv")

tools = code_interpreter.get_tools()

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant. You can call functions to execute Python code and CLI commands.")

prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""


chat_history.add_user_message(prompt)
result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings, tool_registry=ToolRegistry(tools))
for tok in result:
    print(tok, end="", flush=True)
print()
chat_history.add_list_of_dicts(agent.last_messages_buffer)

chat_history.add_user_message(prompt2)
result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings, tool_registry=ToolRegistry(tools))
for tok in result:
    print(tok, end="", flush=True)
print()
chat_history.add_list_of_dicts(agent.last_messages_buffer)

chat_history.add_user_message(prompt3)
result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings, tool_registry=ToolRegistry(tools))
for tok in result:
    print(tok, end="", flush=True)
print()
chat_history.add_list_of_dicts(agent.last_messages_buffer)

chat_history.save_history("./test_chat_history_after_llama.json")

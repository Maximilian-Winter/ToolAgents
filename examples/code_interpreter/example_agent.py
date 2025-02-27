import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import AnthropicChatAPI

from ToolAgents.messages import ChatHistory

from dotenv import load_dotenv

from code_interpreter import CodeInterpreter

load_dotenv()

# Anthropic API
api = AnthropicChatAPI(
    api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022"
)
settings = AnthropicSettings()

agent = ChatToolAgent(chat_api=api, debug_output=True)

settings.temperature = 0.3
settings.top_p = 1.0
settings.max_tokens = 4096

code_interpreter = CodeInterpreter(venv_path="./.venv")

tools = code_interpreter.get_tools()

chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful assistant. You can call functions to execute Python code and CLI commands."
)

prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""

tool_registry = ToolRegistry()
tool_registry.add_tools(tools)

chat_history.add_user_message(prompt)
result = agent.get_streaming_response(
    messages=chat_history.get_messages(), settings=settings, tool_registry=tool_registry
)
for tok in result:
    print(tok.chunk, end="", flush=True)
print()
chat_history.add_messages(agent.last_messages_buffer)

chat_history.add_user_message(prompt2)
result = agent.get_streaming_response(
    messages=chat_history.get_messages(), settings=settings, tool_registry=tool_registry
)
for tok in result:
    print(tok.chunk, end="", flush=True)
print()
chat_history.add_messages(agent.last_messages_buffer)

chat_history.add_user_message(prompt3)
result = agent.get_streaming_response(
    messages=chat_history.get_messages(), settings=settings, tool_registry=tool_registry
)
for tok in result:
    print(tok.chunk, end="", flush=True)
print()
chat_history.add_messages(agent.last_messages_buffer)

chat_history.save_to_json("./test_chat_history_after.json")

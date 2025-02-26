import json
import os
from ToolAgents import ToolRegistry
from ToolAgents.agent_tools.file_tools import FilesystemTools
from ToolAgents.agent_tools.git_tools import GitTools
from ToolAgents.agent_tools.github_tools import GitHubTools
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import AnthropicChatAPI, OpenAIChatAPI, GroqChatAPI, MistralChatAPI, CompletionProvider


from dotenv import load_dotenv


load_dotenv()

# Mistral API
api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-large-latest")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0

# Initialize the FilesystemTools with working directory, in this case a windows path
file_tools = FilesystemTools("H:\\MaxDev42\\ToolAgentsUseCases")

# Initialize the GitTools with a function to get the working directory of the FilesystemTools
git_tools = GitTools(file_tools.get_working_directory)

# Initialize the GitHubTools with the initial owner and repo.
git_hub_tools = GitHubTools("Maximilian-Winter", "dolphin-bot")

# Define the tools
tools = file_tools.get_tools()
tools.extend(git_tools.get_tools())
tools.extend(git_hub_tools.get_tools())

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)


chat_history = ChatHistory()
chat_history.add_message(ChatMessage.create_system_message(f"You are an expert coding AI agent. You have access to the following tools to work with the filesystem, git and GitHub:\n\n{tool_registry.get_tools_documentation()}"))


while True:
    user_input = input("User >")

    if user_input == "exit":
        break

    chat_history.add_message(ChatMessage.create_user_message(user_input))
    chat_response = None
    result = agent.get_streaming_response(
        messages=chat_history.get_messages(),
        settings=settings, tool_registry=tool_registry)
    for res in result:
        if res.get_tool_results():
            print()
            print(f"Tool Use: {res.get_tool_name()}")
            print(f"Tool Arguments: {json.dumps(res.get_tool_arguments())}")
            print(f"Tool Result: {res.get_tool_results()}")
            print()
        print(res.chunk, end='', flush=True)
        if res.finished:
            chat_response = res.finished_response
            print()
    chat_history.add_messages(chat_response.messages)
    chat_history.save_to_json("coding_agent_history.json")
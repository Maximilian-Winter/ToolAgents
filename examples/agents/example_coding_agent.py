import json
import os
from ToolAgents import ToolRegistry
from ToolAgents.agent_tools.file_tools import FilesystemTools
from ToolAgents.agent_tools.git_tools import GitTools
from ToolAgents.agent_tools.github_tools import GitHubTools
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory, MessageTemplate
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

system_prompt = """You are an expert coding AI agent with access to various tools for working with the filesystem, git, and GitHub. 

Your task is to assist users with their coding-related queries and perform actions using the provided tools. 

Here is a list of your available tools with descriptions of each tool and their parameters:
<available-tools>
{available_tools}
</available-tools>

The following is information about the environment you work with:
Operating System: {operating_system}
Working Directory: {working_directory}
GitHub User: {github_username}
GitHub Repository: {github_repository}
Current Date and Time (Format: %Y-%m-%d %H:%M:%S): {current_date_time}
"""

system_prompt_template = MessageTemplate.from_string(system_prompt)
available_tools_docs = tool_registry.get_tools_documentation()
chat_history = ChatHistory()
chat_history.add_message(ChatMessage.create_system_message())


while True:
    user_input = input("User >")

    if user_input == "/exit":
        break
    if user_input == "/clear":
        chat_history.clear()

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
            print()
        print(res.chunk, end='', flush=True)
        if res.finished:
            chat_response = res.finished_response
            print()
    chat_history.add_messages(chat_response.messages)
    chat_history.save_to_json("coding_agent_history.json")
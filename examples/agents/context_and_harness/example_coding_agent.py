"""
Coding Agent with Harness & Context Manager
=============================================

A full-featured coding agent that uses the AgentHarness for automatic
context management. This is the harness-based equivalent of
examples/agents/misc/example_coding_agent.py.

Features:
- Filesystem tools (read/write files, glob, search, etc.)
- Git tools (status, diff, commit, branch, etc.)
- GitHub tools (issues, PRs, etc.)
- Streaming output with tool call display
- Automatic context window management — long coding sessions
  won't blow up the context window
- Token usage tracking after every turn
- Conversation history saved to JSON

The harness handles all the plumbing: message accumulation, context
trimming, tool-call loops, and the interactive REPL. You just configure
and run.
"""

import datetime
import json
import os
import platform

from dotenv import load_dotenv

from ToolAgents import FunctionTool
from ToolAgents.agent_tools.file_tools import FilesystemTools
from ToolAgents.agent_tools.git_tools import GitTools
from ToolAgents.agent_tools.github_tools import GitHubTools
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.data_models.responses import ChatResponseChunk
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.message_template import MessageTemplate
from ToolAgents.agent_harness import create_harness, HarnessEvent
from ToolAgents.context_manager import ContextEvent

load_dotenv()

# ============================================================
# Configuration — change these to match your setup
# ============================================================

# Working directory for filesystem tools
WORKING_DIRECTORY = os.getcwd()

# GitHub repo (owner/repo) — set to your repo or leave as example
GITHUB_OWNER = "your-username"
GITHUB_REPO = "your-repo"

# ============================================================
# Provider — uncomment your preferred provider
# ============================================================

# OpenRouter
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# OpenAI
# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# Local server (vllm, llama-cpp-server, etc.)
# api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="my-model")

# ============================================================
# Tools — filesystem, git, github
# ============================================================

file_tools = FilesystemTools(WORKING_DIRECTORY)
git_tools = GitTools(file_tools.get_working_directory)
github_tools = GitHubTools(GITHUB_OWNER, GITHUB_REPO)

all_tools = []
all_tools.extend(file_tools.get_tools())
all_tools.extend(git_tools.get_tools())
all_tools.extend(github_tools.get_tools())

# ============================================================
# System prompt with dynamic context
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are an expert coding AI agent with access to various tools for working with the filesystem, git, and GitHub.

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

Guidelines:
- Always read files before modifying them to understand the existing code.
- Use git status and git diff to understand the current state before committing.
- When writing code, follow the existing style and conventions of the project.
- Explain what you're doing before and after performing actions.
- If a tool call fails, analyze the error and try a different approach.
"""

# Build a ToolRegistry just to get documentation (harness has its own internally)
from ToolAgents import ToolRegistry

temp_registry = ToolRegistry()
temp_registry.add_tools(all_tools)
tools_documentation = temp_registry.get_tools_documentation()

system_prompt_template = MessageTemplate.from_string(SYSTEM_PROMPT_TEMPLATE)


def build_system_prompt():
    """Rebuild system prompt with current date/time."""
    return system_prompt_template.generate_message_content(
        available_tools=tools_documentation,
        operating_system=platform.system(),
        working_directory=file_tools.get_working_directory(),
        github_username=github_tools.owner,
        github_repository=github_tools.repo,
        current_date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ============================================================
# Custom I/O handler for coding agent
# ============================================================

class CodingAgentIOHandler:
    """Custom I/O handler that displays tool calls during streaming."""

    def __init__(self):
        self.exit_commands = ("exit", "quit", "/exit", "/quit")

    def get_input(self, prompt: str = "> ") -> str | None:
        try:
            user_input = input("\nYou > ")
            if user_input.strip().lower() in self.exit_commands:
                return None
            if user_input.strip() == "/clear":
                print("[Conversation cleared]")
                return None  # Will be handled by the harness reset
            return user_input
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def on_text(self, text: str) -> None:
        print(f"\nAgent: {text}")

    def on_chunk(self, chunk: ChatResponseChunk) -> None:
        # Show tool calls
        if chunk.has_tool_call and chunk.tool_call:
            tool_name = chunk.tool_call.get("tool_call_name", "")
            tool_args = chunk.tool_call.get("tool_call_arguments")
            if tool_name and tool_args is not None:
                print(f"\n  [Tool: {tool_name}]")
                if isinstance(tool_args, dict):
                    for k, v in tool_args.items():
                        val_str = str(v)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        print(f"    {k}: {val_str}")

        # Show tool results
        if chunk.has_tool_call_result and chunk.tool_call_result:
            result = chunk.tool_call_result.get("tool_call_result", "")
            if len(result) > 500:
                result = result[:500] + f"... ({len(result)} chars total)"
            print(f"  [Result: {result}]")

        # Stream text chunks
        if chunk.chunk:
            print(chunk.chunk, end="", flush=True)

        if chunk.finished:
            print()

    def on_error(self, error: Exception) -> None:
        print(f"\n[ERROR] {error}")


# ============================================================
# Create and configure the harness
# ============================================================

settings = api.get_default_settings()
settings.temperature = 0.45
settings.top_p = 1.0

harness = create_harness(
    provider=api,
    system_prompt=build_system_prompt(),
    max_context_tokens=128000,
    reserve_tokens=4096,
    strategy="sliding_window",
    streaming=True,
    settings=settings,
    tools=all_tools,
)


# ============================================================
# Event handlers
# ============================================================

def on_turn_start(event_data):
    # Update system prompt with fresh timestamp each turn
    harness.set_system_prompt(build_system_prompt())


def on_turn_end(event_data):
    state = harness.context_state
    print(f"  [tokens: context={state.current_context_tokens}, "
          f"total={state.total_tokens_used}, "
          f"turns={harness.turn_count}]")


def on_trimmed(event_data):
    count = len(event_data.trimmed_messages) if event_data.trimmed_messages else 0
    print(f"\n  [Context Manager: trimmed {count} old messages to fit context window]")


harness.events.on(HarnessEvent.TURN_START, on_turn_start)
harness.events.on(HarnessEvent.TURN_END, on_turn_end)
harness.context_manager.events.on(ContextEvent.MESSAGES_TRIMMED, on_trimmed)


# ============================================================
# Run
# ============================================================

print("=" * 60)
print("  Coding Agent (with Harness & Context Manager)")
print("=" * 60)
print(f"  OS: {platform.system()}")
print(f"  Working Dir: {file_tools.get_working_directory()}")
print(f"  GitHub: {github_tools.owner}/{github_tools.repo}")
print(f"  Tools: {len(all_tools)} available")
print(f"  Context: {harness.context_manager.config.max_context_tokens} tokens max")
print()
print("  Commands: type normally to chat, '/exit' to quit")
print("=" * 60)

harness.run(io_handler=CodingAgentIOHandler())

# Save conversation history
if harness.turn_count > 0:
    from ToolAgents.data_models.chat_history import ChatHistory

    history = ChatHistory()
    # Add system message
    history.add_message(ChatMessage.create_system_message(build_system_prompt()))
    # Add all conversation messages
    history.add_messages(harness.messages)
    history.save_to_json("coding_agent_history.json")
    print(f"\nConversation saved to coding_agent_history.json")
    print(f"Total turns: {harness.turn_count}")
    print(f"Total tokens used: {harness.context_state.total_tokens_used}")

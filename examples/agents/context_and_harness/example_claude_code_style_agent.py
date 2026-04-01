"""
Claude Code-Style Coding Agent
================================

A coding agent using the new CodingTools — clean, focused tools inspired
by Claude Code's design: Bash, Read, Write, Edit, Glob, Grep, and SubAgent.

Features:
- Minimal, orthogonal tool set (each tool does ONE thing well)
- Sub-agent support for parallel/focused task delegation
- Automatic context window management via AgentHarness
- Streaming output with tool call display

Usage:
    python example_claude_code_style_agent.py
"""

import datetime
import os
import platform

from dotenv import load_dotenv

from ToolAgents import FunctionTool
from ToolAgents.agent_tools.coding_tools import CodingTools
from ToolAgents.data_models.responses import ChatResponseChunk
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.message_template import MessageTemplate
from ToolAgents.agent_harness import create_harness, HarnessEvent
from ToolAgents.context_manager import ContextEvent

load_dotenv()

# ============================================================
# Configuration
# ============================================================

WORKING_DIRECTORY = os.getcwd()

# ============================================================
# Provider — configure your LLM
# ============================================================

# OpenRouter
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="xiaomi/mimo-v2-pro",
    base_url="https://openrouter.ai/api/v1",
)

# Sub-agent provider (can be a different/cheaper model)
sub_agent_api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="xiaomi/mimo-v2-pro",
    base_url="https://openrouter.ai/api/v1",
)

# ============================================================
# Tools
# ============================================================

coding_tools = CodingTools(working_directory=WORKING_DIRECTORY)

# Get the base tools (bash, read, write, edit, glob, grep, list_directory, diff_files)
all_tools = coding_tools.get_tools()

# Create a single-turn sub-agent tool — it gets its own copy of coding tools
sub_agent_coding_tools = CodingTools(working_directory=WORKING_DIRECTORY)
sub_agent_tool = CodingTools.create_sub_agent_tool(
    sub_agent_api=sub_agent_api,
    sub_agent_name="sub_agent",
    sub_agent_tools=sub_agent_coding_tools.get_tools(),
)
all_tools.append(sub_agent_tool)

# Create multi-turn sub-agent tools (send, list, close)
multi_turn_coding_tools = CodingTools(working_directory=WORKING_DIRECTORY)
multi_turn_tools = CodingTools.create_multi_turn_sub_agent_tools(
    sub_agent_api=sub_agent_api,
    sub_agent_tools=multi_turn_coding_tools.get_tools(),
)
all_tools.extend(multi_turn_tools)

# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are an expert coding AI agent with access to focused, powerful tools.

Your tools:
- **bash**: Execute shell commands (builds, tests, git, etc.)
- **read_file**: Read files with line numbers
- **write_file**: Create or overwrite files
- **edit_file**: Precise string replacement in files (shows diff on success)
- **glob_files**: Find files by pattern
- **grep_files**: Search file contents with regex
- **list_directory**: List directory contents with depth control
- **diff_files**: Compare two files with unified diff
- **sub_agent**: Delegate complex sub-tasks to a separate agent (single-turn)
- **multi_turn_sub_agent**: Send messages to persistent sub-agent sessions
- **multi_turn_sub_agent_list_sessions**: List active sub-agent sessions
- **multi_turn_sub_agent_close_session**: Close a sub-agent session

Environment:
- OS: {operating_system}
- Working Directory: {working_directory}
- Date: {current_date_time}

Guidelines:
- Read files before editing to understand existing code
- Use edit_file for surgical changes, write_file for new files or complete rewrites
- Use bash for git operations, running tests, and builds
- Use sub_agent for independent research or tasks that benefit from focused context
- Use multi_turn_sub_agent for ongoing conversations with sub-agents
- Use list_directory to explore project structure
- Use diff_files to compare file versions
- Be concise and direct in your responses
"""

system_template = MessageTemplate.from_string(SYSTEM_PROMPT_TEMPLATE)


def build_system_prompt():
    return system_template.generate_message_content(
        operating_system=platform.system(),
        working_directory=coding_tools.working_directory,
        current_date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ============================================================
# I/O Handler
# ============================================================

class AgentIOHandler:
    def __init__(self):
        self.exit_commands = ("exit", "quit", "/exit", "/quit")

    def get_input(self, prompt: str = "> ") -> str | None:
        try:
            user_input = input("\nYou > ")
            if user_input.strip().lower() in self.exit_commands:
                return None
            return user_input
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def on_text(self, text: str) -> None:
        print(f"\nAgent: {text}")

    def on_chunk(self, chunk: ChatResponseChunk) -> None:
        if chunk.has_tool_call and chunk.tool_call:
            tool_name = chunk.tool_call.get("tool_call_name", "")
            tool_args = chunk.tool_call.get("tool_call_arguments")
            if tool_name and tool_args is not None:
                print(f"\n  [{tool_name}]")
                if isinstance(tool_args, dict):
                    for k, v in tool_args.items():
                        val_str = str(v)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        print(f"    {k}: {val_str}")

        if chunk.has_tool_call_result and chunk.tool_call_result:
            result = chunk.tool_call_result.get("tool_call_result", "")
            if len(result) > 500:
                result = result[:500] + f"... ({len(result)} chars)"
            print(f"  => {result}")

        if chunk.chunk:
            print(chunk.chunk, end="", flush=True)

        if chunk.finished:
            print()

    def on_error(self, error: Exception) -> None:
        print(f"\n[ERROR] {error}")


# ============================================================
# Create harness
# ============================================================

settings = api.get_default_settings()
settings.temperature = 0.3

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
    harness.set_system_prompt(build_system_prompt())


def on_turn_end(event_data):
    state = harness.context_state
    print(f"  [tokens: {state.current_context_tokens} context, "
          f"{state.total_tokens_used} total | "
          f"turn {harness.turn_count}]")


def on_trimmed(event_data):
    count = len(event_data.trimmed_messages) if event_data.trimmed_messages else 0
    print(f"\n  [Context trimmed: {count} old messages removed]")


harness.events.on(HarnessEvent.TURN_START, on_turn_start)
harness.events.on(HarnessEvent.TURN_END, on_turn_end)
harness.context_manager.events.on(ContextEvent.MESSAGES_TRIMMED, on_trimmed)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Claude Code-Style Coding Agent")
    print("=" * 60)
    print(f"  OS: {platform.system()}")
    print(f"  Working Dir: {coding_tools.working_directory}")
    tool_names = "bash, read, write, edit, glob, grep, list_dir, diff, sub_agent, multi_turn"
    print(f"  Tools: {len(all_tools)} ({tool_names})")
    print(f"  Context: {harness.context_manager.config.max_context_tokens} tokens max")
    print()
    print("  Type normally to chat, '/exit' to quit")
    print("=" * 60)

    harness.run(io_handler=AgentIOHandler())

    if harness.turn_count > 0:
        print(f"\nSession: {harness.turn_count} turns, "
              f"{harness.context_state.total_tokens_used} tokens used")

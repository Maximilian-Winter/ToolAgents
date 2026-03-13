"""
TUI Coding Agent
==================

A terminal user interface for a coding agent built with ToolAgents.tui widgets.
Uses AsyncCodingTools + AsyncAgentHarness + Textual.

Features:
- Scrollable chat history with Markdown rendering
- Side panel showing tool call activity
- Streaming assistant responses
- Status bar with token tracking
- Toggle tool panel with Ctrl+T

Usage:
    pip install textual
    python example_tui_coding_agent.py
"""

import datetime
import os
import platform

from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Header, Footer

from ToolAgents.agent_tools.coding_tools import CodingTools
from ToolAgents.agent_harness import create_async_harness, HarnessEvent
from ToolAgents.context_manager import ContextEvent
from ToolAgents.provider.chat_api_provider.open_ai import AsyncOpenAIChatAPI
from ToolAgents.utilities.message_template import MessageTemplate

from ToolAgents.tui import (
    AgentInput,
    ChatView,
    HarnessBridge,
    StatusBar,
    ToolPanel,
)

load_dotenv()


# ============================================================
# Configuration
# ============================================================

WORKING_DIRECTORY = os.getcwd()

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

Environment:
- OS: {operating_system}
- Working Directory: {working_directory}
- Date: {current_date_time}

Guidelines:
- Read files before editing to understand existing code
- Use edit_file for surgical changes, write_file for new files
- Use bash for git operations, running tests, and builds
- Be concise and direct in your responses
"""

system_template = MessageTemplate.from_string(SYSTEM_PROMPT_TEMPLATE)


def build_system_prompt():
    return system_template.generate_message_content(
        operating_system=platform.system(),
        working_directory=WORKING_DIRECTORY,
        current_date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ============================================================
# TUI Application
# ============================================================

class CodingAgentApp(App):
    """A TUI coding agent built from ToolAgents.tui composable widgets."""

    TITLE = "ToolAgents Coding Agent"

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-area {
        height: 1fr;
    }
    #chat {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+t", "toggle_tools", "Toggle Tools"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            yield ChatView(id="chat")
            yield ToolPanel(id="tools")
        yield StatusBar(id="status")
        yield AgentInput(id="input")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the harness and bridge on mount."""
        # Provider
        api = AsyncOpenAIChatAPI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
        )

        # Tools
        coding_tools = CodingTools(working_directory=WORKING_DIRECTORY)
        # Use sync FunctionTools — the async harness handles execution
        all_tools = coding_tools.get_tools()

        # Settings
        settings = api.get_default_settings()
        settings.temperature = 0.3

        # Harness
        self.harness = create_async_harness(
            provider=api,
            system_prompt=build_system_prompt(),
            max_context_tokens=128000,
            reserve_tokens=4096,
            strategy="sliding_window",
            streaming=True,
            settings=settings,
            tools=all_tools,
        )

        # Wire up harness events
        self.harness.events.on(HarnessEvent.TURN_START, self._on_turn_start)
        self.harness.context_manager.events.on(
            ContextEvent.MESSAGES_TRIMMED, self._on_trimmed
        )

        # Bridge
        self.bridge = HarnessBridge(
            app=self,
            harness=self.harness,
            chat_view=self.query_one("#chat", ChatView),
            tool_panel=self.query_one("#tools", ToolPanel),
            agent_input=self.query_one("#input", AgentInput),
            status_bar=self.query_one("#status", StatusBar),
        )

        # Welcome message
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(
            f"ToolAgents Coding Agent — {platform.system()} — {WORKING_DIRECTORY}"
        )

        # Focus input
        self.query_one("#input", AgentInput).focus()

    async def on_agent_input_submitted(self, event: AgentInput.Submitted) -> None:
        """Handle user message submission."""
        await self.bridge.send_message(event.text)

    def _on_turn_start(self, event_data) -> None:
        """Refresh system prompt each turn."""
        self.harness.set_system_prompt(build_system_prompt())

    def _on_trimmed(self, event_data) -> None:
        """Show context trimming notification."""
        count = len(event_data.trimmed_messages) if event_data.trimmed_messages else 0
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(f"Context trimmed: {count} old messages removed")

    def action_toggle_tools(self) -> None:
        """Toggle the tool panel sidebar."""
        self.query_one("#tools", ToolPanel).toggle()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    app = CodingAgentApp()
    app.run()

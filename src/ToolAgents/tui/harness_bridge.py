"""
Harness bridge — connects an AsyncAgentHarness to the TUI widgets.

The bridge is a controller (not a widget). It listens for user input,
drives the harness, and routes output to the appropriate widgets.

Usage:
    class MyApp(App):
        def on_mount(self):
            harness = create_async_harness(provider=api, tools=tools, streaming=True)
            self.bridge = HarnessBridge(
                app=self,
                harness=harness,
                chat_view=self.query_one(ChatView),
                tool_panel=self.query_one(ToolPanel),
                agent_input=self.query_one(AgentInput),
                status_bar=self.query_one(StatusBar),
            )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import work

if TYPE_CHECKING:
    from textual.app import App

    from ToolAgents.agent_harness.async_harness import AsyncAgentHarness

    from .chat_view import ChatView
    from .input_bar import AgentInput
    from .message_widgets import AssistantMessage
    from .status_bar import StatusBar
    from .tool_panel import ToolCallEntry, ToolPanel


class HarnessBridge:
    """Connects an AsyncAgentHarness to TUI widgets.

    Handles the flow:
    1. User submits text via AgentInput
    2. Bridge disables input, adds UserMessage to ChatView
    3. Calls harness.chat_stream() in a Textual async worker
    4. Routes text chunks to AssistantMessage.stream_token()
    5. Routes tool calls/results to ToolPanel
    6. On completion: re-enables input, updates StatusBar

    Args:
        app: The Textual App instance (needed for @work).
        harness: An AsyncAgentHarness instance.
        chat_view: The ChatView widget.
        tool_panel: The ToolPanel widget (optional).
        agent_input: The AgentInput widget.
        status_bar: The StatusBar widget (optional).
    """

    def __init__(
        self,
        app: "App",
        harness: "AsyncAgentHarness",
        chat_view: "ChatView",
        agent_input: "AgentInput",
        tool_panel: "ToolPanel | None" = None,
        status_bar: "StatusBar | None" = None,
    ) -> None:
        self.app = app
        self.harness = harness
        self.chat_view = chat_view
        self.tool_panel = tool_panel
        self.agent_input = agent_input
        self.status_bar = status_bar

        # Track current streaming state
        self._current_message: AssistantMessage | None = None
        self._current_tool_entry: ToolCallEntry | None = None
        self._processing = False

    async def send_message(self, text: str) -> None:
        """Send a user message through the harness and stream the response.

        This is the main entry point. Call this from an AgentInput.MessageSubmitted handler.

        Args:
            text: The user's message text.
        """
        if self._processing:
            return

        self._processing = True

        # Disable input while processing
        self.agent_input.disable_input()

        if self.status_bar:
            self.status_bar.set_processing(True)

        # Add user message to chat
        self.chat_view.add_user_message(text)

        # Create empty assistant message for streaming
        self._current_message = self.chat_view.add_assistant_message()
        self._current_tool_entry = None

        try:
            # Stream response from harness
            async for chunk in self.harness.chat_stream(text):
                self._process_chunk(chunk)
        except Exception as e:
            self.chat_view.add_system_message(f"Error: {e}")
        finally:
            # Finish the assistant message
            if self._current_message is not None:
                self._current_message.finish()

            # Update status bar
            if self.status_bar:
                state = self.harness.context_state
                self.status_bar.update_status(
                    turn=self.harness.turn_count,
                    context_tokens=state.current_context_tokens,
                    total_tokens=state.total_tokens_used,
                )
                self.status_bar.set_processing(False)

            # Re-enable input
            self._processing = False
            self.agent_input.enable_input()

    def _process_chunk(self, chunk) -> None:
        """Process a single ChatResponseChunk.

        Routes content to the appropriate widget:
        - Text chunks → AssistantMessage.stream_token()
        - Tool calls → ToolPanel.add_tool_call()
        - Tool results → ToolCallEntry.set_result()
        """
        # Handle tool calls
        if chunk.has_tool_call and chunk.tool_call and self.tool_panel is not None:
            tool_name = chunk.tool_call.get("tool_call_name", "")
            tool_args = chunk.tool_call.get("tool_call_arguments")

            if tool_name and tool_args is not None:
                self._current_tool_entry = self.tool_panel.add_tool_call(
                    tool_name, tool_args if isinstance(tool_args, dict) else {}
                )

        # Handle tool results
        if chunk.has_tool_call_result and chunk.tool_call_result:
            result_text = chunk.tool_call_result.get("tool_call_result", "")
            if self._current_tool_entry is not None:
                self._current_tool_entry.set_result(str(result_text))
                self._current_tool_entry = None

        # Handle text chunks — stream to the assistant message
        if chunk.chunk and self._current_message is not None:
            self._current_message.stream_token(chunk.chunk)

    @property
    def is_processing(self) -> bool:
        """Whether the bridge is currently processing a message."""
        return self._processing

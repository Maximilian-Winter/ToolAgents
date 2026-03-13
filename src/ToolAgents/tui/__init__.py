"""
ToolAgents TUI — Composable Textual widgets for building agent interfaces.

A widget library (not a monolithic app) for building terminal user interfaces
powered by ToolAgents harnesses. Users compose widgets into their own Textual apps.

Widgets:
    ChatView        — Scrollable message history container
    UserMessage     — Styled user message
    AssistantMessage — Streaming-capable assistant message
    SystemMessage   — System notification message
    ToolPanel       — Sidebar showing tool call activity
    ToolCallEntry   — Individual tool call display
    AgentInput      — Text input with submit handling
    StatusBar       — Session metrics display

Controller:
    HarnessBridge   — Connects AsyncAgentHarness to TUI widgets

Requires: textual (pip install ToolAgents[tui])
"""

__all__ = [
    # Message widgets
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    # Chat view
    "ChatView",
    # Tool panel
    "ToolPanel",
    "ToolCallEntry",
    # Input
    "AgentInput",
    # Status
    "StatusBar",
    # Bridge
    "HarnessBridge",
]


def __getattr__(name: str):
    if name in {"UserMessage", "AssistantMessage", "SystemMessage"}:
        from .message_widgets import AssistantMessage, SystemMessage, UserMessage

        return {
            "UserMessage": UserMessage,
            "AssistantMessage": AssistantMessage,
            "SystemMessage": SystemMessage,
        }[name]

    if name == "ChatView":
        from .chat_view import ChatView

        return ChatView

    if name in {"ToolPanel", "ToolCallEntry"}:
        from .tool_panel import ToolCallEntry, ToolPanel

        return {
            "ToolPanel": ToolPanel,
            "ToolCallEntry": ToolCallEntry,
        }[name]

    if name == "AgentInput":
        from .input_bar import AgentInput

        return AgentInput

    if name == "StatusBar":
        from .status_bar import StatusBar

        return StatusBar

    if name == "HarnessBridge":
        from .harness_bridge import HarnessBridge

        return HarnessBridge

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

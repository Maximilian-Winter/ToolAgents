"""
Tool panel sidebar for displaying tool call activity.

Shows each tool call as a collapsible entry with name, arguments,
result, and duration. Toggleable via keybinding.
"""

import time
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static


class ToolCallEntry(Collapsible):
    """A single tool call displayed as a collapsible entry.

    Shows tool name in the title bar. Expand to see arguments and result.
    """

    DEFAULT_CSS = """
    ToolCallEntry {
        margin: 0 0 0 0;
        padding: 0;
        height: auto;
    }
    ToolCallEntry .tool-args {
        color: $text-muted;
        padding: 0 1;
    }
    ToolCallEntry .tool-result {
        color: $text;
        padding: 0 1;
    }
    ToolCallEntry .tool-status {
        color: $warning;
        padding: 0 1;
        text-style: italic;
    }
    """

    def __init__(self, tool_name: str, tool_args: dict = None, **kwargs) -> None:
        self._tool_name = tool_name
        self._tool_args = tool_args or {}
        self._result_widget: Static | None = None
        self._status_widget: Static | None = None
        self._start_time = time.monotonic()

        super().__init__(
            title=f"⚙ {tool_name}",
            collapsed=True,
            **kwargs,
        )

    def compose(self):
        """Override compose to add result and status widgets."""
        args_text = self._format_args(self._tool_args)
        yield Static(args_text, classes="tool-args")
        self._status_widget = Static("⏳ Running...", classes="tool-status")
        yield self._status_widget
        self._result_widget = Static("", classes="tool-result")
        yield self._result_widget

    @staticmethod
    def _format_args(args: dict, max_value_len: int = 120) -> str:
        if not args:
            return "(no arguments)"
        lines = []
        for k, v in args.items():
            val_str = str(v)
            if len(val_str) > max_value_len:
                val_str = val_str[:max_value_len] + "..."
            lines.append(f"  {k}: {val_str}")
        return "\n".join(lines)

    def set_result(self, result: str, max_len: int = 500) -> None:
        """Set the tool call result and update display."""
        elapsed = time.monotonic() - self._start_time

        if len(result) > max_len:
            display_result = result[:max_len] + f"... ({len(result)} chars)"
        else:
            display_result = result

        if self._status_widget is not None:
            self._status_widget.update(f"✓ Done ({elapsed:.1f}s)")
            self._status_widget.add_class("tool-done")
        if self._result_widget is not None:
            self._result_widget.update(display_result)

        # Update title with timing
        self.title = f"✓ {self._tool_name} ({elapsed:.1f}s)"


class ToolPanel(VerticalScroll):
    """Sidebar panel showing tool call activity.

    Mount ToolCallEntry widgets here as tools are called.
    Toggle visibility with the `toggle()` method.
    """

    DEFAULT_CSS = """
    ToolPanel {
        dock: right;
        width: 45;
        background: $surface;
        border-left: tall $primary 30%;
        padding: 0 1;
        overflow-y: auto;
    }
    ToolPanel .panel-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        color: $text;
    }
    ToolPanel.hidden {
        display: none;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entry_count = 0

    def compose(self):
        yield Static("Tool Activity", classes="panel-title")

    def add_tool_call(self, tool_name: str, tool_args: dict = None) -> ToolCallEntry:
        """Add a new tool call entry and return the handle.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Dictionary of tool arguments.

        Returns:
            ToolCallEntry: Handle to call set_result() on when done.
        """
        self._entry_count += 1
        entry = ToolCallEntry(tool_name, tool_args)
        self.mount(entry)
        self.scroll_end(animate=False)
        return entry

    def clear_entries(self) -> None:
        """Remove all tool call entries."""
        for entry in self.query(ToolCallEntry):
            entry.remove()
        self._entry_count = 0

    def toggle(self) -> None:
        """Toggle panel visibility."""
        self.toggle_class("hidden")

    @property
    def entry_count(self) -> int:
        return self._entry_count

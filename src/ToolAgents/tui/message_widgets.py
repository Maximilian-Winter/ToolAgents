"""
Message widgets for the TUI chat interface.

Provides styled Markdown-based widgets for different message roles:
- UserMessage: right-margined user prompts
- AssistantMessage: full-width assistant responses with streaming support
- SystemMessage: centered system notifications
"""

from textual.widgets import Markdown, Static


class UserMessage(Static):
    """A user message displayed in the chat view.

    Shows the user's input with distinct styling (right margin, primary background).
    """

    DEFAULT_CSS = """
    UserMessage {
        margin: 1 1 0 8;
        padding: 1 2;
        background: $primary 15%;
        color: $text;
    }
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(text, **kwargs)


class AssistantMessage(Markdown):
    """An assistant message with streaming support.

    Mounts empty, then call `stream_token(chunk)` as tokens arrive.
    When streaming is done, call `finish()`.
    """

    DEFAULT_CSS = """
    AssistantMessage {
        margin: 1 8 0 1;
        padding: 1 2 0 2;
        background: $success 8%;
        color: $text;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._content: str = ""
        self._streaming: bool = True

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    def stream_token(self, token: str) -> None:
        """Append a token to the message and update display."""
        self._content += token
        self.update(self._content)

    def finish(self) -> None:
        """Mark streaming as complete."""
        self._streaming = False
        # Final update to ensure markdown is fully rendered
        self.update(self._content)

    @property
    def content(self) -> str:
        return self._content


class SystemMessage(Static):
    """A system notification message (context trimmed, errors, etc.)."""

    DEFAULT_CSS = """
    SystemMessage {
        margin: 0 4;
        padding: 0 2;
        text-align: center;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(text, **kwargs)

"""
Status bar widget for the TUI.

Shows turn count, context tokens, total tokens used,
and optional custom status text.
"""

from textual.widgets import Static


class StatusBar(Static):
    """Status bar showing agent session metrics.

    Displays turn count, context token usage, and total tokens.
    Docked to the bottom above the input bar.

    Usage:
        status = self.query_one(StatusBar)
        status.update_status(turn=3, context_tokens=45000, total_tokens=120000)
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary 20%;
        color: $text-muted;
        padding: 0 2;
        text-align: center;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Ready", **kwargs)
        self._turn = 0
        self._context_tokens = 0
        self._total_tokens = 0
        self._status_text = ""

    def update_status(
        self,
        turn: int = None,
        context_tokens: int = None,
        total_tokens: int = None,
        status_text: str = None,
    ) -> None:
        """Update the status bar with new metrics.

        Args:
            turn: Current turn number.
            context_tokens: Current context window token count.
            total_tokens: Total tokens used in the session.
            status_text: Optional custom status text to append.
        """
        if turn is not None:
            self._turn = turn
        if context_tokens is not None:
            self._context_tokens = context_tokens
        if total_tokens is not None:
            self._total_tokens = total_tokens
        if status_text is not None:
            self._status_text = status_text

        self._refresh_display()

    def set_processing(self, processing: bool = True) -> None:
        """Show/hide processing indicator."""
        if processing:
            self._status_text = "⏳ Processing..."
        else:
            self._status_text = ""
        self._refresh_display()

    def _refresh_display(self) -> None:
        parts = []

        if self._turn > 0:
            parts.append(f"Turn {self._turn}")

        if self._context_tokens > 0:
            ctx_k = self._context_tokens / 1000
            parts.append(f"Context: {ctx_k:.1f}k")

        if self._total_tokens > 0:
            total_k = self._total_tokens / 1000
            parts.append(f"Total: {total_k:.1f}k tokens")

        if self._status_text:
            parts.append(self._status_text)

        display = " │ ".join(parts) if parts else "Ready"
        self.update(display)

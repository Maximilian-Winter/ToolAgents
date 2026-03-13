"""
Agent input bar for the TUI.

A text input widget that posts a custom Submitted message and clears itself.
Can be disabled during agent processing.
"""

from textual.message import Message
from textual.widgets import Input


class AgentInput(Input):
    """Text input for sending messages to the agent.

    Posts AgentInput.Submitted when the user presses Enter.
    Automatically clears the input after submission.
    Can be disabled/enabled to prevent input during agent processing.

    Usage:
        # In your App compose():
        yield AgentInput(placeholder="Type a message...")

        # Handle submission:
        def on_agent_input_submitted(self, event: AgentInput.Submitted):
            print(event.text)
    """

    DEFAULT_CSS = """
    AgentInput {
        dock: bottom;
        margin: 0 1;
        padding: 0;
    }
    AgentInput.-disabled {
        opacity: 50%;
    }
    """

    class Submitted(Message):
        """Posted when the user submits a message.

        Attributes:
            text: The submitted text.
        """

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(
        self,
        placeholder: str = "Type a message... (Enter to send)",
        **kwargs,
    ) -> None:
        super().__init__(placeholder=placeholder, **kwargs)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key — post our custom Submitted message."""
        event.stop()
        text = self.value.strip()
        if not text:
            return
        self.clear()
        self.post_message(self.Submitted(text))

    def disable_input(self) -> None:
        """Disable input during agent processing."""
        self.disabled = True
        self.add_class("-disabled")

    def enable_input(self) -> None:
        """Re-enable input after agent processing."""
        self.disabled = False
        self.remove_class("-disabled")
        self.focus()

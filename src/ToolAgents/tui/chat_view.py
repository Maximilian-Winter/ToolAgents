"""
Chat view container for the TUI.

A scrollable container that holds message widgets (user, assistant, system).
Auto-scrolls to the bottom as new messages are added.
"""

from textual.containers import VerticalScroll

from .message_widgets import AssistantMessage, SystemMessage, UserMessage


class ChatView(VerticalScroll):
    """Scrollable chat history container.

    Holds UserMessage, AssistantMessage, and SystemMessage widgets.
    Auto-scrolls to bottom when new messages are added.

    Usage:
        chat = self.query_one(ChatView)
        chat.add_user_message("Hello!")
        msg = chat.add_assistant_message()
        msg.stream_token("Hi")
        msg.stream_token(" there!")
        msg.finish()
    """

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def on_mount(self) -> None:
        """Anchor the scroll to the bottom."""
        self.anchor()

    def add_user_message(self, text: str) -> UserMessage:
        """Add a user message to the chat.

        Args:
            text: The user's message text.

        Returns:
            UserMessage: The mounted widget.
        """
        msg = UserMessage(text)
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg

    def add_assistant_message(self) -> AssistantMessage:
        """Add an empty assistant message (for streaming into).

        Returns:
            AssistantMessage: The mounted widget. Call stream_token() on it.
        """
        msg = AssistantMessage()
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg

    def add_system_message(self, text: str) -> SystemMessage:
        """Add a system notification message.

        Args:
            text: The notification text.

        Returns:
            SystemMessage: The mounted widget.
        """
        msg = SystemMessage(text)
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg

    def clear_messages(self) -> None:
        """Remove all messages from the chat."""
        for widget in self.query(UserMessage):
            widget.remove()
        for widget in self.query(AssistantMessage):
            widget.remove()
        for widget in self.query(SystemMessage):
            widget.remove()

import uuid
from typing import List, Dict

from pydantic import BaseModel, Field

from ToolAgents.data_models.messages import ChatMessage


class Context:
    """
    Manages conversation state including messages and ephemeral content.
    Provides interfaces for safe manipulation of chat history.
    """

    def __init__(self, messages: List[ChatMessage] | None = None):
        self._messages: List[ChatMessage] = messages or []
        self._ephemeral_content: Dict[str, EphemeralContent] = {}

    # Message management
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the context."""
        self._messages.append(message)

    def get_messages(self) -> List[ChatMessage]:
        """Get a copy of all messages."""
        return self._messages.copy()

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self._messages)

    def get_last_message(self) -> ChatMessage | None:
        """Get the last message in the context."""
        return self._messages[-1] if self._messages else None

    def remove_message(self, message_id: str) -> bool:
        """Remove a message by ID. Returns True if found and removed."""
        for i, msg in enumerate(self._messages):
            if msg.id == message_id:
                self._messages.pop(i)
                return True
        return False

    # Ephemeral content management
    def add_ephemeral_to_last(
            self,
            content: str,
            remove_after_turns: int = 1,
            identifier: str | None = None
    ) -> str:
        """
        Add ephemeral content to the last user message.
        Returns the identifier for tracking.
        """
        if not self._messages:
            raise ValueError("No messages in context")

        last_msg = self._messages[-1]
        ephemeral_id = identifier or str(uuid.uuid4())

        self._ephemeral_content[ephemeral_id] = EphemeralContent(
            message_id=last_msg.id,
            content=content,
            remove_after_turns=remove_after_turns,
            turns_remaining=remove_after_turns
        )

        return ephemeral_id

    def prepare_for_api(self) -> List[ChatMessage]:
        """
        Get messages with ephemeral content injected.
        This is what you send to the API.
        """
        messages = self._messages.copy()

        # Inject ephemeral content into appropriate messages
        for ephemeral in self._ephemeral_content.values():
            for msg in messages:
                if msg.id == ephemeral.message_id:
                    msg.add_text(f"\n\n{ephemeral.content}")

        return messages

    def after_turn(self) -> None:
        """
        Call this after each agent turn to clean up ephemeral content.
        """
        to_remove = []

        for ephemeral_id, ephemeral in self._ephemeral_content.items():
            ephemeral.turns_remaining -= 1
            if ephemeral.turns_remaining <= 0:
                to_remove.append(ephemeral_id)

        for ephemeral_id in to_remove:
            del self._ephemeral_content[ephemeral_id]

    # Utility methods
    def clear_all_ephemeral(self) -> None:
        """Remove all ephemeral content immediately."""
        self._ephemeral_content.clear()

    def has_tool_calls(self) -> bool:
        """Check if any message contains tool calls."""
        return any(msg.contains_tool_call() for msg in self._messages)

    def get_as_text(self) -> str:
        """Get all messages as formatted text."""
        return "\n\n".join(
            f"[{msg.role.value}]\n{msg.get_as_text()}"
            for msg in self._messages
        )


class EphemeralContent(BaseModel):
    """
    Represents temporary content attached to a message.
    """
    message_id: str = Field(..., description="ID of the message this is attached to")
    content: str = Field(..., description="The ephemeral content")
    remove_after_turns: int = Field(..., description="Total turns before removal")
    turns_remaining: int = Field(..., description="Turns remaining before removal")
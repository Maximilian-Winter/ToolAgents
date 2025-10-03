import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    TextContent,
)


class AgentContext(BaseModel):
    """
    Manages the conversational context for an LLM agent.

    Maintains the system prompt and a sliding window of messages
    that form the agent's working memory.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique context identifier"
    )
    system_prompt: Optional[ChatMessage] = Field(
        default=None,
        description="System message defining agent behavior"
    )
    message_buffer: List[ChatMessage] = Field(
        default_factory=list,
        description="Working memory of recent messages"
    )
    max_buffer_size: Optional[int] = Field(
        default=None,
        description="Maximum messages to retain (None for unlimited)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context metadata"
    )

    def set_system_prompt(self, prompt: str) -> None:
        """Define agent behavior through system message."""
        now = datetime.now()
        self.system_prompt = ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.System,
            content=[TextContent(content=prompt)],
            created_at=now,
            updated_at=now,
        )

    def append(self, message: ChatMessage) -> None:
        """Add message to buffer, maintaining size limit."""
        self.message_buffer.append(message)
        if self.max_buffer_size and len(self.message_buffer) > self.max_buffer_size:
            self._trim_buffer()

    def extend(self, messages: List[ChatMessage]) -> None:
        """Add multiple messages to buffer."""
        for message in messages:
            self.append(message)

    def _trim_buffer(self) -> None:
        """Remove oldest messages to maintain buffer size."""
        excess = len(self.message_buffer) - self.max_buffer_size
        self.message_buffer = self.message_buffer[excess:]

    def get_messages(self, include_system_message=True) -> List[ChatMessage]:
        """Retrieve complete context including system prompt."""
        messages = []
        if self.system_prompt and include_system_message:
            messages.append(self.system_prompt)
        messages.extend(self.message_buffer)
        return messages

    def get_working_messages(self) -> List[ChatMessage]:
        """Retrieve only the message buffer without system prompt."""
        return self.message_buffer.copy()

    def clear_buffer(self) -> None:
        """Clear message buffer while preserving system prompt."""
        self.message_buffer.clear()

    def reset(self) -> None:
        """Complete reset including system prompt."""
        self.system_prompt = None
        self.message_buffer.clear()
        self.metadata.clear()

    def fork(self) -> "AgentContext":
        """Create independent copy for branching conversations."""
        return AgentContext(
            system_prompt=self.system_prompt.model_copy() if self.system_prompt else None,
            message_buffer=[msg.model_copy() for msg in self.message_buffer],
            max_buffer_size=self.max_buffer_size,
            metadata=self.metadata.copy()
        )

    def get_token_estimate(self) -> int:
        """Rough token count estimate for context size management."""
        total_chars = 0
        for message in self.get_messages():
            total_chars += len(message.get_as_text())
        return total_chars // 4

    def compress(self, target_size: int) -> None:
        """Reduce context to target message count, preserving recent messages."""
        if len(self.message_buffer) <= target_size:
            return
        self.message_buffer = self.message_buffer[-target_size:]

    def find_last_user_message(self) -> Optional[ChatMessage]:
        """Locate most recent user input."""
        for message in reversed(self.message_buffer):
            if message.role == ChatMessageRole.User:
                return message
        return None

    def find_last_assistant_message(self) -> Optional[ChatMessage]:
        """Locate most recent assistant response."""
        for message in reversed(self.message_buffer):
            if message.role == ChatMessageRole.Assistant:
                return message
        return None

    def get_conversation_pairs(self) -> List[tuple[ChatMessage, Optional[ChatMessage]]]:
        """Extract user-assistant message pairs."""
        pairs = []
        i = 0
        while i < len(self.message_buffer):
            if self.message_buffer[i].role == ChatMessageRole.User:
                user_msg = self.message_buffer[i]
                assistant_msg = None
                if i + 1 < len(self.message_buffer):
                    if self.message_buffer[i + 1].role == ChatMessageRole.Assistant:
                        assistant_msg = self.message_buffer[i + 1]
                        i += 1
                pairs.append((user_msg, assistant_msg))
            i += 1
        return pairs
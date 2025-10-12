import enum
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    TextContent,
)

class ContextMessageBufferStrategy(enum.Enum):
    all_messages = "all_messages"
    last_k_messages = "last_k_messages"

class AgentContext(BaseModel):
    """
    Manages the conversational context for an LLM agent.
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
    message_buffer_strategy: ContextMessageBufferStrategy = Field(
        default=ContextMessageBufferStrategy.all_messages,
        description="Message buffer strategy for conversational context"
    )
    message_buffer_current_index: int = Field(
        default=0,
        description="Current message buffer index"
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

    def get_messages(self, include_system_message=True) -> List[ChatMessage]:
        """Retrieve complete context including system prompt."""
        messages = []
        if self.system_prompt and include_system_message:
            messages.append(self.system_prompt)
        messages.extend(self.message_buffer)
        return messages

    def reset(self, reset_system_message: bool = True) -> None:
        """Complete reset including system prompt."""
        self.system_prompt = None if reset_system_message else self.system_prompt
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

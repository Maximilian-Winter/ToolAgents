from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from agora.db.models.enums import MessageType
from agora.schemas.agent import AgentOut


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class RoomCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    topic: Optional[str] = None


class MessageCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    sender: str = Field(..., min_length=1, max_length=100)
    token: Optional[str] = None
    content: str = Field(..., min_length=1)
    message_type: MessageType = MessageType.statement
    reply_to: Optional[int] = None
    to: Optional[str] = Field(None, max_length=100)


class MessageEdit(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    sender: str = Field(..., min_length=1, max_length=100)
    token: Optional[str] = None
    content: str = Field(..., min_length=1)


class ReactionCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    sender: str = Field(..., min_length=1, max_length=100)
    token: Optional[str] = None
    emoji: str = Field(..., min_length=1, max_length=10)


class ReceiptUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent: str = Field(..., min_length=1, max_length=100)
    token: Optional[str] = None
    last_read: int


class TypingRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    sender: str = Field(..., min_length=1, max_length=100)
    token: Optional[str] = None


# PollParams and WaitParams are NOT Pydantic models -- use Query params directly in routes.


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class RoomOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    name: str
    topic: Optional[str]
    current_round: int = 1
    created_at: datetime


class ReactionSummary(BaseModel):
    emoji: str
    count: int
    senders: list[str]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    room_id: int
    sender: str
    content: str
    message_type: MessageType
    reply_to: Optional[int]
    to: Optional[str] = None
    edited_at: Optional[datetime] = None
    edit_history: Optional[list[dict]] = None
    created_at: datetime
    reactions: list[ReactionSummary] = []


class ReceiptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    agent: str
    last_read: int
    updated_at: datetime


class AgentPresence(BaseModel):
    agent: str
    status: str  # online, idle, offline


class RoomStatus(BaseModel):
    room: RoomOut
    message_count: int
    members: list[AgentOut]
    receipts: list[ReceiptOut]
    presence: list[AgentPresence] = []
    typing: list[str] = []


class PollResponse(BaseModel):
    messages: list[MessageOut]
    receipts: list[ReceiptOut]


class ThreadedMessage(BaseModel):
    """A message with its replies nested inline."""
    message: MessageOut
    replies: list[ThreadedMessage] = []


class ThreadedResponse(BaseModel):
    threads: list[ThreadedMessage]


class ProposalChain(BaseModel):
    """A proposal with its objections, supporting votes, and consensus."""
    proposal: MessageOut
    objections: list[MessageOut] = []
    answers: list[MessageOut] = []
    consensus: list[MessageOut] = []
    reactions: dict[str, int] = {}  # emoji -> count across proposal


class RoomSummary(BaseModel):
    room: str
    current_round: int
    total_messages: int
    participants: list[str]
    proposal_chains: list[ProposalChain] = []
    open_questions: list[MessageOut] = []
    key_statements: list[MessageOut] = []
    decisions: list[MessageOut] = []  # consensus messages not tied to proposals

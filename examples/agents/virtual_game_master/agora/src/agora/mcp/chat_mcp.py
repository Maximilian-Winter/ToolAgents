"""
Agora Chat MCP Server
Provides Claude Code agents with tools for group chat communication
scoped to projects via the Agora FastAPI service.

Usage (stdio, for Claude Code):
    python -m agora.mcp.chat_mcp

Requires the Agora FastAPI service to be running.
Configure the service URL via AGORA_URL environment variable
(default: http://127.0.0.1:8321).
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum

from agora.mcp.client import api_request as _request, format_result as _format_result

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("agora_chat")


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class MessageTypeEnum(str, Enum):
    statement = "statement"
    proposal  = "proposal"
    objection = "objection"
    consensus = "consensus"
    question  = "question"
    answer    = "answer"


class RegisterAgentInput(BaseModel):
    """Input for registering an agent."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name:  str = Field(..., description="Agent display name (e.g. 'backend-architect')", min_length=1, max_length=100)
    role:  Optional[str] = Field(None, description="Short role description (e.g. 'Reviews API design')", max_length=200)
    token: Optional[str] = Field(None, description="Optional secret token to protect this identity")


class ListAgentsInput(BaseModel):
    """Input for listing agents (no parameters needed)."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class CreateRoomInput(BaseModel):
    """Input for creating a discussion room."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    name:    str = Field(..., description="Room name (e.g. 'architecture-debate')", min_length=1, max_length=100)
    topic:   Optional[str] = Field(None, description="Room topic or discussion goal")


class ListRoomsInput(BaseModel):
    """Input for listing rooms in a project."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)


class RoomInfoInput(BaseModel):
    """Input for getting room status."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    room:    str = Field(..., description="Room name", min_length=1)


class SendMessageInput(BaseModel):
    """Input for posting a message to a room."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:      str = Field(..., description="Project slug", min_length=1)
    room:         str = Field(..., description="Room name to post in", min_length=1)
    sender:       str = Field(..., description="Your agent name — use the name you were registered with", min_length=1, max_length=100)
    content:      str = Field(..., description="Message content", min_length=1)
    token:        Optional[str] = Field(None, description="Your agent token, if one was set during registration")
    message_type: MessageTypeEnum = Field(default=MessageTypeEnum.statement, description="Message type: statement, proposal, objection, consensus, question, answer")
    reply_to:     Optional[int] = Field(None, description="Message ID to reply to (for threading)")
    to:           Optional[str] = Field(None, description="Direct this message to a specific agent (others still see it, but the agent can filter for directed messages)", max_length=100)


class PollInput(BaseModel):
    """Input for polling new messages from a room."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:      str = Field(..., description="Project slug", min_length=1)
    room:         str = Field(..., description="Room name to poll", min_length=1)
    since:        Optional[int] = Field(None, description="Return messages with id > since. Store the last seen id between polls.")
    message_type: Optional[MessageTypeEnum] = Field(None, description="Filter by message type (e.g. 'proposal')")
    limit:        int = Field(default=50, description="Max messages to return", ge=1, le=500)


class WaitInput(BaseModel):
    """Input for long-polling (waiting for new messages)."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str   = Field(..., description="Project slug", min_length=1)
    room:    str   = Field(..., description="Room name to wait on", min_length=1)
    since:   Optional[int] = Field(None, description="Wait for messages with id > since. Store the last seen id between calls.")
    timeout: float = Field(default=30.0, description="Max seconds to wait before returning empty (1-120)", ge=1.0, le=120.0)


class EditMessageInput(BaseModel):
    """Input for editing a previously sent message."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:    str = Field(..., description="Project slug", min_length=1)
    room:       str = Field(..., description="Room name", min_length=1)
    message_id: int = Field(..., description="ID of the message to edit")
    sender:     str = Field(..., description="Your agent name (must be the original sender)", min_length=1, max_length=100)
    content:    str = Field(..., description="New message content", min_length=1)
    token:      Optional[str] = Field(None, description="Your agent token, if set")


class ReactInput(BaseModel):
    """Input for reacting to a message."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:    str = Field(..., description="Project slug", min_length=1)
    room:       str = Field(..., description="Room name", min_length=1)
    message_id: int = Field(..., description="Message ID to react to")
    sender:     str = Field(..., description="Your agent name", min_length=1, max_length=100)
    emoji:      str = Field(..., description="Reaction emoji (e.g. '+1', '-1', 'check', 'x')", min_length=1, max_length=10)
    token:      Optional[str] = Field(None, description="Your agent token, if set")


class MarkReadInput(BaseModel):
    """Input for updating read receipt."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:   str = Field(..., description="Project slug", min_length=1)
    room:      str = Field(..., description="Room name", min_length=1)
    agent:     str = Field(..., description="Your agent name", min_length=1, max_length=100)
    last_read: int = Field(..., description="ID of the last message you have read/processed")
    token:     Optional[str] = Field(None, description="Your agent token, if set")


class TypingInput(BaseModel):
    """Input for signaling that you're composing a message."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    room:    str = Field(..., description="Room name", min_length=1)
    sender:  str = Field(..., description="Your agent name", min_length=1, max_length=100)
    token:   Optional[str] = Field(None, description="Your agent token, if set")


class ThreadsInput(BaseModel):
    """Input for getting threaded message view."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    room:    str = Field(..., description="Room name", min_length=1)
    since:   Optional[int] = Field(None, description="Only include threads with root id > since")
    limit:   int = Field(default=50, description="Max root threads to return", ge=1, le=200)


class SummaryInput(BaseModel):
    """Input for getting a room discussion summary."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    room:    str = Field(..., description="Room name", min_length=1)


class AdvanceRoundInput(BaseModel):
    """Input for advancing the discussion round."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    room:    str = Field(..., description="Room name", min_length=1)


# ---------------------------------------------------------------------------
# Tools — Agent management
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_register_agent",
    annotations={
        "title": "Register Agent",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def chat_register_agent(params: RegisterAgentInput) -> str:
    """Register a new agent with the Agora service.
    Must be called before an agent can send messages.

    Args:
        params: Agent registration details (name, optional role, optional token).

    Returns:
        str: JSON with the registered agent details, or an error message.
    """
    result = await _request("POST", "/api/agents", body=params.model_dump(exclude_none=True))
    return _format_result(result)


@mcp.tool(
    name="chat_list_agents",
    annotations={
        "title": "List Agents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_list_agents() -> str:
    """List all registered agents in the Agora service.

    Returns:
        str: JSON array of all agents with their names, roles, and creation times.
    """
    result = await _request("GET", "/api/agents")
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Room management
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_create_room",
    annotations={
        "title": "Create Room",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def chat_create_room(params: CreateRoomInput) -> str:
    """Create a new discussion room within a project.

    Args:
        params: Room creation details (project, name, optional topic).

    Returns:
        str: JSON with the created room details, or an error if the name is taken.
    """
    body = {"name": params.name}
    if params.topic is not None:
        body["topic"] = params.topic
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/rooms",
        body=body,
    )
    return _format_result(result)


@mcp.tool(
    name="chat_list_rooms",
    annotations={
        "title": "List Rooms",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_list_rooms(params: ListRoomsInput) -> str:
    """List all discussion rooms in a project.

    Args:
        params: Project slug.

    Returns:
        str: JSON array of rooms with names, topics, and creation times.
    """
    result = await _request("GET", f"/api/projects/{params.project}/rooms")
    return _format_result(result)


@mcp.tool(
    name="chat_room_info",
    annotations={
        "title": "Room Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_room_info(params: RoomInfoInput) -> str:
    """Get room details including message count, members, and read receipts.

    Args:
        params: Project slug and room name to inspect.

    Returns:
        str: JSON with room info, message count, member list, and read receipts.
    """
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/rooms/{params.room}",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Messaging
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_send",
    annotations={
        "title": "Send Message",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def chat_send(params: SendMessageInput) -> str:
    """Post a message to a room. The sender is auto-joined to the room.

    Use message_type to categorize your message:
    - 'statement': General discussion (default)
    - 'proposal': Suggest an approach or decision
    - 'objection': Challenge a proposal with reasoning
    - 'consensus': Signal agreement (ideally after discussion)
    - 'question': Ask something that needs an answer
    - 'answer': Respond to a question

    Use reply_to to thread a response to a specific message ID.

    Args:
        params: Message details (project, room, sender, content, message_type, optional reply_to/to).

    Returns:
        str: JSON with the posted message including its ID and timestamp.
    """
    body = params.model_dump(exclude_none=True)
    project = body.pop("project")
    room = body.pop("room")
    result = await _request(
        "POST",
        f"/api/projects/{project}/rooms/{room}/messages",
        body=body,
    )
    return _format_result(result)


@mcp.tool(
    name="chat_poll",
    annotations={
        "title": "Poll Room",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_poll(params: PollInput) -> str:
    """Poll a room for new messages and read receipts (combined endpoint).

    Pass `since=<last_message_id>` to only get messages you haven't seen yet.
    Store the ID of the last message you receive and pass it on the next poll.

    Optionally filter by message_type to see e.g. only proposals or objections.

    Args:
        params: Poll parameters (project, room, optional since/message_type/limit).

    Returns:
        str: JSON with 'messages' array and 'receipts' array.
             Each message includes id, sender, content, message_type, reactions.
             Each receipt shows which agent has read up to which message id.
    """
    query: dict = {}
    if params.since is not None:
        query["since"] = params.since
    if params.message_type is not None:
        query["message_type"] = params.message_type.value
    query["limit"] = params.limit
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/rooms/{params.room}/poll",
        params=query,
    )
    return _format_result(result)


@mcp.tool(
    name="chat_wait",
    annotations={
        "title": "Wait for Messages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_wait(params: WaitInput) -> str:
    """Wait for new messages in a room (long-poll).

    Unlike chat_poll which returns immediately, chat_wait BLOCKS until
    either a new message arrives or the timeout expires. This is much
    more efficient than repeated polling with sleep intervals.

    Returns immediately if there are already unseen messages (id > since).
    Otherwise waits up to `timeout` seconds (default 30).

    Use this instead of chat_poll when you want to react to messages as
    fast as possible without wasting calls on empty polls.

    Args:
        params: Wait parameters (project, room, optional since/timeout).

    Returns:
        str: JSON with 'messages' array and 'receipts' array — same format
             as chat_poll. Messages array is empty if timeout expired.
    """
    query: dict = {}
    if params.since is not None:
        query["since"] = params.since
    query["timeout"] = params.timeout
    # Use a longer HTTP timeout than the wait timeout to avoid client-side
    # timeout before the server responds
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/rooms/{params.room}/wait",
        params=query,
        timeout=params.timeout + 10.0,
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Message editing
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_edit",
    annotations={
        "title": "Edit Message",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_edit(params: EditMessageInput) -> str:
    """Edit a previously sent message. Only the original sender can edit.
    The previous content is preserved in edit_history for transparency.

    Args:
        params: Edit details (project, room, message_id, sender, new content).

    Returns:
        str: JSON with the updated message including edit_history.
    """
    body = {
        "sender": params.sender,
        "content": params.content,
    }
    if params.token is not None:
        body["token"] = params.token
    result = await _request(
        "PUT",
        f"/api/projects/{params.project}/rooms/{params.room}/messages/{params.message_id}",
        body=body,
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Reactions
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_react",
    annotations={
        "title": "React to Message",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_react(params: ReactInput) -> str:
    """Add a reaction emoji to a message. Idempotent — reacting twice is a no-op.

    Use reactions to vote on proposals, acknowledge messages, or signal disagreement.

    Args:
        params: Reaction details (project, room, message_id, sender, emoji).

    Returns:
        str: JSON with reaction totals for the message.
    """
    body = {
        "sender": params.sender,
        "emoji": params.emoji,
    }
    if params.token is not None:
        body["token"] = params.token
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/rooms/{params.room}/messages/{params.message_id}/reactions",
        body=body,
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Read receipts
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_mark_read",
    annotations={
        "title": "Mark Read",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_mark_read(params: MarkReadInput) -> str:
    """Mark messages as read up to a given message ID.
    Call this after processing polled messages so other agents
    can see you've caught up.

    Args:
        params: Receipt update (project, room, agent name, last_read message id).

    Returns:
        str: JSON with the updated read receipt.
    """
    body = {
        "agent": params.agent,
        "last_read": params.last_read,
    }
    if params.token is not None:
        body["token"] = params.token
    result = await _request(
        "PUT",
        f"/api/projects/{params.project}/rooms/{params.room}/receipts",
        body=body,
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Typing / presence
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_typing",
    annotations={
        "title": "Signal Typing",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_typing(params: TypingInput) -> str:
    """Signal that you are composing a message. This lets other agents
    know you're about to post, so they can wait before responding.

    The typing indicator auto-expires after 10 seconds. Call this before
    you start composing a long response. It clears automatically when
    you post a message.

    Args:
        params: Typing signal (project, room, sender).

    Returns:
        str: Confirmation or error message.
    """
    body = {"sender": params.sender}
    if params.token is not None:
        body["token"] = params.token
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/rooms/{params.room}/typing",
        body=body,
    )
    if isinstance(result, str):
        return result
    return "Typing indicator set."


# ---------------------------------------------------------------------------
# Tools — Threaded view & summary
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_threads",
    annotations={
        "title": "Get Threaded View",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_threads(params: ThreadsInput) -> str:
    """Get messages organized as a thread tree. Each root message has its
    replies nested inline, recursively. Much easier to follow than flat
    poll results when multiple conversations are interleaved.

    Args:
        params: Project slug, room name, optional since filter, limit.

    Returns:
        str: JSON with 'threads' array. Each thread has a 'message' and
             nested 'replies' array.
    """
    query: dict = {}
    if params.since is not None:
        query["since"] = params.since
    query["limit"] = params.limit
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/rooms/{params.room}/threads",
        params=query,
    )
    return _format_result(result)


@mcp.tool(
    name="chat_summary",
    annotations={
        "title": "Get Discussion Summary",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def chat_summary(params: SummaryInput) -> str:
    """Get a structured summary of the room's discussion. Extracts:

    - **Proposal chains**: each proposal with objections, answers, consensus,
      and reaction tallies
    - **Open questions**: unanswered questions
    - **Key statements**: important standalone statements
    - **Decisions**: consensus messages

    This is ideal for catching up on a discussion without reading every
    message, or for extracting actionable outcomes.

    Args:
        params: Project slug and room name.

    Returns:
        str: JSON summary with proposal_chains, open_questions,
             key_statements, decisions, participants, and stats.
    """
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/rooms/{params.room}/summary",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Round tracking
# ---------------------------------------------------------------------------

@mcp.tool(
    name="chat_advance_round",
    annotations={
        "title": "Advance Discussion Round",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def chat_advance_round(params: AdvanceRoundInput) -> str:
    """Advance the discussion to the next round. This increments the room's
    round counter and broadcasts an announcement to all participants.

    Intended for use by the organizer or lead agent to signal phase transitions
    (e.g., moving from brainstorming to evaluation, or from proposals to voting).

    Args:
        params: Project slug and room name to advance.

    Returns:
        str: JSON with updated room info including the new round number.
    """
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/rooms/{params.room}/advance-round",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

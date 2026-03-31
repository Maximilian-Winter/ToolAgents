"""Chat routes -- rooms, messages, reactions, receipts, typing, real-time."""

import asyncio
import json as _json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agora.db.engine import get_db, async_session
from agora.db.models.agent import Agent
from agora.db.models.chat import Room, Message, Reaction, ReadReceipt, RoomMember
from agora.db.models.enums import MessageType
from agora.api.deps import require_agent, require_project
from agora.realtime import broadcaster, presence
from agora.schemas.agent import AgentOut
from agora.schemas.chat import (
    RoomCreate,
    RoomOut,
    RoomStatus,
    MessageCreate,
    MessageEdit,
    MessageOut,
    ReactionCreate,
    ReceiptUpdate,
    ReceiptOut,
    TypingRequest,
    AgentPresence,
    PollResponse,
    ThreadedResponse,
    RoomSummary,
)
from agora.services.mention_service import store_mentions
from agora.services.chat_service import (
    build_message_out,
    ensure_membership,
    build_threaded_view,
    build_room_summary,
)

router = APIRouter(
    prefix="/api/projects/{project_slug}/rooms",
    tags=["Chat"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _require_room(room_name: str, project_id: int, db: AsyncSession) -> Room:
    """Look up a room by name within a project, or raise 404."""
    result = await db.execute(
        select(Room).where(and_(Room.name == room_name, Room.project_id == project_id))
    )
    room = result.scalar_one_or_none()
    if not room:
        raise HTTPException(404, f"Room '{room_name}' not found in this project")
    return room


# ---------------------------------------------------------------------------
# 1. POST "" -- create room in project
# ---------------------------------------------------------------------------


@router.post("", response_model=RoomOut, status_code=201)
async def create_room(
    project_slug: str,
    body: RoomCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new discussion room within a project."""
    project = await require_project(project_slug, db)

    existing = await db.execute(
        select(Room).where(and_(Room.name == body.name, Room.project_id == project.id))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Room '{body.name}' already exists in this project")

    room = Room(name=body.name, topic=body.topic, project_id=project.id)
    db.add(room)
    await db.commit()
    await db.refresh(room)
    return room


# ---------------------------------------------------------------------------
# 2. GET "" -- list rooms in project
# ---------------------------------------------------------------------------


@router.get("", response_model=list[RoomOut])
async def list_rooms(
    project_slug: str,
    db: AsyncSession = Depends(get_db),
):
    """List all rooms in a project."""
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Room).where(Room.project_id == project.id).order_by(Room.created_at)
    )
    return result.scalars().all()


# ---------------------------------------------------------------------------
# 3. GET "/{room_name}" -- room status
# ---------------------------------------------------------------------------


@router.get("/{room_name}", response_model=RoomStatus)
async def get_room(
    project_slug: str,
    room_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get room details, message count, members, receipts, presence, typing."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    count_result = await db.execute(
        select(func.count()).where(Message.room_id == room.id)
    )
    count = count_result.scalar()

    members_result = await db.execute(
        select(Agent)
        .join(RoomMember, RoomMember.agent_id == Agent.id)
        .where(RoomMember.room_id == room.id)
        .order_by(RoomMember.joined_at)
    )
    members = members_result.scalars().all()

    receipts_result = await db.execute(
        select(ReadReceipt).where(ReadReceipt.room_id == room.id)
    )
    receipts = receipts_result.scalars().all()

    # Presence and typing
    agent_presence = [
        AgentPresence(agent=a.name, status=presence.get_status(a.name))
        for a in members
    ]
    typing_agents = presence.get_typing(room_name)

    return RoomStatus(
        room=RoomOut.model_validate(room),
        message_count=count,
        members=[AgentOut.model_validate(a) for a in members],
        receipts=[ReceiptOut.model_validate(r) for r in receipts],
        presence=agent_presence,
        typing=typing_agents,
    )


# ---------------------------------------------------------------------------
# 4. DELETE "/{room_name}" -- delete room
# ---------------------------------------------------------------------------


@router.delete("/{room_name}", status_code=204)
async def delete_room(
    project_slug: str,
    room_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a room and all its messages."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    await db.delete(room)
    await db.commit()


# ---------------------------------------------------------------------------
# 5. POST "/{room_name}/messages" -- post message
# ---------------------------------------------------------------------------


@router.post("/{room_name}/messages", response_model=MessageOut, status_code=201)
async def post_message(
    project_slug: str,
    room_name: str,
    body: MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Post a message to a room. Auto-joins the sender if not already a member."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    agent = await require_agent(body.sender, body.token, db)

    # Auto-join on first post
    await ensure_membership(room, agent, db)

    if body.reply_to:
        parent = await db.get(Message, body.reply_to)
        if not parent or parent.room_id != room.id:
            raise HTTPException(404, "reply_to message not found in this room")

    msg = Message(
        room_id=room.id,
        sender=agent.name,
        content=body.content,
        message_type=body.message_type,
        reply_to=body.reply_to,
        to=body.to,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg, ["reactions"])

    await store_mentions(project.id, "message", msg.id, body.content, db)
    await db.commit()

    out = build_message_out(msg)

    # Update presence and clear typing
    presence.touch(agent.name)
    presence.clear_typing(room_name, agent.name)

    # Broadcast to SSE / long-poll subscribers
    broadcaster.publish(room_name, "message", out.model_dump(mode="json"))

    return out


# ---------------------------------------------------------------------------
# 6. GET "/{room_name}/messages" -- get messages
# ---------------------------------------------------------------------------


@router.get("/{room_name}/messages", response_model=list[MessageOut])
async def get_messages(
    project_slug: str,
    room_name: str,
    since: Optional[int] = Query(None, description="Return messages with id > since"),
    message_type: Optional[MessageType] = Query(None, description="Filter by message type"),
    for_agent: Optional[str] = Query(None, description="Filter to messages addressed to this agent (+ broadcasts)"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Poll for messages. Pass since=<last_seen_id> to get only new ones."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    q = (
        select(Message)
        .where(Message.room_id == room.id)
        .options(selectinload(Message.reactions))
    )
    if since is not None:
        q = q.where(Message.id > since)
    if message_type is not None:
        q = q.where(Message.message_type == message_type)
    if for_agent is not None:
        q = q.where(or_(Message.to == for_agent, Message.to.is_(None)))
    q = q.order_by(Message.id).limit(limit)

    msgs_result = await db.execute(q)
    msgs = msgs_result.scalars().all()

    return [build_message_out(msg) for msg in msgs]


# ---------------------------------------------------------------------------
# 7. PUT "/{room_name}/messages/{message_id}" -- edit message
# ---------------------------------------------------------------------------


@router.put("/{room_name}/messages/{message_id}", response_model=MessageOut)
async def edit_message(
    project_slug: str,
    room_name: str,
    message_id: int,
    body: MessageEdit,
    db: AsyncSession = Depends(get_db),
):
    """Edit a message's content. Only the original sender can edit."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    agent = await require_agent(body.sender, body.token, db)

    msg = await db.get(Message, message_id, options=[selectinload(Message.reactions)])
    if not msg or msg.room_id != room.id:
        raise HTTPException(404, "Message not found in this room")
    if msg.sender != agent.name:
        raise HTTPException(403, "Only the original sender can edit a message")

    # Preserve previous version in edit_history
    history = []
    if msg.edit_history:
        try:
            history = _json.loads(msg.edit_history)
        except (ValueError, TypeError):
            history = []
    history.append({
        "content": msg.content,
        "edited_at": datetime.now(timezone.utc).isoformat(),
    })
    msg.edit_history = _json.dumps(history)
    msg.content = body.content
    msg.edited_at = datetime.now(timezone.utc)

    await db.commit()

    await store_mentions(project.id, "message", msg.id, body.content, db)
    await db.commit()

    await db.refresh(msg, ["reactions"])

    out = build_message_out(msg)

    # Broadcast edit event
    broadcaster.publish(room_name, "edit", out.model_dump(mode="json"))

    return out


# ---------------------------------------------------------------------------
# 8. GET "/{room_name}/poll" -- combined poll (messages + receipts)
# ---------------------------------------------------------------------------


@router.get("/{room_name}/poll", response_model=PollResponse)
async def poll_room(
    project_slug: str,
    room_name: str,
    since: Optional[int] = Query(None, description="Return messages with id > since"),
    message_type: Optional[MessageType] = Query(None, description="Filter by message type"),
    for_agent: Optional[str] = Query(None, description="Filter to messages addressed to this agent (+ broadcasts)"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Combined poll: returns new messages AND current read receipts in one call."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    # Messages
    q = (
        select(Message)
        .where(Message.room_id == room.id)
        .options(selectinload(Message.reactions))
    )
    if since is not None:
        q = q.where(Message.id > since)
    if message_type is not None:
        q = q.where(Message.message_type == message_type)
    if for_agent is not None:
        q = q.where(or_(Message.to == for_agent, Message.to.is_(None)))
    q = q.order_by(Message.id).limit(limit)

    msgs_result = await db.execute(q)
    msgs = msgs_result.scalars().all()

    # Receipts
    receipts_result = await db.execute(
        select(ReadReceipt).where(ReadReceipt.room_id == room.id)
    )
    receipts = receipts_result.scalars().all()

    return PollResponse(
        messages=[build_message_out(msg) for msg in msgs],
        receipts=[ReceiptOut.model_validate(r) for r in receipts],
    )


# ---------------------------------------------------------------------------
# 9. GET "/{room_name}/wait" -- long-poll
# ---------------------------------------------------------------------------


@router.get("/{room_name}/wait", response_model=PollResponse)
async def wait_for_messages(
    project_slug: str,
    room_name: str,
    since: Optional[int] = Query(None, description="Return messages with id > since"),
    timeout: float = Query(30.0, ge=1.0, le=120.0, description="Max seconds to wait"),
    db: AsyncSession = Depends(get_db),
):
    """Long-poll: blocks until new messages arrive or timeout expires."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    # First, check if there are already unseen messages
    q = (
        select(Message)
        .where(Message.room_id == room.id)
        .options(selectinload(Message.reactions))
    )
    if since is not None:
        q = q.where(Message.id > since)
    q = q.order_by(Message.id).limit(100)

    result = await db.execute(q)
    existing = result.scalars().all()

    if existing:
        receipts_result = await db.execute(
            select(ReadReceipt).where(ReadReceipt.room_id == room.id)
        )
        receipts = receipts_result.scalars().all()
        return PollResponse(
            messages=[build_message_out(m) for m in existing],
            receipts=[ReceiptOut.model_validate(r) for r in receipts],
        )

    # No unseen messages -- subscribe and wait
    queue = broadcaster.subscribe(room_name)
    try:
        try:
            await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            receipts_result = await db.execute(
                select(ReadReceipt).where(ReadReceipt.room_id == room.id)
            )
            receipts = receipts_result.scalars().all()
            return PollResponse(
                messages=[],
                receipts=[ReceiptOut.model_validate(r) for r in receipts],
            )

        # Got an event -- fetch all messages since the requested ID
        async with async_session() as fresh_db:
            q = (
                select(Message)
                .where(Message.room_id == room.id)
                .options(selectinload(Message.reactions))
            )
            if since is not None:
                q = q.where(Message.id > since)
            q = q.order_by(Message.id).limit(100)

            result = await fresh_db.execute(q)
            messages = result.scalars().all()

            receipts_result = await fresh_db.execute(
                select(ReadReceipt).where(ReadReceipt.room_id == room.id)
            )
            receipts = receipts_result.scalars().all()

        return PollResponse(
            messages=[build_message_out(m) for m in messages],
            receipts=[ReceiptOut.model_validate(r) for r in receipts],
        )
    finally:
        broadcaster.unsubscribe(room_name, queue)


# ---------------------------------------------------------------------------
# 10. GET "/{room_name}/stream" -- SSE stream
# ---------------------------------------------------------------------------


@router.get("/{room_name}/stream")
async def stream_room(
    request: Request,
    project_slug: str,
    room_name: str,
    since: Optional[int] = Query(None, description="Backfill messages after this ID"),
    db: AsyncSession = Depends(get_db),
):
    """Stream room events via Server-Sent Events (SSE)."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    room_id = room.id

    async def event_generator():
        queue = broadcaster.subscribe(room_name)
        try:
            # Backfill missed messages
            if since is not None:
                async with async_session() as backfill_db:
                    q = (
                        select(Message)
                        .where(Message.room_id == room_id)
                        .where(Message.id > since)
                        .options(selectinload(Message.reactions))
                        .order_by(Message.id)
                        .limit(500)
                    )
                    result = await backfill_db.execute(q)
                    for msg in result.scalars().all():
                        out = build_message_out(msg)
                        payload = _json.dumps(out.model_dump(mode="json"), default=str)
                        yield f"event: message\ndata: {payload}\n\n"

            # Live stream
            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    payload = _json.dumps(event["data"], default=str)
                    yield f"event: {event['event']}\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            broadcaster.unsubscribe(room_name, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 11. GET "/{room_name}/threads" -- threaded view
# ---------------------------------------------------------------------------


@router.get("/{room_name}/threads", response_model=ThreadedResponse)
async def get_threads(
    project_slug: str,
    room_name: str,
    since: Optional[int] = Query(None, description="Only include threads with root id > since"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Return messages organized as a thread tree."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    threads = await build_threaded_view(room.id, db, since=since, limit=limit)
    return ThreadedResponse(threads=threads)


# ---------------------------------------------------------------------------
# 12. GET "/{room_name}/summary" -- discussion summary
# ---------------------------------------------------------------------------


@router.get("/{room_name}/summary", response_model=RoomSummary)
async def get_summary(
    project_slug: str,
    room_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate a structured summary of the room's discussion."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    summary_data = await build_room_summary(room, db)
    return RoomSummary(**summary_data)


# ---------------------------------------------------------------------------
# 13. POST "/{room_name}/messages/{message_id}/reactions" -- add reaction
# ---------------------------------------------------------------------------


@router.post("/{room_name}/messages/{message_id}/reactions", status_code=201)
async def add_reaction(
    project_slug: str,
    room_name: str,
    message_id: int,
    body: ReactionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a reaction (emoji) to a message. Idempotent."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    await require_agent(body.sender, body.token, db)

    msg = await db.get(Message, message_id)
    if not msg or msg.room_id != room.id:
        raise HTTPException(404, "Message not found in this room")

    existing = await db.execute(
        select(Reaction).where(and_(
            Reaction.message_id == message_id,
            Reaction.sender == body.sender,
            Reaction.emoji == body.emoji,
        ))
    )
    if not existing.scalar_one_or_none():
        reaction = Reaction(message_id=message_id, sender=body.sender, emoji=body.emoji)
        db.add(reaction)
        await db.commit()

    await db.refresh(msg, ["reactions"])
    totals: dict[str, int] = {}
    for r in msg.reactions:
        totals[r.emoji] = totals.get(r.emoji, 0) + 1

    out = {"message_id": message_id, "emoji": body.emoji, "totals": totals}

    # Broadcast reaction event
    broadcaster.publish(room_name, "reaction", out)

    return out


# ---------------------------------------------------------------------------
# 14. DELETE "/{room_name}/messages/{message_id}/reactions" -- remove reaction
# ---------------------------------------------------------------------------


@router.delete("/{room_name}/messages/{message_id}/reactions", status_code=204)
async def remove_reaction(
    project_slug: str,
    room_name: str,
    message_id: int,
    sender: str = Query(...),
    emoji: str = Query(...),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Remove a reaction."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    await require_agent(sender, token, db)

    msg = await db.get(Message, message_id)
    if not msg or msg.room_id != room.id:
        raise HTTPException(404, "Message not found in this room")

    result = await db.execute(
        select(Reaction).where(and_(
            Reaction.message_id == message_id,
            Reaction.sender == sender,
            Reaction.emoji == emoji,
        ))
    )
    reaction = result.scalar_one_or_none()
    if reaction:
        await db.delete(reaction)
        await db.commit()


# ---------------------------------------------------------------------------
# 15. PUT "/{room_name}/receipts" -- update read receipt
# ---------------------------------------------------------------------------


@router.put("/{room_name}/receipts", response_model=ReceiptOut)
async def update_receipt(
    project_slug: str,
    room_name: str,
    body: ReceiptUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Mark the furthest message an agent has read in a room."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    await require_agent(body.agent, body.token, db)
    presence.touch(body.agent)

    existing = await db.execute(
        select(ReadReceipt).where(and_(
            ReadReceipt.room_id == room.id,
            ReadReceipt.agent == body.agent,
        ))
    )
    receipt = existing.scalar_one_or_none()

    if receipt:
        receipt.last_read = max(receipt.last_read, body.last_read)
        receipt.updated_at = datetime.now(timezone.utc)
    else:
        receipt = ReadReceipt(room_id=room.id, agent=body.agent, last_read=body.last_read)
        db.add(receipt)

    await db.commit()
    await db.refresh(receipt)
    return receipt


# ---------------------------------------------------------------------------
# 16. GET "/{room_name}/receipts" -- get receipts
# ---------------------------------------------------------------------------


@router.get("/{room_name}/receipts", response_model=list[ReceiptOut])
async def get_receipts(
    project_slug: str,
    room_name: str,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Get all read receipts for a room."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    receipts = await db.execute(
        select(ReadReceipt)
        .where(ReadReceipt.room_id == room.id)
        .limit(limit)
    )
    return receipts.scalars().all()


# ---------------------------------------------------------------------------
# 17. POST "/{room_name}/typing" -- set typing indicator
# ---------------------------------------------------------------------------


@router.post("/{room_name}/typing", status_code=204)
async def set_typing(
    project_slug: str,
    room_name: str,
    body: TypingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Signal that an agent is composing a message."""
    project = await require_project(project_slug, db)
    await _require_room(room_name, project.id, db)
    await require_agent(body.sender, body.token, db)
    presence.set_typing(room_name, body.sender)

    broadcaster.publish(room_name, "typing", {
        "agent": body.sender,
        "room": room_name,
    })


# ---------------------------------------------------------------------------
# 18. POST "/{room_name}/join" -- explicit join
# ---------------------------------------------------------------------------


@router.post("/{room_name}/join", response_model=AgentOut, status_code=200)
async def join_room(
    project_slug: str,
    room_name: str,
    sender: str = Query(..., min_length=1, max_length=100),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Join a room explicitly for lurking / read-only participation."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    agent = await require_agent(sender, token, db)
    await ensure_membership(room, agent, db)
    await db.commit()
    return agent


# ---------------------------------------------------------------------------
# 19. GET "/{room_name}/search" -- search messages
# ---------------------------------------------------------------------------


@router.get("/{room_name}/search", response_model=list[MessageOut])
async def search_messages(
    project_slug: str,
    room_name: str,
    q: str = Query(..., min_length=1, description="Search query (case-insensitive substring match)"),
    sender: Optional[str] = Query(None, description="Filter by sender"),
    message_type: Optional[MessageType] = Query(None, description="Filter by message type"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Search messages in a room by content substring match."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)

    stmt = (
        select(Message)
        .where(Message.room_id == room.id)
        .where(Message.content.ilike(f"%{q}%"))
        .options(selectinload(Message.reactions))
    )
    if sender is not None:
        stmt = stmt.where(Message.sender == sender)
    if message_type is not None:
        stmt = stmt.where(Message.message_type == message_type)
    stmt = stmt.order_by(Message.id.desc()).limit(limit)

    result = await db.execute(stmt)
    return [build_message_out(m) for m in result.scalars().all()]


# ---------------------------------------------------------------------------
# 20. POST "/{room_name}/advance-round" -- advance round
# ---------------------------------------------------------------------------


@router.post("/{room_name}/advance-round", response_model=RoomOut)
async def advance_round(
    project_slug: str,
    room_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Advance the discussion to the next round."""
    project = await require_project(project_slug, db)
    room = await _require_room(room_name, project.id, db)
    room.current_round += 1
    await db.commit()
    await db.refresh(room)

    announcement = f"--- Round {room.current_round} ---"
    broadcaster.publish(room_name, "round", {
        "room": room_name,
        "round": room.current_round,
        "announcement": announcement,
    })

    return room

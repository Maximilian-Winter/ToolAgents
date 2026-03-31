"""Business logic extracted from chat route handlers."""

import json as _json

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agora.db.models.chat import Room, Message, Reaction, ReadReceipt, RoomMember
from agora.db.models.agent import Agent
from agora.db.models.enums import MessageType
from agora.schemas.chat import (
    MessageOut,
    ReactionSummary,
    ThreadedMessage,
    ProposalChain,
    RoomSummary,
)


async def ensure_membership(room: Room, agent: Agent, db: AsyncSession) -> None:
    """Add agent to room if not already a member."""
    existing = await db.execute(
        select(RoomMember).where(and_(
            RoomMember.room_id == room.id,
            RoomMember.agent_id == agent.id,
        ))
    )
    if not existing.scalar_one_or_none():
        db.add(RoomMember(room_id=room.id, agent_id=agent.id))


def build_message_out(msg: Message) -> MessageOut:
    """Convert ORM Message to Pydantic output with reaction aggregation and edit_history parsing."""
    reaction_map: dict[str, list[str]] = {}
    for r in msg.reactions:
        reaction_map.setdefault(r.emoji, []).append(r.sender)
    reactions = [
        ReactionSummary(emoji=e, count=len(s), senders=s)
        for e, s in reaction_map.items()
    ]
    # Parse edit_history from JSON string
    history = None
    if msg.edit_history:
        try:
            history = _json.loads(msg.edit_history)
        except (ValueError, TypeError):
            history = None
    return MessageOut(
        id=msg.id,
        room_id=msg.room_id,
        sender=msg.sender,
        content=msg.content,
        message_type=msg.message_type,
        reply_to=msg.reply_to,
        to=msg.to,
        edited_at=msg.edited_at,
        edit_history=history,
        created_at=msg.created_at,
        reactions=reactions,
    )


async def build_threaded_view(
    room_id: int,
    db: AsyncSession,
    since: int | None = None,
    limit: int = 50,
) -> list[ThreadedMessage]:
    """Build thread tree. Returns list of root ThreadedMessage objects."""
    q = (
        select(Message)
        .where(Message.room_id == room_id)
        .options(selectinload(Message.reactions))
        .order_by(Message.id)
    )
    result = await db.execute(q)
    all_msgs = result.scalars().all()

    # Build lookup and tree
    msg_map: dict[int, MessageOut] = {}
    children: dict[int, list[int]] = {}
    roots: list[int] = []

    for m in all_msgs:
        out = build_message_out(m)
        msg_map[m.id] = out
        if m.reply_to and m.reply_to in msg_map:
            children.setdefault(m.reply_to, []).append(m.id)
        else:
            roots.append(m.id)

    # Apply since filter on roots only
    if since is not None:
        roots = [r for r in roots if r > since]
    roots = roots[:limit]

    def _build_thread(msg_id: int) -> ThreadedMessage:
        return ThreadedMessage(
            message=msg_map[msg_id],
            replies=[_build_thread(cid) for cid in children.get(msg_id, [])],
        )

    return [_build_thread(r) for r in roots]


async def build_room_summary(room: Room, db: AsyncSession) -> dict:
    """Build discussion summary. Returns dict matching RoomSummary fields."""
    q = (
        select(Message)
        .where(Message.room_id == room.id)
        .options(selectinload(Message.reactions))
        .order_by(Message.id)
    )
    result = await db.execute(q)
    all_msgs = result.scalars().all()

    msg_map: dict[int, Message] = {m.id: m for m in all_msgs}
    msg_out_map: dict[int, MessageOut] = {m.id: build_message_out(m) for m in all_msgs}

    # Track participants
    participants = list(dict.fromkeys(m.sender for m in all_msgs))

    # Build proposal chains
    proposals: list[ProposalChain] = []
    proposal_ids: set[int] = set()
    claimed_ids: set[int] = set()

    for m in all_msgs:
        if m.message_type == MessageType.proposal:
            proposal_ids.add(m.id)

    for pid in proposal_ids:
        chain = ProposalChain(proposal=msg_out_map[pid])
        claimed_ids.add(pid)

        # Collect reaction totals on the proposal
        for r in msg_out_map[pid].reactions:
            chain.reactions[r.emoji] = r.count

        # Find direct replies to this proposal
        for m in all_msgs:
            if m.reply_to == pid:
                out = msg_out_map[m.id]
                claimed_ids.add(m.id)
                if m.message_type == MessageType.objection:
                    chain.objections.append(out)
                elif m.message_type == MessageType.answer:
                    chain.answers.append(out)
                elif m.message_type == MessageType.consensus:
                    chain.consensus.append(out)
            # Check second-level: replies to objections of this proposal
            if m.reply_to in [o.id for o in chain.objections]:
                if m.id not in claimed_ids:
                    claimed_ids.add(m.id)
                    chain.answers.append(msg_out_map[m.id])

        proposals.append(chain)

    # Open questions: questions with no answer reply
    answered_questions: set[int] = set()
    for m in all_msgs:
        if m.message_type == MessageType.answer and m.reply_to:
            parent = msg_map.get(m.reply_to)
            if parent and parent.message_type == MessageType.question:
                answered_questions.add(m.reply_to)

    open_questions = [
        msg_out_map[m.id] for m in all_msgs
        if m.message_type == MessageType.question and m.id not in answered_questions
    ]

    # Key statements: top-level statements (not replies)
    key_statements = [
        msg_out_map[m.id] for m in all_msgs
        if m.message_type == MessageType.statement and m.reply_to is None
        and m.id not in claimed_ids
    ]

    # Standalone decisions: consensus not already in a proposal chain
    decisions = [
        msg_out_map[m.id] for m in all_msgs
        if m.message_type == MessageType.consensus and m.id not in claimed_ids
    ]

    return dict(
        room=room.name,
        current_round=room.current_round,
        total_messages=len(all_msgs),
        participants=participants,
        proposal_chains=proposals,
        open_questions=open_questions,
        key_statements=key_statements,
        decisions=decisions,
    )

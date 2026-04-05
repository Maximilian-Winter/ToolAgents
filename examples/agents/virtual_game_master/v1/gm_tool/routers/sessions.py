"""Session routes — CRUD and session-scoped notes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models import Campaign, Note, NoteTargetType, Session
from schemas import (
    NoteRead,
    SessionCreate,
    SessionRead,
    SessionReadWithNotes,
    SessionUpdate,
)

router = APIRouter(prefix="/campaigns/{campaign_id}/sessions", tags=["sessions"])


async def _get_campaign_or_404(session: AsyncSession, campaign_id: int):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


async def _get_session_or_404(
    db: AsyncSession, campaign_id: int, session_id: int
) -> Session:
    stmt = select(Session).where(
        Session.id == session_id,
        Session.campaign_id == campaign_id,
    )
    result = await db.execute(stmt)
    sess = result.scalar_one_or_none()
    if not sess:
        raise HTTPException(404, "Session not found")
    return sess


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("/", response_model=SessionRead, status_code=201)
async def create_session(
    campaign_id: int,
    data: SessionCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    sess = Session(campaign_id=campaign_id, **data.model_dump())
    db.add(sess)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            409,
            f"Session #{data.session_number} already exists in this campaign",
        )
    await db.refresh(sess)
    return sess


@router.get("/", response_model=list[SessionRead])
async def list_sessions(
    campaign_id: int,
    status: str | None = Query(default=None, description="Filter by status"),
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    stmt = select(Session).where(Session.campaign_id == campaign_id)
    if status:
        stmt = stmt.where(Session.status == status)
    stmt = stmt.order_by(Session.session_number)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{session_id}", response_model=SessionReadWithNotes)
async def get_session_detail(
    campaign_id: int,
    session_id: int,
    db: AsyncSession = Depends(get_session),
):
    sess = await _get_session_or_404(db, campaign_id, session_id)
    notes = await Note.for_session(db, campaign_id, session_id)
    data = SessionRead.model_validate(sess).model_dump()
    data["notes"] = [NoteRead.model_validate(n).model_dump() for n in notes]
    return data


@router.patch("/{session_id}", response_model=SessionRead)
async def update_session(
    campaign_id: int,
    session_id: int,
    data: SessionUpdate,
    db: AsyncSession = Depends(get_session),
):
    sess = await _get_session_or_404(db, campaign_id, session_id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(sess, key, value)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "Session number conflict")
    await db.refresh(sess)
    return sess


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    campaign_id: int,
    session_id: int,
    db: AsyncSession = Depends(get_session),
):
    sess = await _get_session_or_404(db, campaign_id, session_id)
    await db.delete(sess)
    await db.commit()

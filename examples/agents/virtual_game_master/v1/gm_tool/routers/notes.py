"""Notes routes — CRUD, polymorphic attachment, search."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models import Campaign, Note, NoteTargetType
from schemas import NoteCreate, NoteRead, NoteUpdate

router = APIRouter(prefix="/campaigns/{campaign_id}/notes", tags=["notes"])


async def _get_campaign_or_404(session: AsyncSession, campaign_id: int):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


async def _get_note_or_404(
    db: AsyncSession, campaign_id: int, note_id: int
) -> Note:
    stmt = select(Note).where(
        Note.id == note_id,
        Note.campaign_id == campaign_id,
    )
    result = await db.execute(stmt)
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(404, "Note not found")
    return note


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("/", response_model=NoteRead, status_code=201)
async def create_note(
    campaign_id: int,
    data: NoteCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    note = Note(campaign_id=campaign_id, **data.model_dump())
    db.add(note)
    await db.commit()
    await db.refresh(note)
    return note


@router.get("/", response_model=list[NoteRead])
async def list_notes(
    campaign_id: int,
    target_type: NoteTargetType | None = Query(default=None),
    target_id: int | None = Query(default=None),
    session_id: int | None = Query(default=None, description="Filter by game session"),
    db: AsyncSession = Depends(get_session),
):
    """
    List notes with optional filters.

    - target_type + target_id: notes attached to a specific entity
    - session_id: notes created during a specific game session
    - no filters: all notes in the campaign
    """
    await _get_campaign_or_404(db, campaign_id)

    if target_type and target_id is not None:
        return await Note.for_target(db, campaign_id, target_type, target_id)

    if session_id is not None:
        return await Note.for_session(db, campaign_id, session_id)

    stmt = (
        select(Note)
        .where(Note.campaign_id == campaign_id)
        .order_by(Note.created_at.desc())
    )
    if target_type:
        stmt = stmt.where(Note.target_type == target_type)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/search", response_model=list[NoteRead])
async def search_notes(
    campaign_id: int,
    q: str = Query(..., min_length=1, description="Search term"),
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    return await Note.search(db, campaign_id, q)


@router.get("/{note_id}", response_model=NoteRead)
async def get_note(
    campaign_id: int,
    note_id: int,
    db: AsyncSession = Depends(get_session),
):
    return await _get_note_or_404(db, campaign_id, note_id)


@router.patch("/{note_id}", response_model=NoteRead)
async def update_note(
    campaign_id: int,
    note_id: int,
    data: NoteUpdate,
    db: AsyncSession = Depends(get_session),
):
    note = await _get_note_or_404(db, campaign_id, note_id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(note, key, value)
    await db.commit()
    await db.refresh(note)
    return note


@router.delete("/{note_id}", status_code=204)
async def delete_note(
    campaign_id: int,
    note_id: int,
    db: AsyncSession = Depends(get_session),
):
    note = await _get_note_or_404(db, campaign_id, note_id)
    await db.delete(note)
    await db.commit()

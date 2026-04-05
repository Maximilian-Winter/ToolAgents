"""Tags routes — CRUD, polymorphic tagging, reverse lookups."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models import Campaign, NoteTargetType, Tag, TagAssociation
from schemas import (
    TagAssociationCreate,
    TagAssociationRead,
    TagCreate,
    TagRead,
    TagUpdate,
)

router = APIRouter(prefix="/campaigns/{campaign_id}/tags", tags=["tags"])


async def _get_campaign_or_404(session: AsyncSession, campaign_id: int):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


async def _get_tag_or_404(
    db: AsyncSession, campaign_id: int, tag_id: int
) -> Tag:
    stmt = select(Tag).where(
        Tag.id == tag_id,
        Tag.campaign_id == campaign_id,
    )
    result = await db.execute(stmt)
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(404, "Tag not found")
    return tag


# ---------------------------------------------------------------------------
# Tag CRUD
# ---------------------------------------------------------------------------


@router.post("/", response_model=TagRead, status_code=201)
async def create_tag(
    campaign_id: int,
    data: TagCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    tag = Tag(campaign_id=campaign_id, **data.model_dump())
    db.add(tag)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, f"Tag with slug '{data.slug}' already exists")
    await db.refresh(tag)
    return tag


@router.get("/", response_model=list[TagRead])
async def list_tags(
    campaign_id: int,
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    stmt = (
        select(Tag)
        .where(Tag.campaign_id == campaign_id)
        .order_by(Tag.name)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{tag_id}", response_model=TagRead)
async def get_tag(
    campaign_id: int,
    tag_id: int,
    db: AsyncSession = Depends(get_session),
):
    return await _get_tag_or_404(db, campaign_id, tag_id)


@router.patch("/{tag_id}", response_model=TagRead)
async def update_tag(
    campaign_id: int,
    tag_id: int,
    data: TagUpdate,
    db: AsyncSession = Depends(get_session),
):
    tag = await _get_tag_or_404(db, campaign_id, tag_id)
    updates = data.model_dump(exclude_unset=True)
    if "name" in updates and "slug" not in updates:
        from models import slugify
        updates["slug"] = slugify(updates["name"])
    for key, value in updates.items():
        setattr(tag, key, value)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "Tag slug conflict")
    await db.refresh(tag)
    return tag


@router.delete("/{tag_id}", status_code=204)
async def delete_tag(
    campaign_id: int,
    tag_id: int,
    db: AsyncSession = Depends(get_session),
):
    tag = await _get_tag_or_404(db, campaign_id, tag_id)
    await db.delete(tag)
    await db.commit()


# ---------------------------------------------------------------------------
# Tagging — attach / detach / query
# ---------------------------------------------------------------------------


@router.post(
    "/{tag_id}/apply",
    response_model=TagAssociationRead,
    status_code=201,
)
async def apply_tag(
    campaign_id: int,
    tag_id: int,
    data: TagAssociationCreate,
    db: AsyncSession = Depends(get_session),
):
    """Attach a tag to an entity."""
    await _get_tag_or_404(db, campaign_id, tag_id)
    assoc = TagAssociation(
        tag_id=tag_id,
        target_type=data.target_type,
        target_id=data.target_id,
    )
    db.add(assoc)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "Tag already applied to this entity")
    await db.refresh(assoc)
    return assoc


@router.delete("/{tag_id}/apply/{target_type}/{target_id}", status_code=204)
async def remove_tag(
    campaign_id: int,
    tag_id: int,
    target_type: NoteTargetType,
    target_id: int,
    db: AsyncSession = Depends(get_session),
):
    """Detach a tag from an entity."""
    stmt = select(TagAssociation).where(
        TagAssociation.tag_id == tag_id,
        TagAssociation.target_type == target_type,
        TagAssociation.target_id == target_id,
    )
    result = await db.execute(stmt)
    assoc = result.scalar_one_or_none()
    if not assoc:
        raise HTTPException(404, "Tag is not applied to this entity")
    await db.delete(assoc)
    await db.commit()


@router.get(
    "/{tag_id}/entities",
    response_model=list[TagAssociationRead],
)
async def list_tagged_entities(
    campaign_id: int,
    tag_id: int,
    target_type: NoteTargetType | None = Query(default=None, description="Filter by entity type"),
    db: AsyncSession = Depends(get_session),
):
    """List all entities that have this tag."""
    await _get_tag_or_404(db, campaign_id, tag_id)
    return await TagAssociation.entities_with_tag(db, tag_id, target_type)


@router.get(
    "/on/{target_type}/{target_id}",
    response_model=list[TagRead],
)
async def list_tags_on_entity(
    campaign_id: int,
    target_type: NoteTargetType,
    target_id: int,
    db: AsyncSession = Depends(get_session),
):
    """List all tags on a specific entity."""
    await _get_campaign_or_404(db, campaign_id)
    return await TagAssociation.tags_for(db, target_type, target_id)

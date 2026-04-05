"""
GM Tool — World Lore Router

CRUD and search endpoints for world knowledge articles.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models import Campaign, WorldLore, slugify
from schemas import WorldLoreCreate, WorldLoreRead, WorldLoreUpdate

router = APIRouter(
    prefix="/campaigns/{campaign_id}/world-lore",
    tags=["world-lore"],
)


async def _get_campaign(campaign_id: int, session: AsyncSession) -> Campaign:
    result = await session.execute(
        select(Campaign).where(Campaign.id == campaign_id)
    )
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(404, f"Campaign {campaign_id} not found")
    return campaign


@router.post("/", response_model=WorldLoreRead, status_code=201)
async def create_world_lore(
    campaign_id: int,
    body: WorldLoreCreate,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)

    # Check slug uniqueness
    existing = await session.execute(
        select(WorldLore).where(
            WorldLore.campaign_id == campaign_id,
            WorldLore.slug == body.slug,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"World lore with slug '{body.slug}' already exists")

    lore = WorldLore(
        campaign_id=campaign_id,
        topic=body.topic,
        slug=body.slug or slugify(body.topic),
        category=body.category,
        content=body.content,
        is_secret=body.is_secret,
    )
    session.add(lore)
    await session.commit()
    await session.refresh(lore)
    return lore


@router.get("/", response_model=list[WorldLoreRead])
async def list_world_lore(
    campaign_id: int,
    category: str | None = Query(None, description="Filter by category"),
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)
    stmt = select(WorldLore).where(WorldLore.campaign_id == campaign_id)
    if category:
        stmt = stmt.where(WorldLore.category == category)
    stmt = stmt.order_by(WorldLore.topic)
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.get("/search", response_model=list[WorldLoreRead])
async def search_world_lore(
    campaign_id: int,
    q: str = Query(..., min_length=1, description="Search term"),
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)
    results = await WorldLore.search(session, campaign_id, q)
    return results


@router.get("/{lore_id}", response_model=WorldLoreRead)
async def get_world_lore(
    campaign_id: int,
    lore_id: int,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)
    result = await session.execute(
        select(WorldLore).where(
            WorldLore.id == lore_id,
            WorldLore.campaign_id == campaign_id,
        )
    )
    lore = result.scalar_one_or_none()
    if not lore:
        raise HTTPException(404, f"World lore {lore_id} not found")
    return lore


@router.get("/by-slug/{slug}", response_model=WorldLoreRead)
async def get_world_lore_by_slug(
    campaign_id: int,
    slug: str,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)
    result = await session.execute(
        select(WorldLore).where(
            WorldLore.campaign_id == campaign_id,
            WorldLore.slug == slug,
        )
    )
    lore = result.scalar_one_or_none()
    if not lore:
        raise HTTPException(404, f"World lore with slug '{slug}' not found")
    return lore


@router.patch("/{lore_id}", response_model=WorldLoreRead)
async def update_world_lore(
    campaign_id: int,
    lore_id: int,
    body: WorldLoreUpdate,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)
    result = await session.execute(
        select(WorldLore).where(
            WorldLore.id == lore_id,
            WorldLore.campaign_id == campaign_id,
        )
    )
    lore = result.scalar_one_or_none()
    if not lore:
        raise HTTPException(404, f"World lore {lore_id} not found")

    updates = body.model_dump(exclude_unset=True)

    # If topic changed and no explicit slug, regenerate slug
    if "topic" in updates and "slug" not in updates:
        updates["slug"] = slugify(updates["topic"])

    # Check slug uniqueness if slug is changing
    new_slug = updates.get("slug")
    if new_slug and new_slug != lore.slug:
        existing = await session.execute(
            select(WorldLore).where(
                WorldLore.campaign_id == campaign_id,
                WorldLore.slug == new_slug,
                WorldLore.id != lore_id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(409, f"World lore with slug '{new_slug}' already exists")

    for key, value in updates.items():
        setattr(lore, key, value)

    await session.commit()
    await session.refresh(lore)
    return lore


@router.delete("/{lore_id}", status_code=204)
async def delete_world_lore(
    campaign_id: int,
    lore_id: int,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign(campaign_id, session)
    result = await session.execute(
        select(WorldLore).where(
            WorldLore.id == lore_id,
            WorldLore.campaign_id == campaign_id,
        )
    )
    lore = result.scalar_one_or_none()
    if not lore:
        raise HTTPException(404, f"World lore {lore_id} not found")
    await session.delete(lore)
    await session.commit()

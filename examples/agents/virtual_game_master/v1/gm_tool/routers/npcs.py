"""NPC routes — CRUD, search, location associations, relationships."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import get_session
from models import NPC, NPCLocationAssociation, NPCRelationship
from schemas import (
    NPCCreate,
    NPCLocationAssociationCreate,
    NPCLocationAssociationRead,
    NPCRead,
    NPCReadFull,
    NPCRelationshipCreate,
    NPCRelationshipRead,
    NPCUpdate,
)

router = APIRouter(prefix="/campaigns/{campaign_id}/npcs", tags=["npcs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_campaign_or_404(session: AsyncSession, campaign_id: int):
    from models import Campaign

    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


async def _get_npc_or_404(
    session: AsyncSession, campaign_id: int, npc_id: int
) -> NPC:
    stmt = select(NPC).where(
        NPC.id == npc_id,
        NPC.campaign_id == campaign_id,
    )
    result = await session.execute(stmt)
    npc = result.scalar_one_or_none()
    if not npc:
        raise HTTPException(404, "NPC not found")
    return npc


async def _load_npc_full(session: AsyncSession, campaign_id: int, npc_id: int) -> NPC:
    """Load an NPC with all associations and relationships eagerly loaded."""
    stmt = (
        select(NPC)
        .where(NPC.id == npc_id, NPC.campaign_id == campaign_id)
        .options(
            selectinload(NPC.location_associations),
            selectinload(NPC.relationships_out),
            selectinload(NPC.relationships_in),
        )
    )
    result = await session.execute(stmt)
    npc = result.scalar_one_or_none()
    if not npc:
        raise HTTPException(404, "NPC not found")
    return npc


def _npc_to_full(npc: NPC) -> dict:
    """Serialize NPC with merged relationships from both directions."""
    data = NPCReadFull.model_validate(npc).model_dump()
    data["relationships"] = [
        NPCRelationshipRead.model_validate(r).model_dump()
        for r in npc.relationships_out + npc.relationships_in
    ]
    data["location_associations"] = [
        NPCLocationAssociationRead.model_validate(a).model_dump()
        for a in npc.location_associations
    ]
    return data


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("/", response_model=NPCRead, status_code=201)
async def create_npc(
    campaign_id: int,
    data: NPCCreate,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(session, campaign_id)
    npc = NPC(campaign_id=campaign_id, **data.model_dump())
    session.add(npc)
    await session.commit()
    await session.refresh(npc)
    return npc


@router.get("/", response_model=list[NPCRead])
async def list_npcs(
    campaign_id: int,
    status: str | None = Query(default=None, description="Filter by status"),
    location_id: int | None = Query(default=None, description="Filter by associated location"),
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(session, campaign_id)
    stmt = select(NPC).where(NPC.campaign_id == campaign_id)

    if status:
        stmt = stmt.where(NPC.status == status)

    if location_id is not None:
        stmt = stmt.join(NPCLocationAssociation).where(
            NPCLocationAssociation.location_id == location_id
        )

    stmt = stmt.order_by(NPC.name)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/search", response_model=list[NPCRead])
async def search_npcs(
    campaign_id: int,
    q: str = Query(..., min_length=1, description="Search term"),
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(session, campaign_id)
    npcs = await NPC.search(session, campaign_id, q)
    return npcs


@router.get("/{npc_id}", response_model=NPCReadFull)
async def get_npc(
    campaign_id: int,
    npc_id: int,
    session: AsyncSession = Depends(get_session),
):
    npc = await _load_npc_full(session, campaign_id, npc_id)
    return _npc_to_full(npc)


@router.patch("/{npc_id}", response_model=NPCRead)
async def update_npc(
    campaign_id: int,
    npc_id: int,
    data: NPCUpdate,
    session: AsyncSession = Depends(get_session),
):
    npc = await _get_npc_or_404(session, campaign_id, npc_id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(npc, key, value)
    await session.commit()
    await session.refresh(npc)
    return npc


@router.delete("/{npc_id}", status_code=204)
async def delete_npc(
    campaign_id: int,
    npc_id: int,
    session: AsyncSession = Depends(get_session),
):
    npc = await _get_npc_or_404(session, campaign_id, npc_id)
    await session.delete(npc)
    await session.commit()


# ---------------------------------------------------------------------------
# NPC ↔ Location associations
# ---------------------------------------------------------------------------


@router.post(
    "/{npc_id}/locations",
    response_model=NPCLocationAssociationRead,
    status_code=201,
)
async def add_npc_location(
    campaign_id: int,
    npc_id: int,
    data: NPCLocationAssociationCreate,
    session: AsyncSession = Depends(get_session),
):
    await _get_npc_or_404(session, campaign_id, npc_id)
    assoc = NPCLocationAssociation(**data.model_dump())
    session.add(assoc)
    await session.commit()
    await session.refresh(assoc)
    return assoc


@router.get(
    "/{npc_id}/locations",
    response_model=list[NPCLocationAssociationRead],
)
async def list_npc_locations(
    campaign_id: int,
    npc_id: int,
    session: AsyncSession = Depends(get_session),
):
    await _get_npc_or_404(session, campaign_id, npc_id)
    stmt = select(NPCLocationAssociation).where(
        NPCLocationAssociation.npc_id == npc_id
    )
    result = await session.execute(stmt)
    return result.scalars().all()


@router.delete("/{npc_id}/locations/{association_id}", status_code=204)
async def remove_npc_location(
    campaign_id: int,
    npc_id: int,
    association_id: int,
    session: AsyncSession = Depends(get_session),
):
    assoc = await session.get(NPCLocationAssociation, association_id)
    if not assoc or assoc.npc_id != npc_id:
        raise HTTPException(404, "Association not found")
    await session.delete(assoc)
    await session.commit()


# ---------------------------------------------------------------------------
# NPC ↔ NPC relationships
# ---------------------------------------------------------------------------


@router.post(
    "/{npc_id}/relationships",
    response_model=NPCRelationshipRead,
    status_code=201,
)
async def add_npc_relationship(
    campaign_id: int,
    npc_id: int,
    data: NPCRelationshipCreate,
    session: AsyncSession = Depends(get_session),
):
    # verify both NPCs belong to this campaign
    await _get_npc_or_404(session, campaign_id, data.from_npc_id)
    await _get_npc_or_404(session, campaign_id, data.to_npc_id)

    if data.from_npc_id == data.to_npc_id:
        raise HTTPException(400, "An NPC cannot have a relationship with itself")

    rel = NPCRelationship(**data.model_dump())
    session.add(rel)
    await session.commit()
    await session.refresh(rel)
    return rel


@router.get(
    "/{npc_id}/relationships",
    response_model=list[NPCRelationshipRead],
)
async def list_npc_relationships(
    campaign_id: int,
    npc_id: int,
    session: AsyncSession = Depends(get_session),
):
    await _get_npc_or_404(session, campaign_id, npc_id)
    stmt = select(NPCRelationship).where(
        or_(
            NPCRelationship.from_npc_id == npc_id,
            NPCRelationship.to_npc_id == npc_id,
        )
    )
    result = await session.execute(stmt)
    return result.scalars().all()


@router.delete("/{npc_id}/relationships/{relationship_id}", status_code=204)
async def remove_npc_relationship(
    campaign_id: int,
    npc_id: int,
    relationship_id: int,
    session: AsyncSession = Depends(get_session),
):
    rel = await session.get(NPCRelationship, relationship_id)
    if not rel:
        raise HTTPException(404, "Relationship not found")
    if rel.from_npc_id != npc_id and rel.to_npc_id != npc_id:
        raise HTTPException(404, "Relationship not found for this NPC")
    await session.delete(rel)
    await session.commit()

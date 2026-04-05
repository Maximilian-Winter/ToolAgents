"""Player character routes — PCs, companions, inventory, special items, relationships."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import get_session
from models import (
    Campaign,
    Companion,
    InventoryItem,
    PCRelationship,
    PlayerCharacter,
    SpecialItem,
)
from schemas import (
    CompanionCreate,
    CompanionRead,
    CompanionUpdate,
    InventoryItemCreate,
    InventoryItemRead,
    InventoryItemUpdate,
    PCRelationshipCreate,
    PCRelationshipRead,
    PlayerCharacterCreate,
    PlayerCharacterFull,
    PlayerCharacterRead,
    PlayerCharacterUpdate,
    SpecialItemCreate,
    SpecialItemRead,
    SpecialItemUpdate,
)

router = APIRouter(
    prefix="/campaigns/{campaign_id}/player-characters",
    tags=["player characters"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_campaign_or_404(session: AsyncSession, campaign_id: int):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


async def _get_pc_or_404(
    db: AsyncSession, campaign_id: int, pc_id: int
) -> PlayerCharacter:
    stmt = select(PlayerCharacter).where(
        PlayerCharacter.id == pc_id,
        PlayerCharacter.campaign_id == campaign_id,
    )
    result = await db.execute(stmt)
    pc = result.scalar_one_or_none()
    if not pc:
        raise HTTPException(404, "Player character not found")
    return pc


async def _load_pc_full(
    db: AsyncSession, campaign_id: int, pc_id: int
) -> PlayerCharacter:
    stmt = (
        select(PlayerCharacter)
        .where(
            PlayerCharacter.id == pc_id,
            PlayerCharacter.campaign_id == campaign_id,
        )
        .options(
            selectinload(PlayerCharacter.companions),
            selectinload(PlayerCharacter.inventory_items),
            selectinload(PlayerCharacter.special_items),
            selectinload(PlayerCharacter.pc_relationships),
        )
    )
    result = await db.execute(stmt)
    pc = result.scalar_one_or_none()
    if not pc:
        raise HTTPException(404, "Player character not found")
    return pc


def _pc_to_full(pc: PlayerCharacter) -> dict:
    data = PlayerCharacterRead.model_validate(pc).model_dump()
    data["companions"] = [
        CompanionRead.model_validate(c).model_dump() for c in pc.companions
    ]
    data["inventory_items"] = [
        InventoryItemRead.model_validate(i).model_dump() for i in pc.inventory_items
    ]
    data["special_items"] = [
        SpecialItemRead.model_validate(s).model_dump() for s in pc.special_items
    ]
    data["relationships"] = [
        PCRelationshipRead.model_validate(r).model_dump() for r in pc.pc_relationships
    ]
    return data


# ---------------------------------------------------------------------------
# PC CRUD
# ---------------------------------------------------------------------------


@router.post("/", response_model=PlayerCharacterRead, status_code=201)
async def create_player_character(
    campaign_id: int,
    data: PlayerCharacterCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    pc = PlayerCharacter(campaign_id=campaign_id, **data.model_dump())
    db.add(pc)
    await db.commit()
    await db.refresh(pc)
    return pc


@router.get("/", response_model=list[PlayerCharacterRead])
async def list_player_characters(
    campaign_id: int,
    active_only: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    stmt = select(PlayerCharacter).where(
        PlayerCharacter.campaign_id == campaign_id
    )
    if active_only:
        stmt = stmt.where(PlayerCharacter.is_active.is_(True))
    stmt = stmt.order_by(PlayerCharacter.name)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/search", response_model=list[PlayerCharacterRead])
async def search_player_characters(
    campaign_id: int,
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(db, campaign_id)
    return await PlayerCharacter.search(db, campaign_id, q)


@router.get("/{pc_id}", response_model=PlayerCharacterFull)
async def get_player_character(
    campaign_id: int,
    pc_id: int,
    db: AsyncSession = Depends(get_session),
):
    pc = await _load_pc_full(db, campaign_id, pc_id)
    return _pc_to_full(pc)


@router.patch("/{pc_id}", response_model=PlayerCharacterRead)
async def update_player_character(
    campaign_id: int,
    pc_id: int,
    data: PlayerCharacterUpdate,
    db: AsyncSession = Depends(get_session),
):
    pc = await _get_pc_or_404(db, campaign_id, pc_id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(pc, key, value)
    await db.commit()
    await db.refresh(pc)
    return pc


@router.delete("/{pc_id}", status_code=204)
async def delete_player_character(
    campaign_id: int,
    pc_id: int,
    db: AsyncSession = Depends(get_session),
):
    pc = await _get_pc_or_404(db, campaign_id, pc_id)
    await db.delete(pc)
    await db.commit()


# ---------------------------------------------------------------------------
# Companions
# ---------------------------------------------------------------------------


@router.post(
    "/{pc_id}/companions",
    response_model=CompanionRead,
    status_code=201,
)
async def add_companion(
    campaign_id: int,
    pc_id: int,
    data: CompanionCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    companion = Companion(player_character_id=pc_id, **data.model_dump())
    db.add(companion)
    await db.commit()
    await db.refresh(companion)
    return companion


@router.get("/{pc_id}/companions", response_model=list[CompanionRead])
async def list_companions(
    campaign_id: int,
    pc_id: int,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    stmt = (
        select(Companion)
        .where(Companion.player_character_id == pc_id)
        .order_by(Companion.name)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.patch(
    "/{pc_id}/companions/{companion_id}",
    response_model=CompanionRead,
)
async def update_companion(
    campaign_id: int,
    pc_id: int,
    companion_id: int,
    data: CompanionUpdate,
    db: AsyncSession = Depends(get_session),
):
    companion = await db.get(Companion, companion_id)
    if not companion or companion.player_character_id != pc_id:
        raise HTTPException(404, "Companion not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(companion, key, value)
    await db.commit()
    await db.refresh(companion)
    return companion


@router.delete("/{pc_id}/companions/{companion_id}", status_code=204)
async def remove_companion(
    campaign_id: int,
    pc_id: int,
    companion_id: int,
    db: AsyncSession = Depends(get_session),
):
    companion = await db.get(Companion, companion_id)
    if not companion or companion.player_character_id != pc_id:
        raise HTTPException(404, "Companion not found")
    await db.delete(companion)
    await db.commit()


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


@router.post(
    "/{pc_id}/inventory",
    response_model=InventoryItemRead,
    status_code=201,
)
async def add_inventory_item(
    campaign_id: int,
    pc_id: int,
    data: InventoryItemCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    item = InventoryItem(player_character_id=pc_id, **data.model_dump())
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.get("/{pc_id}/inventory", response_model=list[InventoryItemRead])
async def list_inventory(
    campaign_id: int,
    pc_id: int,
    category: str | None = Query(default=None),
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    stmt = select(InventoryItem).where(
        InventoryItem.player_character_id == pc_id
    )
    if category:
        stmt = stmt.where(InventoryItem.category == category)
    stmt = stmt.order_by(InventoryItem.name)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.patch(
    "/{pc_id}/inventory/{item_id}",
    response_model=InventoryItemRead,
)
async def update_inventory_item(
    campaign_id: int,
    pc_id: int,
    item_id: int,
    data: InventoryItemUpdate,
    db: AsyncSession = Depends(get_session),
):
    item = await db.get(InventoryItem, item_id)
    if not item or item.player_character_id != pc_id:
        raise HTTPException(404, "Inventory item not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{pc_id}/inventory/{item_id}", status_code=204)
async def remove_inventory_item(
    campaign_id: int,
    pc_id: int,
    item_id: int,
    db: AsyncSession = Depends(get_session),
):
    item = await db.get(InventoryItem, item_id)
    if not item or item.player_character_id != pc_id:
        raise HTTPException(404, "Inventory item not found")
    await db.delete(item)
    await db.commit()


# ---------------------------------------------------------------------------
# Special Items
# ---------------------------------------------------------------------------


@router.post(
    "/{pc_id}/special-items",
    response_model=SpecialItemRead,
    status_code=201,
)
async def add_special_item(
    campaign_id: int,
    pc_id: int,
    data: SpecialItemCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    item = SpecialItem(
        player_character_id=pc_id,
        campaign_id=campaign_id,
        **data.model_dump(),
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.get("/{pc_id}/special-items", response_model=list[SpecialItemRead])
async def list_special_items(
    campaign_id: int,
    pc_id: int,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    stmt = (
        select(SpecialItem)
        .where(SpecialItem.player_character_id == pc_id)
        .order_by(SpecialItem.name)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.patch(
    "/{pc_id}/special-items/{item_id}",
    response_model=SpecialItemRead,
)
async def update_special_item(
    campaign_id: int,
    pc_id: int,
    item_id: int,
    data: SpecialItemUpdate,
    db: AsyncSession = Depends(get_session),
):
    item = await db.get(SpecialItem, item_id)
    if not item or item.player_character_id != pc_id:
        raise HTTPException(404, "Special item not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{pc_id}/special-items/{item_id}", status_code=204)
async def remove_special_item(
    campaign_id: int,
    pc_id: int,
    item_id: int,
    db: AsyncSession = Depends(get_session),
):
    item = await db.get(SpecialItem, item_id)
    if not item or item.player_character_id != pc_id:
        raise HTTPException(404, "Special item not found")
    await db.delete(item)
    await db.commit()


# ---------------------------------------------------------------------------
# PC Relationships
# ---------------------------------------------------------------------------


@router.post(
    "/{pc_id}/relationships",
    response_model=PCRelationshipRead,
    status_code=201,
)
async def add_pc_relationship(
    campaign_id: int,
    pc_id: int,
    data: PCRelationshipCreate,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    rel = PCRelationship(player_character_id=pc_id, **data.model_dump())
    db.add(rel)
    await db.commit()
    await db.refresh(rel)
    return rel


@router.get(
    "/{pc_id}/relationships",
    response_model=list[PCRelationshipRead],
)
async def list_pc_relationships(
    campaign_id: int,
    pc_id: int,
    db: AsyncSession = Depends(get_session),
):
    await _get_pc_or_404(db, campaign_id, pc_id)
    stmt = select(PCRelationship).where(
        PCRelationship.player_character_id == pc_id
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.delete("/{pc_id}/relationships/{rel_id}", status_code=204)
async def remove_pc_relationship(
    campaign_id: int,
    pc_id: int,
    rel_id: int,
    db: AsyncSession = Depends(get_session),
):
    rel = await db.get(PCRelationship, rel_id)
    if not rel or rel.player_character_id != pc_id:
        raise HTTPException(404, "Relationship not found")
    await db.delete(rel)
    await db.commit()

"""Location routes — CRUD, path resolution, tree, search, connections."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import get_session
from models import Location, LocationConnection, NPCLocationAssociation
from schemas import (
    LocationConnectionCreate,
    LocationConnectionRead,
    LocationCreate,
    LocationRead,
    LocationReadWithPath,
    LocationTree,
    LocationUpdate,
    NPCSummary,
)

router = APIRouter(prefix="/campaigns/{campaign_id}/locations", tags=["locations"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_campaign_or_404(session: AsyncSession, campaign_id: int):
    from models import Campaign

    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


async def _get_location_or_404(
    session: AsyncSession, campaign_id: int, location_id: int
) -> Location:
    stmt = select(Location).where(
        Location.id == location_id,
        Location.campaign_id == campaign_id,
    )
    result = await session.execute(stmt)
    location = result.scalar_one_or_none()
    if not location:
        raise HTTPException(404, "Location not found")
    return location


async def _enrich_with_path(session: AsyncSession, location: Location) -> dict:
    """Convert a Location to a dict with its computed path."""
    path = await Location.get_location_path(session, location.id)
    data = LocationReadWithPath.model_validate(location).model_dump()
    data["path"] = path
    return data


async def _build_tree(
    session: AsyncSession,
    campaign_id: int,
    parent_id: int | None,
) -> list[dict]:
    """Recursively build a location tree with NPC summaries."""
    stmt = (
        select(Location)
        .where(
            Location.campaign_id == campaign_id,
            Location.parent_id == parent_id
            if parent_id is not None
            else Location.parent_id.is_(None),
        )
        .options(selectinload(Location.npc_associations).selectinload(NPCLocationAssociation.npc))
        .order_by(Location.name)
    )
    result = await session.execute(stmt)
    locations = result.scalars().all()

    tree = []
    for loc in locations:
        path = await Location.get_location_path(session, loc.id)
        npcs = [
            NPCSummary.model_validate(assoc.npc).model_dump()
            for assoc in loc.npc_associations
        ]
        children = await _build_tree(session, campaign_id, loc.id)
        # Build from LocationRead first to avoid touching lazy-loaded relations
        node = LocationRead.model_validate(loc).model_dump()
        node["path"] = path
        node["children"] = children
        node["npcs"] = npcs
        tree.append(node)
    return tree


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("/", response_model=LocationReadWithPath, status_code=201)
async def create_location(
    campaign_id: int,
    data: LocationCreate,
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(session, campaign_id)

    # validate parent belongs to same campaign
    if data.parent_id is not None:
        await _get_location_or_404(session, campaign_id, data.parent_id)

    # check slug uniqueness among siblings
    parent_filter = (
        Location.parent_id == data.parent_id
        if data.parent_id is not None
        else Location.parent_id.is_(None)
    )
    stmt = select(Location).where(
        Location.campaign_id == campaign_id,
        Location.slug == data.slug,
        parent_filter,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            409,
            f"A sibling location with slug '{data.slug}' already exists",
        )

    location = Location(campaign_id=campaign_id, **data.model_dump())
    session.add(location)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            409,
            f"A sibling location with slug '{data.slug}' already exists",
        )
    await session.refresh(location)
    return await _enrich_with_path(session, location)


@router.get("/", response_model=list[LocationReadWithPath])
async def list_locations(
    campaign_id: int,
    parent_id: int | None = Query(default=None, description="Filter by parent (use 'root' for top-level)"),
    root_only: bool = Query(default=False, description="Only return root locations"),
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(session, campaign_id)
    stmt = select(Location).where(Location.campaign_id == campaign_id)

    if root_only:
        stmt = stmt.where(Location.parent_id.is_(None))
    elif parent_id is not None:
        stmt = stmt.where(Location.parent_id == parent_id)

    stmt = stmt.order_by(Location.name)
    result = await session.execute(stmt)
    locations = result.scalars().all()

    return [await _enrich_with_path(session, loc) for loc in locations]


@router.get("/resolve", response_model=LocationReadWithPath)
async def resolve_path(
    campaign_id: int,
    path: str = Query(..., description="Slash-separated slug path, e.g. waterdeep/castle-ward"),
    session: AsyncSession = Depends(get_session),
):
    """Resolve a slash-separated slug path to a location."""
    await _get_campaign_or_404(session, campaign_id)
    location = await Location.resolve_path(session, campaign_id, path)
    if not location:
        raise HTTPException(404, f"No location found at path '{path}'")
    return await _enrich_with_path(session, location)


@router.get("/tree", response_model=list[LocationTree])
async def get_location_tree(
    campaign_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Return the full location hierarchy as a nested tree with NPC summaries."""
    await _get_campaign_or_404(session, campaign_id)
    return await _build_tree(session, campaign_id, parent_id=None)


@router.get("/search", response_model=list[LocationReadWithPath])
async def search_locations(
    campaign_id: int,
    q: str = Query(..., min_length=1, description="Search term"),
    session: AsyncSession = Depends(get_session),
):
    await _get_campaign_or_404(session, campaign_id)
    locations = await Location.search(session, campaign_id, q)
    return [await _enrich_with_path(session, loc) for loc in locations]


@router.get("/{location_id}", response_model=LocationReadWithPath)
async def get_location(
    campaign_id: int,
    location_id: int,
    session: AsyncSession = Depends(get_session),
):
    location = await _get_location_or_404(session, campaign_id, location_id)
    return await _enrich_with_path(session, location)


@router.get("/{location_id}/children", response_model=list[LocationReadWithPath])
async def list_children(
    campaign_id: int,
    location_id: int,
    session: AsyncSession = Depends(get_session),
):
    await _get_location_or_404(session, campaign_id, location_id)
    stmt = (
        select(Location)
        .where(
            Location.campaign_id == campaign_id,
            Location.parent_id == location_id,
        )
        .order_by(Location.name)
    )
    result = await session.execute(stmt)
    children = result.scalars().all()
    return [await _enrich_with_path(session, loc) for loc in children]


@router.patch("/{location_id}", response_model=LocationReadWithPath)
async def update_location(
    campaign_id: int,
    location_id: int,
    data: LocationUpdate,
    session: AsyncSession = Depends(get_session),
):
    location = await _get_location_or_404(session, campaign_id, location_id)
    updates = data.model_dump(exclude_unset=True)

    # if name changes and no explicit slug, regenerate slug
    if "name" in updates and "slug" not in updates:
        from models import slugify

        updates["slug"] = slugify(updates["name"])

    # validate new parent if changing
    if "parent_id" in updates and updates["parent_id"] is not None:
        if updates["parent_id"] == location_id:
            raise HTTPException(400, "A location cannot be its own parent")
        await _get_location_or_404(session, campaign_id, updates["parent_id"])

    for key, value in updates.items():
        setattr(location, key, value)

    await session.commit()
    await session.refresh(location)
    return await _enrich_with_path(session, location)


@router.delete("/{location_id}", status_code=204)
async def delete_location(
    campaign_id: int,
    location_id: int,
    session: AsyncSession = Depends(get_session),
):
    location = await _get_location_or_404(session, campaign_id, location_id)
    await session.delete(location)
    await session.commit()


# ---------------------------------------------------------------------------
# Connections (lateral links between locations)
# ---------------------------------------------------------------------------


@router.post(
    "/{location_id}/connections",
    response_model=LocationConnectionRead,
    status_code=201,
)
async def create_connection(
    campaign_id: int,
    location_id: int,
    data: LocationConnectionCreate,
    session: AsyncSession = Depends(get_session),
):
    # ensure both locations exist in this campaign
    loc_from = await _get_location_or_404(session, campaign_id, data.from_location_id)
    loc_to = await _get_location_or_404(session, campaign_id, data.to_location_id)

    if loc_from.id == loc_to.id:
        raise HTTPException(400, "Cannot connect a location to itself")

    conn = LocationConnection(**data.model_dump())
    session.add(conn)
    await session.commit()
    await session.refresh(conn)
    return conn


@router.get(
    "/{location_id}/connections",
    response_model=list[LocationConnectionRead],
)
async def list_connections(
    campaign_id: int,
    location_id: int,
    session: AsyncSession = Depends(get_session),
):
    await _get_location_or_404(session, campaign_id, location_id)
    stmt = select(LocationConnection).where(
        (LocationConnection.from_location_id == location_id)
        | (LocationConnection.to_location_id == location_id)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


@router.delete("/connections/{connection_id}", status_code=204)
async def delete_connection(
    campaign_id: int,
    connection_id: int,
    session: AsyncSession = Depends(get_session),
):
    conn = await session.get(LocationConnection, connection_id)
    if not conn:
        raise HTTPException(404, "Connection not found")
    await session.delete(conn)
    await session.commit()

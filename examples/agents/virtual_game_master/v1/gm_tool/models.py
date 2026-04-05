"""
GM Tool — SQLAlchemy Models

Covers: Campaigns, Locations (hierarchical with slug paths),
NPCs, NPC-Location associations, NPC-NPC relationships,
and Location-Location connections.

Uses async SQLAlchemy 2.0 style with aiosqlite.
"""

from __future__ import annotations

import enum
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Enum,
    ForeignKey,
    Index,
    Text,
    UniqueConstraint,
    event,
    select,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(AsyncAttrs, DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LocationType(str, enum.Enum):
    REGION = "region"
    CITY = "city"
    DISTRICT = "district"
    BUILDING = "building"
    ROOM = "room"
    LANDMARK = "landmark"
    WILDERNESS = "wilderness"
    OTHER = "other"


class NPCStatus(str, enum.Enum):
    ALIVE = "alive"
    DEAD = "dead"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ConnectionType(str, enum.Enum):
    ROAD = "road"
    RIVER = "river"
    TUNNEL = "tunnel"
    PORTAL = "portal"
    SEA_ROUTE = "sea_route"
    PATH = "path"
    SECRET = "secret"
    OTHER = "other"


class NPCRelationshipType(str, enum.Enum):
    ALLY = "ally"
    RIVAL = "rival"
    ENEMY = "enemy"
    FAMILY = "family"
    FRIEND = "friend"
    EMPLOYER = "employer"
    EMPLOYEE = "employee"
    MENTOR = "mentor"
    STUDENT = "student"
    LOVER = "lover"
    CONTACT = "contact"
    OTHER = "other"


class SessionStatus(str, enum.Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class NoteTargetType(str, enum.Enum):
    """Which entity type a note or tag is attached to."""
    CAMPAIGN = "campaign"
    SESSION = "session"
    LOCATION = "location"
    NPC = "npc"
    PLAYER_CHARACTER = "player_character"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Convert a display name to a URL/path-friendly slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(index=True)
    game_system: Mapped[str] = mapped_column(default="generic")
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationships
    locations: Mapped[list[Location]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
    )
    npcs: Mapped[list[NPC]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list[Session]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
    )
    tags: Mapped[list[Tag]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
    )
    player_characters: Mapped[list[PlayerCharacter]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Campaign {self.id}: {self.name!r}>"


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class Location(Base):
    __tablename__ = "locations"
    __table_args__ = (
        UniqueConstraint("campaign_id", "parent_id", "slug", name="uq_location_slug"),
        Index("ix_location_campaign_parent", "campaign_id", "parent_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("locations.id"), default=None
    )

    name: Mapped[str]
    slug: Mapped[str] = mapped_column(index=True)
    location_type: Mapped[LocationType] = mapped_column(
        Enum(LocationType), default=LocationType.OTHER
    )
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    secrets: Mapped[Optional[str]] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationships
    campaign: Mapped[Campaign] = relationship(back_populates="locations")
    parent: Mapped[Optional[Location]] = relationship(
        back_populates="children",
        remote_side="Location.id",
    )
    children: Mapped[list[Location]] = relationship(
        back_populates="parent",
        cascade="all, delete-orphan",
    )
    npc_associations: Mapped[list[NPCLocationAssociation]] = relationship(
        back_populates="location",
        cascade="all, delete-orphan",
    )

    # lateral connections (both directions)
    connections_from: Mapped[list[LocationConnection]] = relationship(
        foreign_keys="LocationConnection.from_location_id",
        back_populates="from_location",
        cascade="all, delete-orphan",
    )
    connections_to: Mapped[list[LocationConnection]] = relationship(
        foreign_keys="LocationConnection.to_location_id",
        back_populates="to_location",
        cascade="all, delete-orphan",
    )

    @property
    def connections(self) -> list[LocationConnection]:
        """All connections regardless of direction."""
        return self.connections_from + self.connections_to

    @property
    def npcs(self) -> list[NPC]:
        """Convenience: all NPCs at this location."""
        return [assoc.npc for assoc in self.npc_associations]

    def auto_slug(self) -> None:
        """Set slug from name if not already set."""
        if not self.slug:
            self.slug = slugify(self.name)

    # --- path helpers ---

    def get_path(self) -> str:
        """
        Build the full slash-separated path by walking up the parent chain.

        NOTE: requires that parent relationships are eagerly loaded.
        For async contexts, use the standalone `get_location_path` helper instead.
        """
        parts: list[str] = []
        current: Optional[Location] = self
        while current is not None:
            parts.append(current.slug)
            current = current.parent
        return "/".join(reversed(parts))

    @staticmethod
    async def resolve_path(
        session: AsyncSession,
        campaign_id: int,
        path: str,
    ) -> Optional[Location]:
        """
        Resolve a slash-separated slug path to a Location.

        Example:
            waterdeep/castle-ward/blackstaff-tower
        """
        slugs = [s for s in path.strip("/").split("/") if s]
        if not slugs:
            return None

        current: Optional[Location] = None
        for slug in slugs:
            parent_id = current.id if current else None
            stmt = select(Location).where(
                Location.campaign_id == campaign_id,
                Location.slug == slug,
                Location.parent_id == parent_id
                if parent_id is not None
                else Location.parent_id.is_(None),
            )
            result = await session.execute(stmt)
            current = result.scalar_one_or_none()
            if current is None:
                return None
        return current

    @staticmethod
    async def get_location_path(session: AsyncSession, location_id: int) -> str:
        """
        Async-safe path builder — loads parents one by one.
        """
        parts: list[str] = []
        current_id: Optional[int] = location_id

        while current_id is not None:
            stmt = select(Location).where(Location.id == current_id)
            result = await session.execute(stmt)
            loc = result.scalar_one_or_none()
            if loc is None:
                break
            parts.append(loc.slug)
            current_id = loc.parent_id

        return "/".join(reversed(parts))

    @staticmethod
    async def search(
        session: AsyncSession,
        campaign_id: int,
        query: str,
    ) -> list[Location]:
        """Search locations by name (case-insensitive partial match)."""
        stmt = (
            select(Location)
            .where(
                Location.campaign_id == campaign_id,
                Location.name.ilike(f"%{query}%"),
            )
            .order_by(Location.name)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    def __repr__(self) -> str:
        return f"<Location {self.id}: {self.name!r}>"


# auto-generate slug from name on insert
@event.listens_for(Location, "before_insert")
def _location_before_insert(_mapper, _connection, target: Location):
    if not target.slug and target.name:
        target.slug = slugify(target.name)


# ---------------------------------------------------------------------------
# LocationConnection (lateral links between locations)
# ---------------------------------------------------------------------------


class LocationConnection(Base):
    __tablename__ = "location_connections"
    __table_args__ = (
        UniqueConstraint(
            "from_location_id", "to_location_id", name="uq_location_connection"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    from_location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    to_location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    connection_type: Mapped[ConnectionType] = mapped_column(
        Enum(ConnectionType), default=ConnectionType.OTHER
    )
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    is_bidirectional: Mapped[bool] = mapped_column(default=True)
    is_secret: Mapped[bool] = mapped_column(default=False)

    # relationships
    from_location: Mapped[Location] = relationship(
        foreign_keys=[from_location_id], back_populates="connections_from"
    )
    to_location: Mapped[Location] = relationship(
        foreign_keys=[to_location_id], back_populates="connections_to"
    )

    def __repr__(self) -> str:
        arrow = "<->" if self.is_bidirectional else "->"
        return (
            f"<Connection {self.from_location_id} "
            f"{arrow} {self.to_location_id}: {self.connection_type.value}>"
        )


# ---------------------------------------------------------------------------
# NPC
# ---------------------------------------------------------------------------


class NPC(Base):
    __tablename__ = "npcs"
    __table_args__ = (Index("ix_npc_campaign", "campaign_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))
    primary_location_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("locations.id"), default=None
    )

    name: Mapped[str] = mapped_column(index=True)
    title: Mapped[Optional[str]] = mapped_column(default=None)
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    appearance: Mapped[Optional[str]] = mapped_column(Text, default=None)
    personality: Mapped[Optional[str]] = mapped_column(Text, default=None)
    motivations: Mapped[Optional[str]] = mapped_column(Text, default=None)
    secrets: Mapped[Optional[str]] = mapped_column(Text, default=None)
    status: Mapped[NPCStatus] = mapped_column(
        Enum(NPCStatus), default=NPCStatus.ALIVE
    )

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationships
    campaign: Mapped[Campaign] = relationship(back_populates="npcs")
    primary_location: Mapped[Optional[Location]] = relationship(
        foreign_keys=[primary_location_id],
    )
    location_associations: Mapped[list[NPCLocationAssociation]] = relationship(
        back_populates="npc",
        cascade="all, delete-orphan",
    )

    # NPC-to-NPC relationships (both directions)
    relationships_out: Mapped[list[NPCRelationship]] = relationship(
        foreign_keys="NPCRelationship.from_npc_id",
        back_populates="from_npc",
        cascade="all, delete-orphan",
    )
    relationships_in: Mapped[list[NPCRelationship]] = relationship(
        foreign_keys="NPCRelationship.to_npc_id",
        back_populates="to_npc",
        cascade="all, delete-orphan",
    )

    @property
    def all_relationships(self) -> list[NPCRelationship]:
        """All relationships regardless of direction."""
        return self.relationships_out + self.relationships_in

    @property
    def locations(self) -> list[Location]:
        """Convenience: all locations this NPC is associated with."""
        return [assoc.location for assoc in self.location_associations]

    @staticmethod
    async def search(
        session: AsyncSession,
        campaign_id: int,
        query: str,
    ) -> list[NPC]:
        """Search NPCs by name (case-insensitive partial match)."""
        stmt = (
            select(NPC)
            .where(
                NPC.campaign_id == campaign_id,
                NPC.name.ilike(f"%{query}%"),
            )
            .order_by(NPC.name)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    def __repr__(self) -> str:
        return f"<NPC {self.id}: {self.name!r} ({self.status.value})>"


# ---------------------------------------------------------------------------
# NPC ↔ Location association
# ---------------------------------------------------------------------------


class NPCLocationAssociation(Base):
    __tablename__ = "npc_location_associations"
    __table_args__ = (
        UniqueConstraint("npc_id", "location_id", name="uq_npc_location"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    npc_id: Mapped[int] = mapped_column(ForeignKey("npcs.id"))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    role: Mapped[Optional[str]] = mapped_column(default=None)
    notes: Mapped[Optional[str]] = mapped_column(Text, default=None)

    # relationships
    npc: Mapped[NPC] = relationship(back_populates="location_associations")
    location: Mapped[Location] = relationship(back_populates="npc_associations")

    def __repr__(self) -> str:
        return f"<NPCLocation npc={self.npc_id} @ location={self.location_id}: {self.role}>"


# ---------------------------------------------------------------------------
# NPC ↔ NPC relationship
# ---------------------------------------------------------------------------


class NPCRelationship(Base):
    __tablename__ = "npc_relationships"
    __table_args__ = (
        UniqueConstraint("from_npc_id", "to_npc_id", name="uq_npc_relationship"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    from_npc_id: Mapped[int] = mapped_column(ForeignKey("npcs.id"))
    to_npc_id: Mapped[int] = mapped_column(ForeignKey("npcs.id"))
    relationship_type: Mapped[NPCRelationshipType] = mapped_column(
        Enum(NPCRelationshipType), default=NPCRelationshipType.OTHER
    )
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    is_secret: Mapped[bool] = mapped_column(default=False)

    # relationships
    from_npc: Mapped[NPC] = relationship(
        foreign_keys=[from_npc_id], back_populates="relationships_out"
    )
    to_npc: Mapped[NPC] = relationship(
        foreign_keys=[to_npc_id], back_populates="relationships_in"
    )

    def __repr__(self) -> str:
        return (
            f"<NPCRelationship {self.from_npc_id} "
            f"-> {self.to_npc_id}: {self.relationship_type.value}>"
        )


# ---------------------------------------------------------------------------
# PlayerCharacter
# ---------------------------------------------------------------------------


class PlayerCharacter(Base):
    __tablename__ = "player_characters"
    __table_args__ = (Index("ix_pc_campaign", "campaign_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))
    primary_location_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("locations.id"), default=None
    )

    # core identity
    name: Mapped[str] = mapped_column(index=True)
    player_name: Mapped[Optional[str]] = mapped_column(default=None)
    race: Mapped[Optional[str]] = mapped_column(default=None)
    character_class: Mapped[Optional[str]] = mapped_column(default=None)
    subclass: Mapped[Optional[str]] = mapped_column(default=None)
    background: Mapped[Optional[str]] = mapped_column(default=None)
    level: Mapped[int] = mapped_column(default=1)
    age: Mapped[Optional[str]] = mapped_column(default=None)

    # descriptive
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    appearance: Mapped[Optional[str]] = mapped_column(Text, default=None)
    personality: Mapped[Optional[str]] = mapped_column(Text, default=None)
    backstory: Mapped[Optional[str]] = mapped_column(Text, default=None)
    notable_features: Mapped[Optional[str]] = mapped_column(Text, default=None)

    is_active: Mapped[bool] = mapped_column(default=True)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationships
    campaign: Mapped[Campaign] = relationship(back_populates="player_characters")
    primary_location: Mapped[Optional[Location]] = relationship(
        foreign_keys=[primary_location_id],
    )
    companions: Mapped[list[Companion]] = relationship(
        back_populates="player_character",
        cascade="all, delete-orphan",
    )
    inventory_items: Mapped[list[InventoryItem]] = relationship(
        back_populates="player_character",
        cascade="all, delete-orphan",
    )
    special_items: Mapped[list[SpecialItem]] = relationship(
        back_populates="player_character",
        cascade="all, delete-orphan",
    )
    pc_relationships: Mapped[list[PCRelationship]] = relationship(
        back_populates="player_character",
        cascade="all, delete-orphan",
    )

    @staticmethod
    async def search(
        session: AsyncSession,
        campaign_id: int,
        query: str,
    ) -> list[PlayerCharacter]:
        stmt = (
            select(PlayerCharacter)
            .where(
                PlayerCharacter.campaign_id == campaign_id,
                PlayerCharacter.name.ilike(f"%{query}%"),
            )
            .order_by(PlayerCharacter.name)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    def __repr__(self) -> str:
        return f"<PlayerCharacter {self.id}: {self.name!r} ({self.character_class})>"


# ---------------------------------------------------------------------------
# Companion (linked to a PlayerCharacter)
# ---------------------------------------------------------------------------


class Companion(Base):
    __tablename__ = "companions"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_character_id: Mapped[int] = mapped_column(
        ForeignKey("player_characters.id")
    )

    name: Mapped[str]
    race: Mapped[Optional[str]] = mapped_column(default=None)
    character_class: Mapped[Optional[str]] = mapped_column(default=None)
    subclass: Mapped[Optional[str]] = mapped_column(default=None)
    background: Mapped[Optional[str]] = mapped_column(default=None)
    age: Mapped[Optional[str]] = mapped_column(default=None)
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    appearance: Mapped[Optional[str]] = mapped_column(Text, default=None)
    personality: Mapped[Optional[str]] = mapped_column(Text, default=None)
    notable_features: Mapped[Optional[str]] = mapped_column(Text, default=None)

    is_active: Mapped[bool] = mapped_column(default=True)

    # relationships
    player_character: Mapped[PlayerCharacter] = relationship(
        back_populates="companions"
    )

    def __repr__(self) -> str:
        return f"<Companion {self.id}: {self.name!r}>"


# ---------------------------------------------------------------------------
# InventoryItem (general gear on a PlayerCharacter)
# ---------------------------------------------------------------------------


class InventoryItem(Base):
    __tablename__ = "inventory_items"
    __table_args__ = (Index("ix_inventory_pc", "player_character_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_character_id: Mapped[int] = mapped_column(
        ForeignKey("player_characters.id")
    )

    name: Mapped[str]
    quantity: Mapped[int] = mapped_column(default=1)
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    category: Mapped[Optional[str]] = mapped_column(default=None)

    # relationships
    player_character: Mapped[PlayerCharacter] = relationship(
        back_populates="inventory_items"
    )

    def __repr__(self) -> str:
        qty = f" x{self.quantity}" if self.quantity > 1 else ""
        return f"<InventoryItem {self.id}: {self.name!r}{qty}>"


# ---------------------------------------------------------------------------
# SpecialItem (magic items, artifacts — with properties & limitations)
# ---------------------------------------------------------------------------


class SpecialItem(Base):
    __tablename__ = "special_items"
    __table_args__ = (Index("ix_special_item_pc", "player_character_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_character_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("player_characters.id"), default=None
    )
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))

    name: Mapped[str]
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    properties: Mapped[Optional[str]] = mapped_column(Text, default=None)
    limitations: Mapped[Optional[str]] = mapped_column(Text, default=None)
    requires_attunement: Mapped[bool] = mapped_column(default=False)
    is_equipped: Mapped[bool] = mapped_column(default=False)

    # relationships
    player_character: Mapped[Optional[PlayerCharacter]] = relationship(
        back_populates="special_items"
    )

    def __repr__(self) -> str:
        return f"<SpecialItem {self.id}: {self.name!r}>"


# ---------------------------------------------------------------------------
# PCRelationship (relationships between PCs / companions / NPCs)
# ---------------------------------------------------------------------------


class PCRelationship(Base):
    """Tracks relationships from a PC's perspective to any named entity."""

    __tablename__ = "pc_relationships"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_character_id: Mapped[int] = mapped_column(
        ForeignKey("player_characters.id")
    )
    target_name: Mapped[str]
    relationship_type: Mapped[Optional[str]] = mapped_column(default=None)
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)
    is_secret: Mapped[bool] = mapped_column(default=False)

    # relationships
    player_character: Mapped[PlayerCharacter] = relationship(
        back_populates="pc_relationships"
    )

    def __repr__(self) -> str:
        return f"<PCRelationship {self.player_character_id} -> {self.target_name!r}>"


# ---------------------------------------------------------------------------
# Session (game session tracking)
# ---------------------------------------------------------------------------


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("campaign_id", "session_number", name="uq_session_number"),
        Index("ix_session_campaign", "campaign_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))
    session_number: Mapped[int]
    title: Mapped[Optional[str]] = mapped_column(default=None)
    scheduled_date: Mapped[Optional[str]] = mapped_column(default=None)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.PLANNED
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    prep_notes: Mapped[Optional[str]] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationships
    campaign: Mapped[Campaign] = relationship(back_populates="sessions")
    notes: Mapped[list[Note]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="Note.session_id",
    )

    def __repr__(self) -> str:
        return f"<Session {self.id}: #{self.session_number} ({self.status.value})>"


# ---------------------------------------------------------------------------
# Note (attachable to any entity)
# ---------------------------------------------------------------------------


class Note(Base):
    __tablename__ = "notes"
    __table_args__ = (
        Index("ix_note_target", "target_type", "target_id"),
        Index("ix_note_session", "session_id"),
        Index("ix_note_campaign", "campaign_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))

    # polymorphic target — what this note is attached to
    target_type: Mapped[NoteTargetType] = mapped_column(Enum(NoteTargetType))
    target_id: Mapped[int]

    # optional session link — "this note was created during session X"
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sessions.id"), default=None
    )

    title: Mapped[Optional[str]] = mapped_column(default=None)
    content: Mapped[str] = mapped_column(Text)
    is_secret: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationships
    session: Mapped[Optional[Session]] = relationship(
        back_populates="notes",
        foreign_keys=[session_id],
    )

    @staticmethod
    async def for_target(
        session: AsyncSession,
        campaign_id: int,
        target_type: NoteTargetType,
        target_id: int,
    ) -> list[Note]:
        """Get all notes attached to a specific entity."""
        stmt = (
            select(Note)
            .where(
                Note.campaign_id == campaign_id,
                Note.target_type == target_type,
                Note.target_id == target_id,
            )
            .order_by(Note.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def for_session(
        session: AsyncSession,
        campaign_id: int,
        session_id: int,
    ) -> list[Note]:
        """Get all notes created during a specific game session."""
        stmt = (
            select(Note)
            .where(
                Note.campaign_id == campaign_id,
                Note.session_id == session_id,
            )
            .order_by(Note.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def search(
        session: AsyncSession,
        campaign_id: int,
        query: str,
    ) -> list[Note]:
        """Search notes by title and content (case-insensitive)."""
        stmt = (
            select(Note)
            .where(
                Note.campaign_id == campaign_id,
                (Note.title.ilike(f"%{query}%")) | (Note.content.ilike(f"%{query}%")),
            )
            .order_by(Note.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    def __repr__(self) -> str:
        return f"<Note {self.id}: {self.target_type.value}/{self.target_id} — {self.title!r}>"


# ---------------------------------------------------------------------------
# Tag + TagAssociation (cross-cutting labels)
# ---------------------------------------------------------------------------


class Tag(Base):
    __tablename__ = "tags"
    __table_args__ = (
        UniqueConstraint("campaign_id", "slug", name="uq_tag_slug"),
        Index("ix_tag_campaign", "campaign_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))
    name: Mapped[str]
    slug: Mapped[str] = mapped_column(index=True)
    color: Mapped[Optional[str]] = mapped_column(default=None)
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)

    # relationships
    campaign: Mapped[Campaign] = relationship(back_populates="tags")
    associations: Mapped[list[TagAssociation]] = relationship(
        back_populates="tag",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Tag {self.id}: {self.name!r}>"


# auto-generate slug for tags
@event.listens_for(Tag, "before_insert")
def _tag_before_insert(_mapper, _connection, target: Tag):
    if not target.slug and target.name:
        target.slug = slugify(target.name)


class TagAssociation(Base):
    """Polymorphic many-to-many: attach any tag to any entity type."""

    __tablename__ = "tag_associations"
    __table_args__ = (
        UniqueConstraint(
            "tag_id", "target_type", "target_id", name="uq_tag_target"
        ),
        Index("ix_tag_assoc_target", "target_type", "target_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id"))
    target_type: Mapped[NoteTargetType] = mapped_column(Enum(NoteTargetType))
    target_id: Mapped[int]

    # relationships
    tag: Mapped[Tag] = relationship(back_populates="associations")

    @staticmethod
    async def tags_for(
        session: AsyncSession,
        target_type: NoteTargetType,
        target_id: int,
    ) -> list[Tag]:
        """Get all tags on a specific entity."""
        stmt = (
            select(Tag)
            .join(TagAssociation)
            .where(
                TagAssociation.target_type == target_type,
                TagAssociation.target_id == target_id,
            )
            .order_by(Tag.name)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def entities_with_tag(
        session: AsyncSession,
        tag_id: int,
        target_type: Optional[NoteTargetType] = None,
    ) -> list[TagAssociation]:
        """Get all entities that have a given tag, optionally filtered by type."""
        stmt = select(TagAssociation).where(TagAssociation.tag_id == tag_id)
        if target_type:
            stmt = stmt.where(TagAssociation.target_type == target_type)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    def __repr__(self) -> str:
        return f"<TagAssociation tag={self.tag_id} → {self.target_type.value}/{self.target_id}>"


# ---------------------------------------------------------------------------
# Database setup helper
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite+aiosqlite:///gm_tool.db"


async def init_db(url: str = DATABASE_URL) -> create_async_engine:
    """Create all tables and return the engine."""
    engine = create_async_engine(url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine

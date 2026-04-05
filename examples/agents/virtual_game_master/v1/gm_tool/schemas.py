"""
GM Tool — Pydantic Schemas

Request and response models for the FastAPI layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models import (
    ConnectionType,
    LocationType,
    NoteTargetType,
    NPCRelationshipType,
    NPCStatus,
    SessionStatus,
    slugify,
)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


class CampaignCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    game_system: str = Field(default="generic", max_length=100)
    description: Optional[str] = None


class CampaignUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    game_system: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = None


class CampaignRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    game_system: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class LocationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, max_length=200)
    parent_id: Optional[int] = None
    location_type: LocationType = LocationType.OTHER
    description: Optional[str] = None
    secrets: Optional[str] = None

    @model_validator(mode="after")
    def ensure_slug(self):
        if self.slug:
            self.slug = slugify(self.slug)
        else:
            self.slug = slugify(self.name)
        return self


class LocationUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, max_length=200)
    parent_id: Optional[int] = None
    location_type: Optional[LocationType] = None
    description: Optional[str] = None
    secrets: Optional[str] = None

    @model_validator(mode="after")
    def normalize_slug(self):
        if self.slug:
            self.slug = slugify(self.slug)
        return self


class LocationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    parent_id: Optional[int]
    name: str
    slug: str
    location_type: LocationType
    description: Optional[str]
    secrets: Optional[str]
    created_at: datetime
    updated_at: datetime


class LocationReadWithPath(LocationRead):
    """LocationRead extended with the computed full path."""

    path: str = ""


class LocationTree(LocationRead):
    """Recursive location with nested children."""

    path: str = ""
    children: list[LocationTree] = []
    npcs: list[NPCSummary] = []


# ---------------------------------------------------------------------------
# LocationConnection
# ---------------------------------------------------------------------------


class LocationConnectionCreate(BaseModel):
    from_location_id: int
    to_location_id: int
    connection_type: ConnectionType = ConnectionType.OTHER
    description: Optional[str] = None
    is_bidirectional: bool = True
    is_secret: bool = False


class LocationConnectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    from_location_id: int
    to_location_id: int
    connection_type: ConnectionType
    description: Optional[str]
    is_bidirectional: bool
    is_secret: bool


# ---------------------------------------------------------------------------
# NPC
# ---------------------------------------------------------------------------


class NPCCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    title: Optional[str] = Field(default=None, max_length=200)
    primary_location_id: Optional[int] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    motivations: Optional[str] = None
    secrets: Optional[str] = None
    status: NPCStatus = NPCStatus.ALIVE


class NPCUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    title: Optional[str] = Field(default=None, max_length=200)
    primary_location_id: Optional[int] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    motivations: Optional[str] = None
    secrets: Optional[str] = None
    status: Optional[NPCStatus] = None


class NPCSummary(BaseModel):
    """Lightweight NPC reference used inside location trees etc."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    title: Optional[str]
    status: NPCStatus


class NPCRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    primary_location_id: Optional[int]
    name: str
    title: Optional[str]
    description: Optional[str]
    appearance: Optional[str]
    personality: Optional[str]
    motivations: Optional[str]
    secrets: Optional[str]
    status: NPCStatus
    created_at: datetime
    updated_at: datetime


class NPCReadFull(NPCRead):
    """NPC with resolved location associations and relationships."""

    location_associations: list[NPCLocationAssociationRead] = []
    relationships: list[NPCRelationshipRead] = []


# ---------------------------------------------------------------------------
# NPC ↔ Location association
# ---------------------------------------------------------------------------


class NPCLocationAssociationCreate(BaseModel):
    npc_id: int
    location_id: int
    role: Optional[str] = Field(default=None, max_length=100)
    notes: Optional[str] = None


class NPCLocationAssociationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    npc_id: int
    location_id: int
    role: Optional[str]
    notes: Optional[str]


# ---------------------------------------------------------------------------
# NPC ↔ NPC relationship
# ---------------------------------------------------------------------------


class NPCRelationshipCreate(BaseModel):
    from_npc_id: int
    to_npc_id: int
    relationship_type: NPCRelationshipType = NPCRelationshipType.OTHER
    description: Optional[str] = None
    is_secret: bool = False


class NPCRelationshipRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    from_npc_id: int
    to_npc_id: int
    relationship_type: NPCRelationshipType
    description: Optional[str]
    is_secret: bool


# ---------------------------------------------------------------------------
# PlayerCharacter
# ---------------------------------------------------------------------------


class PlayerCharacterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    player_name: Optional[str] = Field(default=None, max_length=200)
    race: Optional[str] = Field(default=None, max_length=100)
    character_class: Optional[str] = Field(default=None, max_length=100)
    subclass: Optional[str] = Field(default=None, max_length=100)
    background: Optional[str] = Field(default=None, max_length=200)
    level: int = Field(default=1, ge=1)
    age: Optional[str] = Field(default=None, max_length=50)
    primary_location_id: Optional[int] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    backstory: Optional[str] = None
    notable_features: Optional[str] = None


class PlayerCharacterUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    player_name: Optional[str] = Field(default=None, max_length=200)
    race: Optional[str] = Field(default=None, max_length=100)
    character_class: Optional[str] = Field(default=None, max_length=100)
    subclass: Optional[str] = Field(default=None, max_length=100)
    background: Optional[str] = Field(default=None, max_length=200)
    level: Optional[int] = Field(default=None, ge=1)
    age: Optional[str] = Field(default=None, max_length=50)
    primary_location_id: Optional[int] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    backstory: Optional[str] = None
    notable_features: Optional[str] = None
    is_active: Optional[bool] = None


class PlayerCharacterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    primary_location_id: Optional[int]
    name: str
    player_name: Optional[str]
    race: Optional[str]
    character_class: Optional[str]
    subclass: Optional[str]
    background: Optional[str]
    level: int
    age: Optional[str]
    description: Optional[str]
    appearance: Optional[str]
    personality: Optional[str]
    backstory: Optional[str]
    notable_features: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class PlayerCharacterFull(PlayerCharacterRead):
    """PC with companions, inventory, special items, and relationships."""

    companions: list[CompanionRead] = []
    inventory_items: list[InventoryItemRead] = []
    special_items: list[SpecialItemRead] = []
    relationships: list[PCRelationshipRead] = []


# ---------------------------------------------------------------------------
# Companion
# ---------------------------------------------------------------------------


class CompanionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    race: Optional[str] = Field(default=None, max_length=100)
    character_class: Optional[str] = Field(default=None, max_length=100)
    subclass: Optional[str] = Field(default=None, max_length=100)
    background: Optional[str] = Field(default=None, max_length=200)
    age: Optional[str] = Field(default=None, max_length=50)
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    notable_features: Optional[str] = None


class CompanionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    race: Optional[str] = Field(default=None, max_length=100)
    character_class: Optional[str] = Field(default=None, max_length=100)
    subclass: Optional[str] = Field(default=None, max_length=100)
    background: Optional[str] = Field(default=None, max_length=200)
    age: Optional[str] = Field(default=None, max_length=50)
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    notable_features: Optional[str] = None
    is_active: Optional[bool] = None


class CompanionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_character_id: int
    name: str
    race: Optional[str]
    character_class: Optional[str]
    subclass: Optional[str]
    background: Optional[str]
    age: Optional[str]
    description: Optional[str]
    appearance: Optional[str]
    personality: Optional[str]
    notable_features: Optional[str]
    is_active: bool


# ---------------------------------------------------------------------------
# InventoryItem
# ---------------------------------------------------------------------------


class InventoryItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    quantity: int = Field(default=1, ge=0)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=100)


class InventoryItemUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    quantity: Optional[int] = Field(default=None, ge=0)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=100)


class InventoryItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_character_id: int
    name: str
    quantity: int
    description: Optional[str]
    category: Optional[str]


# ---------------------------------------------------------------------------
# SpecialItem
# ---------------------------------------------------------------------------


class SpecialItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    properties: Optional[str] = None
    limitations: Optional[str] = None
    requires_attunement: bool = False
    is_equipped: bool = False


class SpecialItemUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    properties: Optional[str] = None
    limitations: Optional[str] = None
    requires_attunement: Optional[bool] = None
    is_equipped: Optional[bool] = None
    player_character_id: Optional[int] = None


class SpecialItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_character_id: Optional[int]
    campaign_id: int
    name: str
    description: Optional[str]
    properties: Optional[str]
    limitations: Optional[str]
    requires_attunement: bool
    is_equipped: bool


# ---------------------------------------------------------------------------
# PCRelationship
# ---------------------------------------------------------------------------


class PCRelationshipCreate(BaseModel):
    target_name: str = Field(..., min_length=1, max_length=200)
    relationship_type: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = None
    is_secret: bool = False


class PCRelationshipRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_character_id: int
    target_name: str
    relationship_type: Optional[str]
    description: Optional[str]
    is_secret: bool


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    session_number: int = Field(..., ge=1)
    title: Optional[str] = Field(default=None, max_length=200)
    scheduled_date: Optional[str] = Field(
        default=None, max_length=30, description="Free-form date string, e.g. '2024-03-15' or 'next Friday'"
    )
    status: SessionStatus = SessionStatus.PLANNED
    summary: Optional[str] = None
    prep_notes: Optional[str] = None


class SessionUpdate(BaseModel):
    session_number: Optional[int] = Field(default=None, ge=1)
    title: Optional[str] = Field(default=None, max_length=200)
    scheduled_date: Optional[str] = Field(default=None, max_length=30)
    status: Optional[SessionStatus] = None
    summary: Optional[str] = None
    prep_notes: Optional[str] = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    session_number: int
    title: Optional[str]
    scheduled_date: Optional[str]
    status: SessionStatus
    summary: Optional[str]
    prep_notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class SessionReadWithNotes(SessionRead):
    """Session with its attached notes."""

    notes: list[NoteRead] = []


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------


class NoteCreate(BaseModel):
    target_type: NoteTargetType
    target_id: int
    session_id: Optional[int] = None
    title: Optional[str] = Field(default=None, max_length=200)
    content: str = Field(..., min_length=1)
    is_secret: bool = False


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    content: Optional[str] = Field(default=None, min_length=1)
    session_id: Optional[int] = None
    is_secret: Optional[bool] = None


class NoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    target_type: NoteTargetType
    target_id: int
    session_id: Optional[int]
    title: Optional[str]
    content: str
    is_secret: bool
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------


class TagCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    slug: Optional[str] = Field(default=None, max_length=100)
    color: Optional[str] = Field(
        default=None, max_length=7, description="Hex color, e.g. '#ff6600'"
    )
    description: Optional[str] = None

    @model_validator(mode="after")
    def ensure_slug(self):
        if self.slug:
            self.slug = slugify(self.slug)
        else:
            self.slug = slugify(self.name)
        return self


class TagUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    slug: Optional[str] = Field(default=None, max_length=100)
    color: Optional[str] = Field(default=None, max_length=7)
    description: Optional[str] = None

    @model_validator(mode="after")
    def normalize_slug(self):
        if self.slug:
            self.slug = slugify(self.slug)
        return self


class TagRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    name: str
    slug: str
    color: Optional[str]
    description: Optional[str]


class TagAssociationCreate(BaseModel):
    tag_id: int
    target_type: NoteTargetType
    target_id: int


class TagAssociationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tag_id: int
    target_type: NoteTargetType
    target_id: int


# ---------------------------------------------------------------------------
# WorldLore
# ---------------------------------------------------------------------------


class WorldLoreCreate(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, max_length=200)
    category: Optional[str] = Field(default=None, max_length=100)
    content: str = Field(..., min_length=1)
    is_secret: bool = False

    @model_validator(mode="after")
    def ensure_slug(self):
        if self.slug:
            self.slug = slugify(self.slug)
        else:
            self.slug = slugify(self.topic)
        return self


class WorldLoreUpdate(BaseModel):
    topic: Optional[str] = Field(default=None, min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, max_length=200)
    category: Optional[str] = Field(default=None, max_length=100)
    content: Optional[str] = Field(default=None, min_length=1)
    is_secret: Optional[bool] = None

    @model_validator(mode="after")
    def normalize_slug(self):
        if self.slug:
            self.slug = slugify(self.slug)
        return self


class WorldLoreRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    campaign_id: int
    topic: str
    slug: str
    category: Optional[str]
    content: str
    is_secret: bool
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Rebuild forward refs for recursive / cross-referencing models
# ---------------------------------------------------------------------------

LocationTree.model_rebuild()
NPCReadFull.model_rebuild()
SessionReadWithNotes.model_rebuild()
PlayerCharacterFull.model_rebuild()

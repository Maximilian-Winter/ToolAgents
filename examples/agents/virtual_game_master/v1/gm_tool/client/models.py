"""
GM Tool Client — Response Models

Lightweight Pydantic models for deserializing API responses.
These are independent of the server-side schemas so the client
package can be distributed separately.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


class Campaign(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    game_system: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class Location(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    parent_id: Optional[int] = None
    name: str
    slug: str
    location_type: str
    description: Optional[str] = None
    secrets: Optional[str] = None
    path: str = ""
    created_at: datetime
    updated_at: datetime


class NPCSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    title: Optional[str] = None
    status: str


class LocationTree(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    parent_id: Optional[int] = None
    name: str
    slug: str
    location_type: str
    description: Optional[str] = None
    secrets: Optional[str] = None
    path: str = ""
    children: list[LocationTree] = []
    npcs: list[NPCSummary] = []
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# LocationConnection
# ---------------------------------------------------------------------------


class LocationConnection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    from_location_id: int
    to_location_id: int
    connection_type: str
    description: Optional[str] = None
    is_bidirectional: bool = True
    is_secret: bool = False


# ---------------------------------------------------------------------------
# NPC
# ---------------------------------------------------------------------------


class NPC(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    primary_location_id: Optional[int] = None
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    motivations: Optional[str] = None
    secrets: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime


class NPCLocationAssociation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    npc_id: int
    location_id: int
    role: Optional[str] = None
    notes: Optional[str] = None


class NPCRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    from_npc_id: int
    to_npc_id: int
    relationship_type: str
    description: Optional[str] = None
    is_secret: bool = False


class NPCFull(NPC):
    location_associations: list[NPCLocationAssociation] = []
    relationships: list[NPCRelationship] = []


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    session_number: int
    title: Optional[str] = None
    scheduled_date: Optional[str] = None
    status: str
    summary: Optional[str] = None
    prep_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SessionWithNotes(Session):
    notes: list[Note] = []


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------


class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    target_type: str
    target_id: int
    session_id: Optional[int] = None
    title: Optional[str] = None
    content: str
    is_secret: bool = False
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------


class Tag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    name: str
    slug: str
    color: Optional[str] = None
    description: Optional[str] = None


class TagAssociation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    tag_id: int
    target_type: str
    target_id: int


# ---------------------------------------------------------------------------
# PlayerCharacter
# ---------------------------------------------------------------------------


class PlayerCharacter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    campaign_id: int
    primary_location_id: Optional[int] = None
    name: str
    player_name: Optional[str] = None
    race: Optional[str] = None
    character_class: Optional[str] = None
    subclass: Optional[str] = None
    background: Optional[str] = None
    level: int = 1
    age: Optional[str] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    backstory: Optional[str] = None
    notable_features: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class Companion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    player_character_id: int
    name: str
    race: Optional[str] = None
    character_class: Optional[str] = None
    subclass: Optional[str] = None
    background: Optional[str] = None
    age: Optional[str] = None
    description: Optional[str] = None
    appearance: Optional[str] = None
    personality: Optional[str] = None
    notable_features: Optional[str] = None
    is_active: bool = True


class InventoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    player_character_id: int
    name: str
    quantity: int = 1
    description: Optional[str] = None
    category: Optional[str] = None


class SpecialItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    player_character_id: Optional[int] = None
    campaign_id: int
    name: str
    description: Optional[str] = None
    properties: Optional[str] = None
    limitations: Optional[str] = None
    requires_attunement: bool = False
    is_equipped: bool = False


class PCRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    player_character_id: int
    target_name: str
    relationship_type: Optional[str] = None
    description: Optional[str] = None
    is_secret: bool = False


class PlayerCharacterFull(PlayerCharacter):
    companions: list[Companion] = []
    inventory_items: list[InventoryItem] = []
    special_items: list[SpecialItem] = []
    relationships: list[PCRelationship] = []


# Rebuild forward refs
LocationTree.model_rebuild()
SessionWithNotes.model_rebuild()

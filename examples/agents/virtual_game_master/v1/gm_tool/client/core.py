"""
GM Tool Client — Core

Async HTTP client with typed sub-clients for each resource.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from .models import (
    Campaign,
    Companion,
    InventoryItem,
    Location,
    LocationConnection,
    LocationTree,
    Note,
    NPC,
    NPCFull,
    NPCLocationAssociation,
    NPCRelationship,
    PCRelationship,
    PlayerCharacter,
    PlayerCharacterFull,
    Session,
    SessionWithNotes,
    SpecialItem,
    Tag,
    TagAssociation,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GMClientError(Exception):
    """Base error for the GM client."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class NotFoundError(GMClientError):
    pass


class ConflictError(GMClientError):
    pass


# ---------------------------------------------------------------------------
# Base HTTP helper
# ---------------------------------------------------------------------------


class _BaseClient:
    """Thin wrapper around httpx.AsyncClient with error handling."""

    def __init__(self, http: httpx.AsyncClient):
        self._http = http

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> Any:
        response = await self._http.request(
            method, path, params=params, json=json
        )
        if response.status_code == 204:
            return None
        if response.status_code == 404:
            detail = response.json().get("detail", "Not found")
            raise NotFoundError(404, detail)
        if response.status_code == 409:
            detail = response.json().get("detail", "Conflict")
            raise ConflictError(409, detail)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            raise GMClientError(response.status_code, detail)
        return response.json()

    async def _get(self, path: str, params: dict | None = None) -> Any:
        # filter out None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        return await self._request("GET", path, params=params or None)

    async def _post(self, path: str, **data) -> Any:
        # filter out None values unless explicitly set
        return await self._request("POST", path, json=data)

    async def _patch(self, path: str, **data) -> Any:
        # only send fields that were explicitly passed
        return await self._request("PATCH", path, json=data)

    async def _delete(self, path: str) -> None:
        await self._request("DELETE", path)


# ---------------------------------------------------------------------------
# Sub-clients
# ---------------------------------------------------------------------------


class CampaignClient(_BaseClient):

    async def list(self) -> list[Campaign]:
        data = await self._get("/campaigns/")
        return [Campaign(**c) for c in data]

    async def get(self, campaign_id: int) -> Campaign:
        data = await self._get(f"/campaigns/{campaign_id}")
        return Campaign(**data)

    async def create(
        self,
        name: str,
        game_system: str = "generic",
        description: str | None = None,
    ) -> Campaign:
        data = await self._post(
            "/campaigns/",
            name=name,
            game_system=game_system,
            description=description,
        )
        return Campaign(**data)

    async def update(self, campaign_id: int, **fields) -> Campaign:
        data = await self._patch(f"/campaigns/{campaign_id}", **fields)
        return Campaign(**data)

    async def delete(self, campaign_id: int) -> None:
        await self._delete(f"/campaigns/{campaign_id}")


class LocationClient(_BaseClient):

    def _base(self, campaign_id: int) -> str:
        return f"/campaigns/{campaign_id}/locations"

    async def list(
        self,
        campaign_id: int,
        *,
        parent_id: int | None = None,
        root_only: bool = False,
    ) -> list[Location]:
        data = await self._get(
            self._base(campaign_id) + "/",
            {"parent_id": parent_id, "root_only": root_only or None},
        )
        return [Location(**loc) for loc in data]

    async def get(self, campaign_id: int, location_id: int) -> Location:
        data = await self._get(f"{self._base(campaign_id)}/{location_id}")
        return Location(**data)

    async def create(
        self,
        campaign_id: int,
        name: str,
        location_type: str = "other",
        parent_id: int | None = None,
        slug: str | None = None,
        description: str | None = None,
        secrets: str | None = None,
    ) -> Location:
        body: dict[str, Any] = {"name": name, "location_type": location_type}
        if parent_id is not None:
            body["parent_id"] = parent_id
        if slug is not None:
            body["slug"] = slug
        if description is not None:
            body["description"] = description
        if secrets is not None:
            body["secrets"] = secrets
        data = await self._request("POST", self._base(campaign_id) + "/", json=body)
        return Location(**data)

    async def update(self, campaign_id: int, location_id: int, **fields) -> Location:
        data = await self._patch(
            f"{self._base(campaign_id)}/{location_id}", **fields
        )
        return Location(**data)

    async def delete(self, campaign_id: int, location_id: int) -> None:
        await self._delete(f"{self._base(campaign_id)}/{location_id}")

    async def children(self, campaign_id: int, location_id: int) -> list[Location]:
        data = await self._get(f"{self._base(campaign_id)}/{location_id}/children")
        return [Location(**loc) for loc in data]

    async def resolve_path(self, campaign_id: int, path: str) -> Location:
        data = await self._get(self._base(campaign_id) + "/resolve", {"path": path})
        return Location(**data)

    async def tree(self, campaign_id: int) -> list[LocationTree]:
        data = await self._get(self._base(campaign_id) + "/tree")
        return [LocationTree(**node) for node in data]

    async def search(self, campaign_id: int, query: str) -> list[Location]:
        data = await self._get(self._base(campaign_id) + "/search", {"q": query})
        return [Location(**loc) for loc in data]

    # --- connections ---

    async def connect(
        self,
        campaign_id: int,
        from_location_id: int,
        to_location_id: int,
        connection_type: str = "other",
        description: str | None = None,
        is_bidirectional: bool = True,
        is_secret: bool = False,
    ) -> LocationConnection:
        data = await self._request(
            "POST",
            f"{self._base(campaign_id)}/{from_location_id}/connections",
            json={
                "from_location_id": from_location_id,
                "to_location_id": to_location_id,
                "connection_type": connection_type,
                "description": description,
                "is_bidirectional": is_bidirectional,
                "is_secret": is_secret,
            },
        )
        return LocationConnection(**data)

    async def connections(
        self, campaign_id: int, location_id: int
    ) -> list[LocationConnection]:
        data = await self._get(
            f"{self._base(campaign_id)}/{location_id}/connections"
        )
        return [LocationConnection(**c) for c in data]

    async def delete_connection(
        self, campaign_id: int, connection_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/connections/{connection_id}"
        )


class NPCClient(_BaseClient):

    def _base(self, campaign_id: int) -> str:
        return f"/campaigns/{campaign_id}/npcs"

    async def list(
        self,
        campaign_id: int,
        *,
        status: str | None = None,
        location_id: int | None = None,
    ) -> list[NPC]:
        data = await self._get(
            self._base(campaign_id) + "/",
            {"status": status, "location_id": location_id},
        )
        return [NPC(**npc) for npc in data]

    async def get(self, campaign_id: int, npc_id: int) -> NPCFull:
        data = await self._get(f"{self._base(campaign_id)}/{npc_id}")
        return NPCFull(**data)

    async def create(
        self,
        campaign_id: int,
        name: str,
        title: str | None = None,
        primary_location_id: int | None = None,
        description: str | None = None,
        appearance: str | None = None,
        personality: str | None = None,
        motivations: str | None = None,
        secrets: str | None = None,
        status: str = "alive",
    ) -> NPC:
        body: dict[str, Any] = {"name": name, "status": status}
        for key in (
            "title", "primary_location_id", "description", "appearance",
            "personality", "motivations", "secrets",
        ):
            val = locals()[key]
            if val is not None:
                body[key] = val
        data = await self._request("POST", self._base(campaign_id) + "/", json=body)
        return NPC(**data)

    async def update(self, campaign_id: int, npc_id: int, **fields) -> NPC:
        data = await self._patch(
            f"{self._base(campaign_id)}/{npc_id}", **fields
        )
        return NPC(**data)

    async def delete(self, campaign_id: int, npc_id: int) -> None:
        await self._delete(f"{self._base(campaign_id)}/{npc_id}")

    async def search(self, campaign_id: int, query: str) -> list[NPC]:
        data = await self._get(self._base(campaign_id) + "/search", {"q": query})
        return [NPC(**npc) for npc in data]

    # --- location associations ---

    async def add_location(
        self,
        campaign_id: int,
        npc_id: int,
        location_id: int,
        role: str | None = None,
        notes: str | None = None,
    ) -> NPCLocationAssociation:
        body: dict[str, Any] = {"npc_id": npc_id, "location_id": location_id}
        if role is not None:
            body["role"] = role
        if notes is not None:
            body["notes"] = notes
        data = await self._request(
            "POST", f"{self._base(campaign_id)}/{npc_id}/locations", json=body
        )
        return NPCLocationAssociation(**data)

    async def locations(
        self, campaign_id: int, npc_id: int
    ) -> list[NPCLocationAssociation]:
        data = await self._get(f"{self._base(campaign_id)}/{npc_id}/locations")
        return [NPCLocationAssociation(**a) for a in data]

    async def remove_location(
        self, campaign_id: int, npc_id: int, association_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{npc_id}/locations/{association_id}"
        )

    # --- NPC relationships ---

    async def add_relationship(
        self,
        campaign_id: int,
        from_npc_id: int,
        to_npc_id: int,
        relationship_type: str = "other",
        description: str | None = None,
        is_secret: bool = False,
    ) -> NPCRelationship:
        data = await self._request(
            "POST",
            f"{self._base(campaign_id)}/{from_npc_id}/relationships",
            json={
                "from_npc_id": from_npc_id,
                "to_npc_id": to_npc_id,
                "relationship_type": relationship_type,
                "description": description,
                "is_secret": is_secret,
            },
        )
        return NPCRelationship(**data)

    async def relationships(
        self, campaign_id: int, npc_id: int
    ) -> list[NPCRelationship]:
        data = await self._get(
            f"{self._base(campaign_id)}/{npc_id}/relationships"
        )
        return [NPCRelationship(**r) for r in data]

    async def remove_relationship(
        self, campaign_id: int, npc_id: int, relationship_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{npc_id}/relationships/{relationship_id}"
        )


class SessionClient(_BaseClient):

    def _base(self, campaign_id: int) -> str:
        return f"/campaigns/{campaign_id}/sessions"

    async def list(
        self, campaign_id: int, *, status: str | None = None
    ) -> list[Session]:
        data = await self._get(self._base(campaign_id) + "/", {"status": status})
        return [Session(**s) for s in data]

    async def get(self, campaign_id: int, session_id: int) -> SessionWithNotes:
        data = await self._get(f"{self._base(campaign_id)}/{session_id}")
        return SessionWithNotes(**data)

    async def create(
        self,
        campaign_id: int,
        session_number: int,
        title: str | None = None,
        scheduled_date: str | None = None,
        status: str = "planned",
        summary: str | None = None,
        prep_notes: str | None = None,
    ) -> Session:
        body: dict[str, Any] = {"session_number": session_number, "status": status}
        for key in ("title", "scheduled_date", "summary", "prep_notes"):
            val = locals()[key]
            if val is not None:
                body[key] = val
        data = await self._request("POST", self._base(campaign_id) + "/", json=body)
        return Session(**data)

    async def update(self, campaign_id: int, session_id: int, **fields) -> Session:
        data = await self._patch(
            f"{self._base(campaign_id)}/{session_id}", **fields
        )
        return Session(**data)

    async def delete(self, campaign_id: int, session_id: int) -> None:
        await self._delete(f"{self._base(campaign_id)}/{session_id}")


class NoteClient(_BaseClient):

    def _base(self, campaign_id: int) -> str:
        return f"/campaigns/{campaign_id}/notes"

    async def list(
        self,
        campaign_id: int,
        *,
        target_type: str | None = None,
        target_id: int | None = None,
        session_id: int | None = None,
    ) -> list[Note]:
        data = await self._get(
            self._base(campaign_id) + "/",
            {"target_type": target_type, "target_id": target_id, "session_id": session_id},
        )
        return [Note(**n) for n in data]

    async def get(self, campaign_id: int, note_id: int) -> Note:
        data = await self._get(f"{self._base(campaign_id)}/{note_id}")
        return Note(**data)

    async def create(
        self,
        campaign_id: int,
        target_type: str,
        target_id: int,
        content: str,
        title: str | None = None,
        session_id: int | None = None,
        is_secret: bool = False,
    ) -> Note:
        body: dict[str, Any] = {
            "target_type": target_type,
            "target_id": target_id,
            "content": content,
            "is_secret": is_secret,
        }
        if title is not None:
            body["title"] = title
        if session_id is not None:
            body["session_id"] = session_id
        data = await self._request("POST", self._base(campaign_id) + "/", json=body)
        return Note(**data)

    async def update(self, campaign_id: int, note_id: int, **fields) -> Note:
        data = await self._patch(
            f"{self._base(campaign_id)}/{note_id}", **fields
        )
        return Note(**data)

    async def delete(self, campaign_id: int, note_id: int) -> None:
        await self._delete(f"{self._base(campaign_id)}/{note_id}")

    async def search(self, campaign_id: int, query: str) -> list[Note]:
        data = await self._get(self._base(campaign_id) + "/search", {"q": query})
        return [Note(**n) for n in data]


class TagClient(_BaseClient):

    def _base(self, campaign_id: int) -> str:
        return f"/campaigns/{campaign_id}/tags"

    async def list(self, campaign_id: int) -> list[Tag]:
        data = await self._get(self._base(campaign_id) + "/")
        return [Tag(**t) for t in data]

    async def get(self, campaign_id: int, tag_id: int) -> Tag:
        data = await self._get(f"{self._base(campaign_id)}/{tag_id}")
        return Tag(**data)

    async def create(
        self,
        campaign_id: int,
        name: str,
        color: str | None = None,
        description: str | None = None,
    ) -> Tag:
        body: dict[str, Any] = {"name": name}
        if color is not None:
            body["color"] = color
        if description is not None:
            body["description"] = description
        data = await self._request("POST", self._base(campaign_id) + "/", json=body)
        return Tag(**data)

    async def update(self, campaign_id: int, tag_id: int, **fields) -> Tag:
        data = await self._patch(
            f"{self._base(campaign_id)}/{tag_id}", **fields
        )
        return Tag(**data)

    async def delete(self, campaign_id: int, tag_id: int) -> None:
        await self._delete(f"{self._base(campaign_id)}/{tag_id}")

    async def apply(
        self,
        campaign_id: int,
        tag_id: int,
        target_type: str,
        target_id: int,
    ) -> TagAssociation:
        data = await self._request(
            "POST",
            f"{self._base(campaign_id)}/{tag_id}/apply",
            json={
                "tag_id": tag_id,
                "target_type": target_type,
                "target_id": target_id,
            },
        )
        return TagAssociation(**data)

    async def remove(
        self,
        campaign_id: int,
        tag_id: int,
        target_type: str,
        target_id: int,
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{tag_id}/apply/{target_type}/{target_id}"
        )

    async def tags_on(
        self, campaign_id: int, target_type: str, target_id: int
    ) -> list[Tag]:
        data = await self._get(
            f"{self._base(campaign_id)}/on/{target_type}/{target_id}"
        )
        return [Tag(**t) for t in data]

    async def entities_with(
        self,
        campaign_id: int,
        tag_id: int,
        target_type: str | None = None,
    ) -> list[TagAssociation]:
        data = await self._get(
            f"{self._base(campaign_id)}/{tag_id}/entities",
            {"target_type": target_type},
        )
        return [TagAssociation(**a) for a in data]


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class PlayerCharacterClient(_BaseClient):

    def _base(self, campaign_id: int) -> str:
        return f"/campaigns/{campaign_id}/player-characters"

    async def list(
        self, campaign_id: int, *, active_only: bool = False
    ) -> list[PlayerCharacter]:
        params = {}
        if active_only:
            params["active_only"] = True
        data = await self._get(self._base(campaign_id) + "/", params or None)
        return [PlayerCharacter(**pc) for pc in data]

    async def get(self, campaign_id: int, pc_id: int) -> PlayerCharacterFull:
        data = await self._get(f"{self._base(campaign_id)}/{pc_id}")
        return PlayerCharacterFull(**data)

    async def create(
        self,
        campaign_id: int,
        name: str,
        **fields,
    ) -> PlayerCharacter:
        body: dict[str, Any] = {"name": name, **fields}
        data = await self._request("POST", self._base(campaign_id) + "/", json=body)
        return PlayerCharacter(**data)

    async def update(self, campaign_id: int, pc_id: int, **fields) -> PlayerCharacter:
        data = await self._patch(f"{self._base(campaign_id)}/{pc_id}", **fields)
        return PlayerCharacter(**data)

    async def delete(self, campaign_id: int, pc_id: int) -> None:
        await self._delete(f"{self._base(campaign_id)}/{pc_id}")

    async def search(self, campaign_id: int, query: str) -> list[PlayerCharacter]:
        data = await self._get(self._base(campaign_id) + "/search", {"q": query})
        return [PlayerCharacter(**pc) for pc in data]

    # --- companions ---

    async def add_companion(
        self, campaign_id: int, pc_id: int, name: str, **fields
    ) -> Companion:
        body: dict[str, Any] = {"name": name, **fields}
        data = await self._request(
            "POST", f"{self._base(campaign_id)}/{pc_id}/companions", json=body
        )
        return Companion(**data)

    async def list_companions(
        self, campaign_id: int, pc_id: int
    ) -> list[Companion]:
        data = await self._get(f"{self._base(campaign_id)}/{pc_id}/companions")
        return [Companion(**c) for c in data]

    async def update_companion(
        self, campaign_id: int, pc_id: int, companion_id: int, **fields
    ) -> Companion:
        data = await self._patch(
            f"{self._base(campaign_id)}/{pc_id}/companions/{companion_id}",
            **fields,
        )
        return Companion(**data)

    async def remove_companion(
        self, campaign_id: int, pc_id: int, companion_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{pc_id}/companions/{companion_id}"
        )

    # --- inventory ---

    async def add_inventory_item(
        self, campaign_id: int, pc_id: int, name: str, **fields
    ) -> InventoryItem:
        body: dict[str, Any] = {"name": name, **fields}
        data = await self._request(
            "POST", f"{self._base(campaign_id)}/{pc_id}/inventory", json=body
        )
        return InventoryItem(**data)

    async def list_inventory(
        self, campaign_id: int, pc_id: int, *, category: str | None = None
    ) -> list[InventoryItem]:
        params = {}
        if category:
            params["category"] = category
        data = await self._get(
            f"{self._base(campaign_id)}/{pc_id}/inventory", params or None
        )
        return [InventoryItem(**i) for i in data]

    async def update_inventory_item(
        self, campaign_id: int, pc_id: int, item_id: int, **fields
    ) -> InventoryItem:
        data = await self._patch(
            f"{self._base(campaign_id)}/{pc_id}/inventory/{item_id}", **fields
        )
        return InventoryItem(**data)

    async def remove_inventory_item(
        self, campaign_id: int, pc_id: int, item_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{pc_id}/inventory/{item_id}"
        )

    # --- special items ---

    async def add_special_item(
        self, campaign_id: int, pc_id: int, name: str, **fields
    ) -> SpecialItem:
        body: dict[str, Any] = {"name": name, **fields}
        data = await self._request(
            "POST", f"{self._base(campaign_id)}/{pc_id}/special-items", json=body
        )
        return SpecialItem(**data)

    async def list_special_items(
        self, campaign_id: int, pc_id: int
    ) -> list[SpecialItem]:
        data = await self._get(
            f"{self._base(campaign_id)}/{pc_id}/special-items"
        )
        return [SpecialItem(**s) for s in data]

    async def update_special_item(
        self, campaign_id: int, pc_id: int, item_id: int, **fields
    ) -> SpecialItem:
        data = await self._patch(
            f"{self._base(campaign_id)}/{pc_id}/special-items/{item_id}",
            **fields,
        )
        return SpecialItem(**data)

    async def remove_special_item(
        self, campaign_id: int, pc_id: int, item_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{pc_id}/special-items/{item_id}"
        )

    # --- PC relationships ---

    async def add_relationship(
        self,
        campaign_id: int,
        pc_id: int,
        target_name: str,
        relationship_type: str | None = None,
        description: str | None = None,
        is_secret: bool = False,
    ) -> PCRelationship:
        body: dict[str, Any] = {
            "target_name": target_name,
            "is_secret": is_secret,
        }
        if relationship_type:
            body["relationship_type"] = relationship_type
        if description:
            body["description"] = description
        data = await self._request(
            "POST",
            f"{self._base(campaign_id)}/{pc_id}/relationships",
            json=body,
        )
        return PCRelationship(**data)

    async def list_relationships(
        self, campaign_id: int, pc_id: int
    ) -> list[PCRelationship]:
        data = await self._get(
            f"{self._base(campaign_id)}/{pc_id}/relationships"
        )
        return [PCRelationship(**r) for r in data]

    async def remove_relationship(
        self, campaign_id: int, pc_id: int, rel_id: int
    ) -> None:
        await self._delete(
            f"{self._base(campaign_id)}/{pc_id}/relationships/{rel_id}"
        )


class GMClient:
    """
    Async client for the GM Tool API.

    Usage:
        async with GMClient("http://localhost:8000") as gm:
            campaigns = await gm.campaigns.list()
    """

    def __init__(self, base_url: str = "http://localhost:8000", **httpx_kwargs):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
            **httpx_kwargs,
        )
        self.campaigns = CampaignClient(self._http)
        self.locations = LocationClient(self._http)
        self.npcs = NPCClient(self._http)
        self.pcs = PlayerCharacterClient(self._http)
        self.sessions = SessionClient(self._http)
        self.notes = NoteClient(self._http)
        self.tags = TagClient(self._http)

    async def __aenter__(self) -> GMClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def health(self) -> dict:
        """Check if the API is reachable."""
        response = await self._http.get("/")
        return response.json()

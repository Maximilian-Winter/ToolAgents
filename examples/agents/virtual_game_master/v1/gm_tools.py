"""
gm_tools.py — GM Tool Integration for Virtual Game Master
==========================================================

Provides:
  1. get_read_tools()  — 2 retrieval FunctionTools for the AI Game Master
  2. get_write_tools() — write FunctionTools for the save/update flow
  3. YAML importer for seeding world lore from scenario files

The guidance text and update agent prompt live in separate .txt files
(gm_tool_guidance.txt, gm_tool_update_prompt.txt) alongside the other
system message templates.

Requires the GM Tool FastAPI server to be running:
    cd gm_tool && uvicorn app:app --port 8000

Usage:
    from gm_tools import GMToolkit

    toolkit = GMToolkit(base_url="http://localhost:8000", campaign_id=1)

    read_tools  = toolkit.get_read_tools()   # 2 tools for the game master
    write_tools = toolkit.get_write_tools()   # write tools for the save flow
"""

from __future__ import annotations

import enum
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool


# ══════════════════════════════════════════════════════════════════
# Resource type enum
# ══════════════════════════════════════════════════════════════════


class CampaignResource(str, enum.Enum):
    """The types of data stored in the campaign database."""

    LOCATIONS = "locations"
    NPCS = "npcs"
    PLAYER_CHARACTERS = "player_characters"
    WORLD_LORE = "world_lore"
    NOTES = "notes"
    SESSIONS = "sessions"


# ══════════════════════════════════════════════════════════════════
# GMToolkit
# ══════════════════════════════════════════════════════════════════


class GMToolkit:
    """
    Integrates the GM Tool campaign database with the Virtual Game Master.

    Two entry points for tools:
      - get_read_tools()  → 2 retrieval tools (for the game master LLM)
      - get_write_tools() → write tools (for the save/summarisation flow)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        campaign_id: int = 1,
    ):
        self.base_url = base_url
        self.campaign_id = campaign_id
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    def close(self):
        self._http.close()

    def is_available(self) -> bool:
        """Check if the GM Tool API is reachable."""
        try:
            r = self._http.get("/")
            return r.status_code < 500
        except httpx.ConnectError:
            return False

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None):
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        r = self._http.get(path, params=params or None)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        if r.status_code == 204:
            return None
        return r.json()

    def _post(self, path: str, body: dict):
        body = {k: v for k, v in body.items() if v is not None}
        r = self._http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, body: dict):
        body = {k: v for k, v in body.items() if v is not None}
        r = self._http.patch(path, json=body)
        r.raise_for_status()
        return r.json()

    def _base(self, resource: str) -> str:
        return f"/campaigns/{self.campaign_id}/{resource}"

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_location(loc: dict, detailed: bool = False) -> str:
        lines = [f"**{loc['name']}** (id={loc['id']}, type={loc['location_type']})"]
        if loc.get("path"):
            lines.append(f"  Path: {loc['path']}")
        if loc.get("description"):
            lines.append(f"  {loc['description']}")
        if detailed and loc.get("secrets"):
            lines.append(f"  [SECRET] {loc['secrets']}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_npc(npc: dict, detailed: bool = False) -> str:
        status = npc.get("status", "unknown").upper()
        lines = [f"**{npc['name']}** (id={npc['id']}, {status})"]
        if npc.get("title"):
            lines[0] += f" — {npc['title']}"
        for field in ("description", "appearance", "personality", "motivations"):
            if npc.get(field):
                lines.append(f"  {field.title()}: {npc[field]}")
        if detailed:
            if npc.get("secrets"):
                lines.append(f"  [SECRET] {npc['secrets']}")
            for assoc in npc.get("location_associations", []):
                role = f" ({assoc['role']})" if assoc.get("role") else ""
                lines.append(f"  Location: id={assoc['location_id']}{role}")
            for rel in npc.get("relationships", []):
                secret = " [SECRET]" if rel.get("is_secret") else ""
                desc = f" — {rel['description']}" if rel.get("description") else ""
                if rel["from_npc_id"] == npc["id"]:
                    lines.append(f"  Rel -> npc_id={rel['to_npc_id']}: {rel['relationship_type']}{desc}{secret}")
                else:
                    lines.append(f"  Rel <- npc_id={rel['from_npc_id']}: {rel['relationship_type']}{desc}{secret}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_pc(pc: dict) -> str:
        lines = [f"**{pc['name']}** (id={pc['id']}, Lv{pc.get('level', '?')})"]
        if pc.get("player_name"):
            lines[0] += f" — played by {pc['player_name']}"
        for f in ("race", "character_class", "subclass", "background"):
            if pc.get(f):
                lines.append(f"  {f.replace('_', ' ').title()}: {pc[f]}")
        for f in ("description", "appearance", "personality", "backstory"):
            if pc.get(f):
                lines.append(f"  {f.title()}: {pc[f]}")
        for c in pc.get("companions", []):
            active = "" if c.get("is_active", True) else " (inactive)"
            lines.append(f"  Companion: {c['name']}{active}")
        for item in pc.get("inventory_items", []):
            qty = f" x{item['quantity']}" if item.get("quantity", 1) > 1 else ""
            lines.append(f"  Item: {item['name']}{qty}")
        for item in pc.get("special_items", []):
            equipped = " [EQUIPPED]" if item.get("is_equipped") else ""
            lines.append(f"  Special: {item['name']}{equipped}")
        for rel in pc.get("relationships", []):
            rtype = f" ({rel['relationship_type']})" if rel.get("relationship_type") else ""
            lines.append(f"  Rel: {rel['target_name']}{rtype}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_world_lore(entry: dict) -> str:
        cat = f" [{entry['category']}]" if entry.get("category") else ""
        secret = " [SECRET]" if entry.get("is_secret") else ""
        return f"**{entry['topic']}**{cat}{secret} (id={entry['id']})\n{entry['content']}"

    @staticmethod
    def _fmt_note(note: dict) -> str:
        title = note.get("title") or "(untitled)"
        secret = " [SECRET]" if note.get("is_secret") else ""
        return f"**{title}** (id={note['id']}, on {note['target_type']} #{note['target_id']}){secret}\n  {note['content']}"

    @staticmethod
    def _fmt_session(session: dict) -> str:
        lines = [f"**Session #{session['session_number']}** (id={session['id']}, {session['status']})"]
        if session.get("title"):
            lines[0] += f" — {session['title']}"
        for f in ("scheduled_date", "summary", "prep_notes"):
            if session.get(f):
                lines.append(f"  {f.replace('_', ' ').title()}: {session[f]}")
        for note in session.get("notes", []):
            lines.append(f"  Note: {note.get('title', '(untitled)')}: {note['content'][:100]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Search dispatcher
    # ------------------------------------------------------------------

    def search(self, resource_type: CampaignResource, query: str) -> str:
        """Search a resource type by text query. Returns formatted results."""
        try:
            if resource_type == CampaignResource.LOCATIONS:
                data = self._get(self._base("locations") + "/search", {"q": query})
                if not data:
                    return f"No locations found matching '{query}'."
                return "\n\n".join(self._fmt_location(loc) for loc in data)

            elif resource_type == CampaignResource.NPCS:
                data = self._get(self._base("npcs") + "/search", {"q": query})
                if not data:
                    return f"No NPCs found matching '{query}'."
                return "\n\n".join(self._fmt_npc(npc) for npc in data)

            elif resource_type == CampaignResource.PLAYER_CHARACTERS:
                data = self._get(self._base("player-characters") + "/search", {"q": query})
                if not data:
                    return f"No player characters found matching '{query}'."
                return "\n\n".join(self._fmt_pc(pc) for pc in data)

            elif resource_type == CampaignResource.WORLD_LORE:
                data = self._get(self._base("world-lore") + "/search", {"q": query})
                if not data:
                    return f"No world lore found matching '{query}'."
                return "\n\n".join(self._fmt_world_lore(e) for e in data)

            elif resource_type == CampaignResource.NOTES:
                data = self._get(self._base("notes") + "/search", {"q": query})
                if not data:
                    return f"No notes found matching '{query}'."
                return "\n\n".join(self._fmt_note(n) for n in data)

            elif resource_type == CampaignResource.SESSIONS:
                data = self._get(self._base("sessions") + "/")
                if not data:
                    return "No sessions found."
                q_lower = query.lower()
                matches = [
                    s for s in data
                    if q_lower in (s.get("title") or "").lower()
                    or q_lower in (s.get("summary") or "").lower()
                    or q_lower in str(s.get("session_number", ""))
                ]
                if not matches:
                    return f"No sessions matching '{query}'."
                return "\n\n".join(self._fmt_session(s) for s in matches)

            else:
                return f"Unknown resource type: {resource_type}"
        except Exception as e:
            return f"Error searching {resource_type}: {e}"

    # ------------------------------------------------------------------
    # Detail fetcher
    # ------------------------------------------------------------------

    def get_details(self, resource_type: CampaignResource, record_id: int) -> str:
        """Get full details of a specific record by ID."""
        try:
            if resource_type == CampaignResource.LOCATIONS:
                loc = self._get(f"{self._base('locations')}/{record_id}")
                if not loc:
                    return f"Location {record_id} not found."
                result = self._fmt_location(loc, detailed=True)
                conns = self._get(f"{self._base('locations')}/{record_id}/connections")
                if conns:
                    result += "\n  Connections:"
                    for c in conns:
                        direction = "to" if c["from_location_id"] == record_id else "from"
                        other_id = c["to_location_id"] if direction == "to" else c["from_location_id"]
                        secret = " [SECRET]" if c.get("is_secret") else ""
                        desc = f" — {c['description']}" if c.get("description") else ""
                        result += f"\n    {c['connection_type']} {direction} location_id={other_id}{desc}{secret}"
                children = self._get(f"{self._base('locations')}/{record_id}/children")
                if children:
                    result += "\n  Sub-locations:"
                    for child in children:
                        result += f"\n    - {child['name']} (id={child['id']}, {child['location_type']})"
                return result

            elif resource_type == CampaignResource.NPCS:
                npc = self._get(f"{self._base('npcs')}/{record_id}")
                if not npc:
                    return f"NPC {record_id} not found."
                return self._fmt_npc(npc, detailed=True)

            elif resource_type == CampaignResource.PLAYER_CHARACTERS:
                pc = self._get(f"{self._base('player-characters')}/{record_id}")
                if not pc:
                    return f"Player character {record_id} not found."
                return self._fmt_pc(pc)

            elif resource_type == CampaignResource.WORLD_LORE:
                entry = self._get(f"{self._base('world-lore')}/{record_id}")
                if not entry:
                    return f"World lore {record_id} not found."
                return self._fmt_world_lore(entry)

            elif resource_type == CampaignResource.NOTES:
                note = self._get(f"{self._base('notes')}/{record_id}")
                if not note:
                    return f"Note {record_id} not found."
                return self._fmt_note(note)

            elif resource_type == CampaignResource.SESSIONS:
                session = self._get(f"{self._base('sessions')}/{record_id}")
                if not session:
                    return f"Session {record_id} not found."
                return self._fmt_session(session)

            else:
                return f"Unknown resource type: {resource_type}"
        except Exception as e:
            return f"Error fetching {resource_type} #{record_id}: {e}"

    # ------------------------------------------------------------------
    # get_read_tools — 2 retrieval tools for the game master
    # ------------------------------------------------------------------

    def get_read_tools(self) -> list[FunctionTool]:
        """
        Returns 2 retrieval tools for the AI Game Master:
          1. SearchCampaignDatabase — search by keyword + resource type enum
          2. GetCampaignRecord     — get full details by resource type + ID
        """
        tk = self

        class SearchCampaignDatabase(BaseModel):
            """Search the campaign database for information about the game world.
            Use this to find NPCs, locations, world lore, notes, player characters,
            or session records by keyword. Pick the resource type that matches
            what you are looking for."""

            resource_type: CampaignResource = Field(
                ...,
                description=(
                    "What to search: "
                    "'locations' for places, "
                    "'npcs' for non-player characters, "
                    "'player_characters' for PC sheets, "
                    "'world_lore' for setting/background knowledge, "
                    "'notes' for GM notes, "
                    "'sessions' for game session records."
                ),
            )
            query: str = Field(
                ...,
                description="Search term — matches names, titles, and content.",
            )

            def run(self) -> str:
                return tk.search(self.resource_type, self.query)

        class GetCampaignRecord(BaseModel):
            """Get the full details of a specific record from the campaign database.
            Use this after searching, when you need the complete information for a
            record whose ID you already know."""

            resource_type: CampaignResource = Field(
                ..., description="The type of record to retrieve.",
            )
            record_id: int = Field(
                ..., description="The record's database ID (from search results).",
            )

            def run(self) -> str:
                return tk.get_details(self.resource_type, self.record_id)

        return [
            FunctionTool(SearchCampaignDatabase),
            FunctionTool(GetCampaignRecord),
        ]

    # ------------------------------------------------------------------
    # get_write_tools — write tools for the save/update flow
    # ------------------------------------------------------------------

    def get_write_tools(self) -> list[FunctionTool]:
        """
        Returns write tools for creating/updating campaign records.
        Intended for the VGM's generate_save_state() flow or any
        other agent that needs to update the campaign database.
        """
        tk = self

        class CreateNPC(BaseModel):
            """Create a new NPC encountered during gameplay."""
            name: str = Field(..., description="NPC name.")
            title: Optional[str] = Field(None, description="Title or role.")
            description: Optional[str] = Field(None, description="Brief description.")
            appearance: Optional[str] = Field(None, description="Physical appearance.")
            personality: Optional[str] = Field(None, description="Personality traits.")
            motivations: Optional[str] = Field(None, description="Goals and motivations.")
            secrets: Optional[str] = Field(None, description="Hidden information (GM only).")
            status: str = Field("alive", description="Status: alive, dead, missing, unknown.")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    npc = tk._post(tk._base("npcs") + "/", body)
                    return f"Created NPC: {npc['name']} (id={npc['id']})"
                except Exception as e:
                    return f"Error: {e}"

        class UpdateNPC(BaseModel):
            """Update an existing NPC. Only provide fields to change."""
            npc_id: int = Field(..., description="NPC database ID.")
            name: Optional[str] = None
            title: Optional[str] = None
            description: Optional[str] = None
            personality: Optional[str] = None
            motivations: Optional[str] = None
            secrets: Optional[str] = None
            status: Optional[str] = None

            def run(self) -> str:
                fields = self.model_dump(exclude={"npc_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    npc = tk._patch(f"{tk._base('npcs')}/{self.npc_id}", fields)
                    return f"Updated NPC: {npc['name']} (id={npc['id']})"
                except Exception as e:
                    return f"Error: {e}"

        class CreateLocation(BaseModel):
            """Record a new location discovered during gameplay."""
            name: str = Field(..., description="Location name.")
            location_type: str = Field("other", description="Type: region, city, district, building, room, landmark, wilderness, other.")
            parent_id: Optional[int] = Field(None, description="Parent location ID.")
            description: Optional[str] = None
            secrets: Optional[str] = None

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    loc = tk._post(tk._base("locations") + "/", body)
                    return f"Created location: {loc['name']} (id={loc['id']})"
                except Exception as e:
                    return f"Error: {e}"

        class UpdateLocation(BaseModel):
            """Update an existing location."""
            location_id: int = Field(..., description="Location database ID.")
            description: Optional[str] = None
            secrets: Optional[str] = None

            def run(self) -> str:
                fields = self.model_dump(exclude={"location_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    loc = tk._patch(f"{tk._base('locations')}/{self.location_id}", fields)
                    return f"Updated location: {loc['name']} (id={loc['id']})"
                except Exception as e:
                    return f"Error: {e}"

        class CreateNote(BaseModel):
            """Create a note about something that happened in the game."""
            target_type: str = Field(..., description="Entity type: campaign, session, location, npc, player_character.")
            target_id: int = Field(..., description="ID of the entity.")
            content: str = Field(..., description="Note content.")
            title: Optional[str] = None
            is_secret: bool = Field(False, description="GM-only note?")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    note = tk._post(tk._base("notes") + "/", body)
                    return f"Created note (id={note['id']})"
                except Exception as e:
                    return f"Error: {e}"

        class UpdateSessionSummary(BaseModel):
            """Update a game session's summary or status."""
            session_id: int = Field(..., description="Session database ID.")
            summary: Optional[str] = None
            status: Optional[str] = None

            def run(self) -> str:
                fields = self.model_dump(exclude={"session_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    s = tk._patch(f"{tk._base('sessions')}/{self.session_id}", fields)
                    return f"Updated session #{s['session_number']}"
                except Exception as e:
                    return f"Error: {e}"

        class CreateWorldLore(BaseModel):
            """Add new world knowledge discovered or established during gameplay."""
            topic: str = Field(..., description="Topic name.")
            content: str = Field(..., description="Article content.")
            category: Optional[str] = None
            is_secret: bool = False

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    entry = tk._post(tk._base("world-lore") + "/", body)
                    return f"Created world lore: '{entry['topic']}' (id={entry['id']})"
                except Exception as e:
                    return f"Error: {e}"

        class SearchDatabase(BaseModel):
            """Search the database to check if a record already exists before creating."""
            resource_type: CampaignResource = Field(..., description="What to search.")
            query: str = Field(..., description="Search term.")

            def run(self) -> str:
                return tk.search(self.resource_type, self.query)

        return [
            FunctionTool(cls) for cls in [
                CreateNPC, UpdateNPC,
                CreateLocation, UpdateLocation,
                CreateNote, UpdateSessionSummary,
                CreateWorldLore, SearchDatabase,
            ]
        ]

    # ------------------------------------------------------------------
    # YAML Importer
    # ------------------------------------------------------------------

    def import_world_lore_from_yaml(
        self,
        yaml_path: str,
        category: str | None = None,
    ) -> list[dict]:
        """
        Import ``game_world_information`` from a scenario YAML file
        into the WorldLore table for the current campaign.
        """
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        world_info = data.get("game_world_information", {})
        if not world_info:
            print(f"  No 'game_world_information' key found in {yaml_path}")
            return []

        created = []
        for topic, sections in world_info.items():
            content = self._render_lore_content(topic, sections)
            body: dict = {"topic": topic, "content": content}
            if category:
                body["category"] = category
            try:
                entry = self._post(self._base("world-lore") + "/", body)
                created.append(entry)
                print(f"  + {topic} (id={entry['id']})")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print(f"  ~ {topic} (already exists, skipped)")
                else:
                    print(f"  ! {topic} — error: {e}")
            except Exception as e:
                print(f"  ! {topic} — error: {e}")

        print(f"  Imported {len(created)} world lore articles.")
        return created

    @staticmethod
    def _render_lore_content(topic: str, sections) -> str:
        """Render a YAML world-info entry into readable markdown."""
        if isinstance(sections, str):
            return sections.strip()
        if not isinstance(sections, dict):
            return str(sections)

        lines = [f"# {topic}", ""]
        for section_name, section_body in sections.items():
            lines.append(f"## {section_name}")
            if isinstance(section_body, str):
                lines.append(section_body.strip())
            elif isinstance(section_body, list):
                for item in section_body:
                    lines.append(f"- {item}")
            elif isinstance(section_body, dict):
                for k, v in section_body.items():
                    if isinstance(v, list):
                        lines.append(f"**{k}:**")
                        for item in v:
                            lines.append(f"- {item}")
                    else:
                        lines.append(f"**{k}:** {v}")
            lines.append("")
        return "\n".join(lines).strip()

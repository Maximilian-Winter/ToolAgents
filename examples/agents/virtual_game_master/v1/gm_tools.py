"""
gm_tools.py — GM Tool Integration for Virtual Game Master
==========================================================

Provides FunctionTool-compatible tools that let the AI Game Master
(or any ToolAgents agent) query and update the GM Tool campaign database
during gameplay.

Requires the GM Tool FastAPI server to be running:
    cd gm_tool && uvicorn app:app --port 8000

Usage:
    from gm_tools import GMToolkit

    toolkit = GMToolkit(base_url="http://localhost:8000", campaign_id=1)
    tools = toolkit.get_tools()  # list[FunctionTool]

    # Register with a ToolRegistry or pass to create_harness(tools=tools)
"""

from __future__ import annotations

import json
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool


class GMToolkit:
    """
    Creates FunctionTools backed by the GM Tool REST API.

    Each tool uses synchronous httpx so it works in both sync CLI
    and threaded contexts. The GM Tool server must be running.
    """

    def __init__(self, base_url: str = "http://localhost:8000", campaign_id: int = 1):
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
    # Formatters — turn API responses into LLM-readable text
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_location(loc: dict) -> str:
        lines = [f"**{loc['name']}** (id={loc['id']}, type={loc['location_type']})"]
        if loc.get("path"):
            lines.append(f"  Path: {loc['path']}")
        if loc.get("description"):
            lines.append(f"  {loc['description']}")
        if loc.get("secrets"):
            lines.append(f"  [SECRET] {loc['secrets']}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_npc(npc: dict) -> str:
        status = npc.get("status", "unknown").upper()
        lines = [f"**{npc['name']}** (id={npc['id']}, {status})"]
        if npc.get("title"):
            lines[0] += f" — {npc['title']}"
        for field in ("description", "appearance", "personality", "motivations"):
            if npc.get(field):
                lines.append(f"  {field.title()}: {npc[field]}")
        if npc.get("secrets"):
            lines.append(f"  [SECRET] {npc['secrets']}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_npc_full(npc: dict) -> str:
        lines = [GMToolkit._fmt_npc(npc)]
        if npc.get("location_associations"):
            lines.append("  Locations:")
            for assoc in npc["location_associations"]:
                role = f" ({assoc['role']})" if assoc.get("role") else ""
                lines.append(f"    - location_id={assoc['location_id']}{role}")
        if npc.get("relationships"):
            lines.append("  Relationships:")
            for rel in npc["relationships"]:
                secret = " [SECRET]" if rel.get("is_secret") else ""
                desc = f" — {rel['description']}" if rel.get("description") else ""
                if rel["from_npc_id"] == npc["id"]:
                    lines.append(f"    -> npc_id={rel['to_npc_id']}: {rel['relationship_type']}{desc}{secret}")
                else:
                    lines.append(f"    <- npc_id={rel['from_npc_id']}: {rel['relationship_type']}{desc}{secret}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_pc(pc: dict) -> str:
        lines = [f"**{pc['name']}** (id={pc['id']}, Lv{pc.get('level', '?')})"]
        if pc.get("player_name"):
            lines[0] += f" — played by {pc['player_name']}"
        for field in ("race", "character_class", "subclass", "background"):
            if pc.get(field):
                lines.append(f"  {field.replace('_', ' ').title()}: {pc[field]}")
        for field in ("description", "appearance", "personality", "backstory"):
            if pc.get(field):
                lines.append(f"  {field.title()}: {pc[field]}")
        if pc.get("companions"):
            lines.append("  Companions:")
            for c in pc["companions"]:
                active = "" if c.get("is_active", True) else " (inactive)"
                lines.append(f"    - {c['name']}{active}")
        if pc.get("inventory_items"):
            lines.append("  Inventory:")
            for item in pc["inventory_items"]:
                qty = f" x{item['quantity']}" if item.get("quantity", 1) > 1 else ""
                lines.append(f"    - {item['name']}{qty}")
        if pc.get("special_items"):
            lines.append("  Special Items:")
            for item in pc["special_items"]:
                equipped = " [EQUIPPED]" if item.get("is_equipped") else ""
                lines.append(f"    - {item['name']}{equipped}")
                if item.get("properties"):
                    lines.append(f"      Properties: {item['properties']}")
        if pc.get("relationships"):
            lines.append("  Relationships:")
            for rel in pc["relationships"]:
                rtype = f" ({rel['relationship_type']})" if rel.get("relationship_type") else ""
                lines.append(f"    - {rel['target_name']}{rtype}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_location_tree(nodes: list, indent: int = 0) -> str:
        lines = []
        prefix = "  " * indent
        for node in nodes:
            npc_info = ""
            if node.get("npcs"):
                names = [f"{n['name']} ({n['status']})" for n in node["npcs"]]
                npc_info = f"  [NPCs: {', '.join(names)}]"
            lines.append(f"{prefix}- {node['name']} ({node['location_type']}){npc_info}")
            if node.get("children"):
                lines.append(GMToolkit._fmt_location_tree(node["children"], indent + 1))
        return "\n".join(lines)

    @staticmethod
    def _fmt_world_lore(entry: dict) -> str:
        cat = f" [{entry['category']}]" if entry.get("category") else ""
        secret = " [SECRET]" if entry.get("is_secret") else ""
        lines = [f"**{entry['topic']}**{cat}{secret} (id={entry['id']})"]
        lines.append(entry["content"])
        return "\n".join(lines)

    @staticmethod
    def _fmt_note(note: dict) -> str:
        title = note.get("title") or "(untitled)"
        secret = " [SECRET]" if note.get("is_secret") else ""
        lines = [f"**{title}** (id={note['id']}, on {note['target_type']} #{note['target_id']}){secret}"]
        lines.append(f"  {note['content']}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_session(session: dict) -> str:
        lines = [f"**Session #{session['session_number']}** (id={session['id']}, {session['status']})"]
        if session.get("title"):
            lines[0] += f" — {session['title']}"
        if session.get("scheduled_date"):
            lines.append(f"  Scheduled: {session['scheduled_date']}")
        if session.get("summary"):
            lines.append(f"  Summary: {session['summary']}")
        if session.get("prep_notes"):
            lines.append(f"  Prep Notes: {session['prep_notes']}")
        if session.get("notes"):
            lines.append(f"  Notes ({len(session['notes'])}):")
            for note in session["notes"]:
                lines.append(f"    - {note.get('title', '(untitled)')}: {note['content'][:100]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # YAML Importer — seed world lore from scenario files
    # ------------------------------------------------------------------

    def import_world_lore_from_yaml(
        self,
        yaml_path: str,
        category: str | None = None,
    ) -> list[dict]:
        """
        Import ``game_world_information`` from a scenario YAML file
        into the WorldLore table for the current campaign.

        Each top-level key under ``game_world_information`` becomes one
        lore article.  Nested dicts/lists are rendered as readable markdown.

        Args:
            yaml_path: Path to the scenario YAML file.
            category:  Optional category to assign to all imported entries.

        Returns:
            List of created lore entries (as dicts from the API).
        """
        import yaml  # local import — only needed for this method

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

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def get_tools(self) -> list[FunctionTool]:
        """Build and return all GM Tool FunctionTools for this campaign."""
        tk = self  # capture for tool closures

        # ==============================================================
        # READ TOOLS — World Reference
        # ==============================================================

        class SearchLocations(BaseModel):
            """Search for locations in the campaign world by name.
            Returns matching locations with their paths and descriptions."""

            query: str = Field(..., description="Search term for location name.")

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("locations") + "/search", {"q": self.query})
                except Exception as e:
                    return f"Error searching locations: {e}"
                if not data:
                    return f"No locations found matching '{self.query}'."
                return "\n\n".join(tk._fmt_location(loc) for loc in data)

        class GetLocationTree(BaseModel):
            """Get the full hierarchical tree of all locations in the campaign.
            Shows the world structure with nested locations and NPCs at each."""

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("locations") + "/tree")
                except Exception as e:
                    return f"Error getting location tree: {e}"
                if not data:
                    return "No locations in this campaign yet."
                return tk._fmt_location_tree(data)

        class GetLocationDetails(BaseModel):
            """Get full details of a specific location by ID, including
            description, secrets, and connected locations."""

            location_id: int = Field(..., description="The location's database ID.")

            def run(self) -> str:
                try:
                    loc = tk._get(f"{tk._base('locations')}/{self.location_id}")
                except Exception as e:
                    return f"Error: {e}"
                if not loc:
                    return f"Location {self.location_id} not found."
                result = tk._fmt_location(loc)
                # Also fetch connections
                try:
                    conns = tk._get(f"{tk._base('locations')}/{self.location_id}/connections")
                    if conns:
                        result += "\n  Connections:"
                        for c in conns:
                            direction = "to" if c["from_location_id"] == self.location_id else "from"
                            other_id = c["to_location_id"] if direction == "to" else c["from_location_id"]
                            secret = " [SECRET]" if c.get("is_secret") else ""
                            result += f"\n    {c['connection_type']} {direction} location_id={other_id}{secret}"
                            if c.get("description"):
                                result += f" — {c['description']}"
                except Exception:
                    pass
                # Also fetch children
                try:
                    children = tk._get(f"{tk._base('locations')}/{self.location_id}/children")
                    if children:
                        result += "\n  Sub-locations:"
                        for child in children:
                            result += f"\n    - {child['name']} (id={child['id']}, {child['location_type']})"
                except Exception:
                    pass
                return result

        class ResolveLocationPath(BaseModel):
            """Resolve a slash-separated location path to its details.
            Example: 'sword-coast/waterdeep/castle-ward'"""

            path: str = Field(..., description="Slash-separated slug path (e.g. 'region/city/district').")

            def run(self) -> str:
                try:
                    loc = tk._get(tk._base("locations") + "/resolve", {"path": self.path})
                except Exception as e:
                    return f"Error: {e}"
                if not loc:
                    return f"No location found at path '{self.path}'."
                return tk._fmt_location(loc)

        class SearchNPCs(BaseModel):
            """Search for NPCs (Non-Player Characters) by name."""

            query: str = Field(..., description="Search term for NPC name.")

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("npcs") + "/search", {"q": self.query})
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return f"No NPCs found matching '{self.query}'."
                return "\n\n".join(tk._fmt_npc(npc) for npc in data)

        class GetNPCDetails(BaseModel):
            """Get full details of an NPC including personality, motivations,
            secrets, location associations, and relationships with other NPCs."""

            npc_id: int = Field(..., description="The NPC's database ID.")

            def run(self) -> str:
                try:
                    npc = tk._get(f"{tk._base('npcs')}/{self.npc_id}")
                except Exception as e:
                    return f"Error: {e}"
                if not npc:
                    return f"NPC {self.npc_id} not found."
                return tk._fmt_npc_full(npc)

        class ListNPCsAtLocation(BaseModel):
            """List all NPCs associated with a specific location."""

            location_id: int = Field(..., description="The location's database ID.")

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("npcs") + "/", {"location_id": self.location_id})
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return f"No NPCs found at location {self.location_id}."
                return "\n\n".join(tk._fmt_npc(npc) for npc in data)

        class GetPlayerCharacter(BaseModel):
            """Get full details of a player character including companions,
            inventory, special items, and relationships."""

            pc_id: int = Field(..., description="The player character's database ID.")

            def run(self) -> str:
                try:
                    pc = tk._get(f"{tk._base('player-characters')}/{self.pc_id}")
                except Exception as e:
                    return f"Error: {e}"
                if not pc:
                    return f"Player character {self.pc_id} not found."
                return tk._fmt_pc(pc)

        class ListPlayerCharacters(BaseModel):
            """List all player characters in the campaign."""

            active_only: bool = Field(True, description="If true, only show active PCs.")

            def run(self) -> str:
                params = {}
                if self.active_only:
                    params["active_only"] = True
                try:
                    data = tk._get(tk._base("player-characters") + "/", params or None)
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return "No player characters in this campaign."
                lines = []
                for pc in data:
                    cls = pc.get("character_class", "?")
                    race = pc.get("race", "?")
                    lines.append(f"- [{pc['id']}] {pc['name']} — {race} {cls} Lv{pc.get('level', '?')}")
                return "\n".join(lines)

        class SearchNotes(BaseModel):
            """Search GM notes by title and content."""

            query: str = Field(..., description="Search term.")

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("notes") + "/search", {"q": self.query})
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return f"No notes found matching '{self.query}'."
                return "\n\n".join(tk._fmt_note(n) for n in data)

        class GetSessionInfo(BaseModel):
            """Get details of a game session including summary, prep notes,
            and associated notes."""

            session_id: int = Field(..., description="The session's database ID.")

            def run(self) -> str:
                try:
                    session = tk._get(f"{tk._base('sessions')}/{self.session_id}")
                except Exception as e:
                    return f"Error: {e}"
                if not session:
                    return f"Session {self.session_id} not found."
                return tk._fmt_session(session)

        class ListSessions(BaseModel):
            """List all game sessions in the campaign, optionally filtered by status."""

            status: Optional[str] = Field(
                None,
                description="Filter by status: 'planned', 'in_progress', 'completed', 'cancelled'.",
            )

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("sessions") + "/", {"status": self.status})
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return "No sessions found."
                lines = []
                for s in data:
                    title = f" — {s['title']}" if s.get("title") else ""
                    lines.append(f"- [#{s['session_number']}] (id={s['id']}, {s['status']}){title}")
                return "\n".join(lines)

        # ==============================================================
        # WRITE TOOLS — Update the World
        # ==============================================================

        class CreateNPC(BaseModel):
            """Create a new NPC in the campaign world. Use when a new character
            is introduced during gameplay that should be tracked."""

            name: str = Field(..., description="NPC name.")
            title: Optional[str] = Field(None, description="Title or role (e.g. 'Captain of the Guard').")
            description: Optional[str] = Field(None, description="Brief description.")
            appearance: Optional[str] = Field(None, description="Physical appearance.")
            personality: Optional[str] = Field(None, description="Personality traits.")
            motivations: Optional[str] = Field(None, description="Goals and motivations.")
            secrets: Optional[str] = Field(None, description="Hidden information (GM only).")
            status: str = Field("alive", description="Status: alive, dead, missing, unknown.")
            primary_location_id: Optional[int] = Field(None, description="ID of their primary location.")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    npc = tk._post(tk._base("npcs") + "/", body)
                    return f"Created NPC: {npc['name']} (id={npc['id']})"
                except Exception as e:
                    return f"Error creating NPC: {e}"

        class UpdateNPC(BaseModel):
            """Update an existing NPC's information. Only provide fields you want to change."""

            npc_id: int = Field(..., description="The NPC's database ID.")
            name: Optional[str] = Field(None, description="New name.")
            title: Optional[str] = Field(None, description="New title.")
            description: Optional[str] = Field(None, description="Updated description.")
            appearance: Optional[str] = Field(None, description="Updated appearance.")
            personality: Optional[str] = Field(None, description="Updated personality.")
            motivations: Optional[str] = Field(None, description="Updated motivations.")
            secrets: Optional[str] = Field(None, description="Updated secrets.")
            status: Optional[str] = Field(None, description="New status: alive, dead, missing, unknown.")

            def run(self) -> str:
                fields = self.model_dump(exclude={"npc_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    npc = tk._patch(f"{tk._base('npcs')}/{self.npc_id}", fields)
                    return f"Updated NPC: {npc['name']} (id={npc['id']})"
                except Exception as e:
                    return f"Error updating NPC: {e}"

        class CreateLocation(BaseModel):
            """Create a new location in the campaign world. Locations can be
            nested under a parent to build a hierarchy."""

            name: str = Field(..., description="Location name.")
            location_type: str = Field(
                "other",
                description="Type: region, city, district, building, room, landmark, wilderness, other.",
            )
            parent_id: Optional[int] = Field(
                None, description="ID of the parent location (for nesting)."
            )
            description: Optional[str] = Field(None, description="Location description.")
            secrets: Optional[str] = Field(None, description="Hidden info (GM only).")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    loc = tk._post(tk._base("locations") + "/", body)
                    return f"Created location: {loc['name']} (id={loc['id']}, path={loc.get('path', '')})"
                except Exception as e:
                    return f"Error creating location: {e}"

        class UpdateLocation(BaseModel):
            """Update an existing location's information."""

            location_id: int = Field(..., description="The location's database ID.")
            name: Optional[str] = Field(None, description="New name.")
            description: Optional[str] = Field(None, description="Updated description.")
            secrets: Optional[str] = Field(None, description="Updated secrets.")
            location_type: Optional[str] = Field(None, description="New type.")

            def run(self) -> str:
                fields = self.model_dump(exclude={"location_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    loc = tk._patch(f"{tk._base('locations')}/{self.location_id}", fields)
                    return f"Updated location: {loc['name']} (id={loc['id']})"
                except Exception as e:
                    return f"Error updating location: {e}"

        class CreateNote(BaseModel):
            """Create a GM note attached to a campaign entity. Use during gameplay
            to record important events, discoveries, or decisions."""

            target_type: str = Field(
                ...,
                description="What to attach the note to: 'campaign', 'session', 'location', 'npc', 'player_character'.",
            )
            target_id: int = Field(..., description="ID of the target entity.")
            content: str = Field(..., description="Note content.")
            title: Optional[str] = Field(None, description="Short note title.")
            session_id: Optional[int] = Field(
                None, description="Link to a session (optional)."
            )
            is_secret: bool = Field(False, description="Mark as GM-only secret note.")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    note = tk._post(tk._base("notes") + "/", body)
                    return f"Created note: '{note.get('title', '(untitled)')}' (id={note['id']})"
                except Exception as e:
                    return f"Error creating note: {e}"

        class UpdateSessionSummary(BaseModel):
            """Update a game session's summary or status. Use after gameplay
            to record what happened."""

            session_id: int = Field(..., description="The session's database ID.")
            summary: Optional[str] = Field(None, description="Session summary text.")
            status: Optional[str] = Field(
                None,
                description="New status: planned, in_progress, completed, cancelled.",
            )
            prep_notes: Optional[str] = Field(None, description="Updated prep notes.")

            def run(self) -> str:
                fields = self.model_dump(exclude={"session_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    session = tk._patch(f"{tk._base('sessions')}/{self.session_id}", fields)
                    return f"Updated session #{session['session_number']} (id={session['id']})"
                except Exception as e:
                    return f"Error updating session: {e}"

        class CreateSession(BaseModel):
            """Create a new game session entry for tracking."""

            session_number: int = Field(..., description="Session number (must be unique in campaign).")
            title: Optional[str] = Field(None, description="Session title.")
            status: str = Field("planned", description="Status: planned, in_progress, completed, cancelled.")
            prep_notes: Optional[str] = Field(None, description="GM prep notes for this session.")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    session = tk._post(tk._base("sessions") + "/", body)
                    return f"Created session #{session['session_number']} (id={session['id']})"
                except Exception as e:
                    return f"Error creating session: {e}"

        class AddNPCRelationship(BaseModel):
            """Record a relationship between two NPCs."""

            from_npc_id: int = Field(..., description="Source NPC ID.")
            to_npc_id: int = Field(..., description="Target NPC ID.")
            relationship_type: str = Field(
                "other",
                description="Type: ally, rival, enemy, family, friend, employer, employee, mentor, student, lover, contact, other.",
            )
            description: Optional[str] = Field(None, description="Description of the relationship.")
            is_secret: bool = Field(False, description="Is this relationship secret?")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    rel = tk._post(f"{tk._base('npcs')}/{self.from_npc_id}/relationships", body)
                    return f"Created relationship: NPC {self.from_npc_id} -> NPC {self.to_npc_id} ({self.relationship_type})"
                except Exception as e:
                    return f"Error: {e}"

        class ConnectLocations(BaseModel):
            """Create a connection between two locations (road, tunnel, portal, etc.)."""

            from_location_id: int = Field(..., description="Source location ID.")
            to_location_id: int = Field(..., description="Destination location ID.")
            connection_type: str = Field(
                "road",
                description="Type: road, river, tunnel, portal, sea_route, path, secret, other.",
            )
            description: Optional[str] = Field(None, description="Description of the connection.")
            is_bidirectional: bool = Field(True, description="Can travel both ways?")
            is_secret: bool = Field(False, description="Is this connection hidden?")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    conn = tk._post(
                        f"{tk._base('locations')}/{self.from_location_id}/connections",
                        body,
                    )
                    direction = "bidirectional" if self.is_bidirectional else "one-way"
                    return f"Connected locations {self.from_location_id} <-> {self.to_location_id} ({self.connection_type}, {direction})"
                except Exception as e:
                    return f"Error: {e}"

        # ==============================================================
        # WORLD LORE TOOLS — Structured world knowledge
        # ==============================================================

        class SearchWorldLore(BaseModel):
            """Search world lore articles by topic name or content.
            Use to find background information about the game world —
            geography, magic, factions, history, threats, etc."""

            query: str = Field(..., description="Search term (matches topic names and content).")

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("world-lore") + "/search", {"q": self.query})
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return f"No world lore found matching '{self.query}'."
                return "\n\n".join(tk._fmt_world_lore(entry) for entry in data)

        class GetWorldLore(BaseModel):
            """Get a specific world lore article by ID. Returns the full content."""

            lore_id: int = Field(..., description="The world lore article's database ID.")

            def run(self) -> str:
                try:
                    entry = tk._get(f"{tk._base('world-lore')}/{self.lore_id}")
                except Exception as e:
                    return f"Error: {e}"
                if not entry:
                    return f"World lore {self.lore_id} not found."
                return tk._fmt_world_lore(entry)

        class ListWorldLore(BaseModel):
            """List all world lore topics in the campaign.
            Optionally filter by category."""

            category: Optional[str] = Field(
                None, description="Filter by category (e.g. 'geography', 'magic', 'factions')."
            )

            def run(self) -> str:
                try:
                    data = tk._get(tk._base("world-lore") + "/", {"category": self.category})
                except Exception as e:
                    return f"Error: {e}"
                if not data:
                    return "No world lore articles in this campaign."
                lines = []
                for entry in data:
                    cat = f" [{entry['category']}]" if entry.get("category") else ""
                    secret = " [SECRET]" if entry.get("is_secret") else ""
                    lines.append(f"- [{entry['id']}] {entry['topic']}{cat}{secret}")
                return "\n".join(lines)

        class CreateWorldLore(BaseModel):
            """Create a new world lore article. Use to add background knowledge
            about the game world — places, magic systems, factions, history, etc."""

            topic: str = Field(..., description="Topic name (e.g. 'The Weave', 'Cult of the Dragon').")
            content: str = Field(..., description="Full article content (markdown supported).")
            category: Optional[str] = Field(
                None, description="Category for organization (e.g. 'geography', 'magic', 'factions', 'threats')."
            )
            is_secret: bool = Field(False, description="Mark as GM-only secret lore.")

            def run(self) -> str:
                body = self.model_dump(exclude_none=True)
                try:
                    entry = tk._post(tk._base("world-lore") + "/", body)
                    return f"Created world lore: '{entry['topic']}' (id={entry['id']})"
                except Exception as e:
                    return f"Error creating world lore: {e}"

        class UpdateWorldLore(BaseModel):
            """Update an existing world lore article. Only provide fields to change."""

            lore_id: int = Field(..., description="The world lore article's database ID.")
            topic: Optional[str] = Field(None, description="New topic name.")
            content: Optional[str] = Field(None, description="Updated content.")
            category: Optional[str] = Field(None, description="Updated category.")
            is_secret: Optional[bool] = Field(None, description="Update secret status.")

            def run(self) -> str:
                fields = self.model_dump(exclude={"lore_id"}, exclude_none=True)
                if not fields:
                    return "No fields to update."
                try:
                    entry = tk._patch(f"{tk._base('world-lore')}/{self.lore_id}", fields)
                    return f"Updated world lore: '{entry['topic']}' (id={entry['id']})"
                except Exception as e:
                    return f"Error updating world lore: {e}"

        # ==============================================================
        # Assemble and return
        # ==============================================================

        tool_classes = [
            # Read
            SearchLocations,
            GetLocationTree,
            GetLocationDetails,
            ResolveLocationPath,
            SearchNPCs,
            GetNPCDetails,
            ListNPCsAtLocation,
            GetPlayerCharacter,
            ListPlayerCharacters,
            SearchNotes,
            GetSessionInfo,
            ListSessions,
            SearchWorldLore,
            GetWorldLore,
            ListWorldLore,
            # Write
            CreateNPC,
            UpdateNPC,
            CreateLocation,
            UpdateLocation,
            CreateNote,
            UpdateSessionSummary,
            CreateSession,
            AddNPCRelationship,
            ConnectLocations,
            CreateWorldLore,
            UpdateWorldLore,
        ]

        return [FunctionTool(cls) for cls in tool_classes]

"""
=============================================================================
example_world_engine.py — Virtual Game Master with Living World
=============================================================================

A standalone example showing how to build a virtual game master using:
  - ToolAgents (ChatToolAgent, ToolRegistry, FunctionTool)
  - PromptComposer         → dynamic system prompt with auto-loaded location context
  - SmartMessageManager    → conversation with archival for long sessions
  - Agora Knowledge Base   → hierarchical world data (locations, NPCs, lore)

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                       Agora KB (World Database)                 │
  │                                                                 │
  │  worlds/baldurs-gate/                                           │
  │    overview.md            ← city lore, factions, mood           │
  │    upper-city/                                                  │
  │      overview.md                                                │
  │      high-hall.md                                               │
  │    lower-city/                                                  │
  │      overview.md                                                │
  │      market.md            ← current location                   │
  │      elfsong-tavern.md                                          │
  │    undercity/                                                   │
  │      overview.md                                                │
  │                                                                 │
  │  Query: "worlds/baldurs-gate/lower-city/"                       │
  │    → returns: [market.md, elfsong-tavern.md, overview.md]       │
  │  Query: "worlds/baldurs-gate/lower-city/market.md"              │
  │    → returns: full location document                            │
  └─────────────────────┬───────────────────────────────────────────┘
                        │ AgoraKBClient (REST)
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                     WorldContextManager                         │
  │                                                                 │
  │  Tracks: current_location_path                                  │
  │  On location change:                                            │
  │    1. Summarize what happened at old location (LLM call)        │
  │    2. Update old location doc in KB with event summary           │
  │    3. Fetch new location doc + parent overview                  │
  │    4. Inject into PromptComposer as dynamic module              │
  └─────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      PromptComposer                             │
  │  ┌──────────┬────────────┬───────────┬──────────┬────────────┐ │
  │  │ GM       │ World      │ Location  │ Party    │ Game       │ │
  │  │ Instruct │ Overview   │ Context   │ State    │ Rules      │ │
  │  │ (static) │ (dynamic)  │ (dynamic) │ (dynamic)│ (static)   │ │
  │  │ pos=0    │ pos=5      │ pos=10    │ pos=15   │ pos=20     │ │
  │  └──────────┴────────────┴───────────┴──────────┴────────────┘ │
  └──────────────────────┬──────────────────────────────────────────┘
                         │ compile() each turn
                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  SmartMessageManager                                            │
  │  [pinned: safety] [archival: old turns] [normal: recent turns]  │
  └──────────────────────┬──────────────────────────────────────────┘
                         │ system_prompt + active_messages
                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ChatToolAgent.get_response()                                   │
  │  Tools: set_location, list_locations, search_world,             │
  │         update_location_notes, read_document                    │
  └─────────────────────────────────────────────────────────────────┘

Usage:
  1. Start Agora server:  python -m agora.runner
  2. Create a project:    (see seed_world() below or use the dashboard)
  3. Run this example:    python example_world_engine.py
"""

import json
import os
from datetime import datetime
from copy import copy
from typing import Optional

import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── ToolAgents imports ──────────────────────────────────────────────
from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.agent_harness.prompt_composer import PromptComposer
from ToolAgents.agent_harness.smart_messages import (
    SmartMessageManager,
    MessageLifecycle,
    ExpiryAction,
)
from ToolAgents.provider import OpenAIChatAPI, GroqChatAPI

load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# PART 1: Agora KB Client — REST interface to the world database
# ═══════════════════════════════════════════════════════════════════

class AgoraKBClient:
    """
    Synchronous client for Agora's Knowledge Base REST API.
    All world data lives as documents in a project's KB.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8321", project_slug: str = "game-world"):
        self.base_url = base_url.rstrip("/")
        self.project_slug = project_slug
        self.client = httpx.Client(timeout=30.0)

    @property
    def kb_url(self) -> str:
        return f"{self.base_url}/api/projects/{self.project_slug}/kb"

    def read_document(self, path: str, section: Optional[str] = None) -> Optional[dict]:
        """
        Read a document by path.
        Optionally extract a specific section by header name.
        Returns dict with 'title', 'body', 'tags', 'path' or None.
        """
        url = f"{self.kb_url}/documents/{path}"
        params = {}
        if section:
            params["section"] = section
        try:
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            return None
        except httpx.HTTPError:
            return None

    def write_document(self, path: str, title: str, body: str, tags: str = "") -> bool:
        """Write or overwrite a document at the given path."""
        url = f"{self.kb_url}/documents"
        payload = {
            "path": path,
            "title": title,
            "body": body,
            "tags": tags,
        }
        try:
            resp = self.client.put(url, json=payload)
            return resp.status_code in (200, 201)
        except httpx.HTTPError:
            return False

    def list_documents(self, prefix: str = "") -> list[dict]:
        """
        List documents under a path prefix.
        E.g., prefix="worlds/baldurs-gate/lower-city/" returns all docs in that area.
        """
        url = f"{self.kb_url}/documents"
        params = {}
        if prefix:
            params["prefix"] = prefix
        try:
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            return []
        except httpx.HTTPError:
            return []

    def search_documents(self, query: str) -> list[dict]:
        """Full-text search across all KB documents in this project."""
        url = f"{self.kb_url}/search"
        params = {"q": query}
        try:
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            return []
        except httpx.HTTPError:
            return []

    def get_tree(self) -> dict:
        """Get the nested directory tree of all documents."""
        url = f"{self.kb_url}/tree"
        try:
            resp = self.client.get(url)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except httpx.HTTPError:
            return {}

    def close(self):
        self.client.close()


# ═══════════════════════════════════════════════════════════════════
# PART 2: World Context Manager — tracks location, auto-loads context
# ═══════════════════════════════════════════════════════════════════

class WorldContextManager:
    """
    Manages the current world state and bridges the Agora KB
    with the PromptComposer. When the location changes:
      1. Summarizes what happened at the old location (via LLM)
      2. Updates the old location's KB document with event notes
      3. Fetches the new location's document + parent overview
      4. Updates the PromptComposer's location module
    """

    def __init__(
            self,
            kb_client: AgoraKBClient,
            composer: PromptComposer,
            summarizer_agent: Optional[ChatToolAgent] = None,
            summarizer_settings=None,
            world_root: str = "worlds",
    ):
        self.kb = kb_client
        self.composer = composer
        self.summarizer_agent = summarizer_agent
        self.summarizer_settings = summarizer_settings
        self.world_root = world_root

        # Current state
        self.current_location_path: Optional[str] = None
        self.current_location_title: str = "Unknown"
        self.current_location_body: str = ""
        self.location_history: list[str] = []
        self.events_at_current_location: list[str] = []

    def get_parent_path(self, path: str) -> Optional[str]:
        """
        Given 'worlds/baldurs-gate/lower-city/market.md',
        returns 'worlds/baldurs-gate/lower-city/overview.md'
        (the parent area's overview).
        """
        parts = path.rsplit("/", 1)
        if len(parts) > 1:
            parent_dir = parts[0]
            return f"{parent_dir}/overview.md"
        return None

    def get_area_path(self, path: str) -> str:
        """
        Given 'worlds/baldurs-gate/lower-city/market.md',
        returns 'worlds/baldurs-gate/lower-city/' for listing siblings.
        """
        parts = path.rsplit("/", 1)
        if len(parts) > 1:
            return parts[0] + "/"
        return self.world_root + "/"

    def list_available_locations(self, prefix: str) -> list[dict]:
        """
        List locations under a path prefix.
        The GM uses this to discover where the party can go.
        """
        docs = self.kb.list_documents(prefix=prefix)
        return [
            {"path": d.get("path", ""), "title": d.get("title", "")}
            for d in docs
            if d.get("path", "").endswith(".md")
        ]

    def load_location(self, location_path: str) -> str:
        """
        Load a location from the KB and update the PromptComposer.
        Returns a status message.
        """
        doc = self.kb.read_document(location_path)
        if not doc:
            return f"Error: Location '{location_path}' not found in the knowledge base."

        self.current_location_path = location_path
        self.current_location_title = doc.get("title", "Unknown Location")
        self.current_location_body = doc.get("body", "")
        self.events_at_current_location = []

        # Build the location context with parent area info
        context_parts = []

        # Try to load parent overview for broader context
        parent_path = self.get_parent_path(location_path)
        if parent_path and parent_path != location_path:
            parent_doc = self.kb.read_document(parent_path)
            if parent_doc:
                context_parts.append(
                    f"## Area: {parent_doc.get('title', 'Unknown Area')}\n"
                    f"{parent_doc.get('body', '')}"
                )

        # The specific location
        context_parts.append(
            f"## Current Location: {self.current_location_title}\n"
            f"Path: {location_path}\n\n"
            f"{self.current_location_body}"
        )

        # List nearby locations (siblings in the same directory)
        area_path = self.get_area_path(location_path)
        siblings = self.list_available_locations(area_path)
        nearby = [
            s for s in siblings
            if s["path"] != location_path and not s["path"].endswith("overview.md")
        ]
        if nearby:
            nearby_list = "\n".join(f"  - {s['title']} ({s['path']})" for s in nearby)
            context_parts.append(f"\n## Nearby Locations:\n{nearby_list}")

        full_context = "\n\n---\n\n".join(context_parts)

        # Update the PromptComposer module
        self.composer.update_module(
            "location_context",
            content=full_context,
            prefix=f"### Current Location: {self.current_location_title}",
            suffix="### End Location Context",
        )

        self.location_history.append(location_path)
        return f"Arrived at: {self.current_location_title}"

    def summarize_and_depart(self, recent_messages: list[ChatMessage]) -> str:
        """
        Before leaving a location:
          1. Use the summarizer LLM to summarize what happened
          2. Append the summary to the location's KB document
        Returns the summary text.
        """
        if not self.current_location_path or not self.summarizer_agent:
            return ""

        # Build a summarization prompt from recent conversation
        conversation_text = "\n".join(
            f"{m.get_role()}: {m.get_as_text()}" for m in recent_messages[-10:]
        )

        summary_messages = [
            ChatMessage.create_system_message(
                "You are a concise note-taker for a tabletop RPG. "
                "Summarize what happened at this location in 2-4 sentences. "
                "Focus on: key events, NPC interactions, items gained/lost, "
                "consequences. Write in past tense, third person. "
                "Output ONLY the summary, nothing else."
            ),
            ChatMessage.create_user_message(
                f"Location: {self.current_location_title}\n"
                f"Path: {self.current_location_path}\n\n"
                f"Recent conversation:\n{conversation_text}\n\n"
                f"Summarize what happened here:"
            ),
        ]

        try:
            response = self.summarizer_agent.get_response(
                messages=summary_messages,
                settings=self.summarizer_settings,
                tool_registry=ToolRegistry(),  # no tools needed for summarization
            )
            summary = response.response.strip()
        except Exception as e:
            summary = f"(Summary unavailable: {e})"

        # Append the event summary to the location's KB document
        if summary and self.current_location_path:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            event_entry = f"\n\n### Event Log — {timestamp}\n{summary}"

            # Read current doc, append event log, write back
            doc = self.kb.read_document(self.current_location_path)
            if doc:
                updated_body = doc.get("body", "") + event_entry
                self.kb.write_document(
                    path=self.current_location_path,
                    title=doc.get("title", self.current_location_title),
                    body=updated_body,
                    tags=doc.get("tags", ""),
                )

        return summary

    def build_location_content_fn(self):
        """
        Returns a content_fn for the PromptComposer that always
        reflects the current location state. Used for the dynamic module.
        """

        def fn() -> str:
            if not self.current_location_path:
                return "No location loaded. Use set_location to travel somewhere."
            return (
                f"You are currently at: {self.current_location_title}\n"
                f"Path: {self.current_location_path}\n\n"
                f"{self.current_location_body}"
            )

        return fn


# ═══════════════════════════════════════════════════════════════════
# PART 3: Party State — tracks characters, inventory, quests
# ═══════════════════════════════════════════════════════════════════

class PartyState:
    """
    Tracks the adventuring party's state.
    Rendered into the system prompt via PromptComposer each turn.
    """

    def __init__(self):
        self.characters: dict[str, dict] = {}
        self.shared_inventory: list[str] = []
        self.active_quests: list[dict] = []
        self.gold: int = 0
        self.notes: list[str] = []

    def add_character(self, name: str, char_class: str, level: int = 1, hp: int = 10):
        self.characters[name] = {
            "class": char_class,
            "level": level,
            "hp": hp,
            "max_hp": hp,
            "conditions": [],
        }

    def build_context(self) -> str:
        """Render party state as text for the system prompt."""
        lines = []

        if self.characters:
            lines.append("## Party Members:")
            for name, info in self.characters.items():
                conditions = ", ".join(info["conditions"]) if info["conditions"] else "healthy"
                lines.append(
                    f"  - **{name}** — Level {info['level']} {info['class']} "
                    f"(HP: {info['hp']}/{info['max_hp']}, Status: {conditions})"
                )

        if self.gold > 0:
            lines.append(f"\n## Gold: {self.gold} gp")

        if self.shared_inventory:
            items = ", ".join(self.shared_inventory)
            lines.append(f"\n## Inventory: {items}")

        if self.active_quests:
            lines.append("\n## Active Quests:")
            for q in self.active_quests:
                status = q.get("status", "active")
                lines.append(f"  - [{status.upper()}] {q['name']}: {q.get('description', '')}")

        if self.notes:
            lines.append("\n## GM Notes:")
            for note in self.notes[-5:]:  # Keep last 5 notes
                lines.append(f"  - {note}")

        return "\n".join(lines) if lines else "No party information yet."


# ═══════════════════════════════════════════════════════════════════
# PART 4: GM Tools — what the Game Master LLM can call
# ═══════════════════════════════════════════════════════════════════

# These will be bound to instances in main()
_world_ctx: Optional[WorldContextManager] = None
_party: Optional[PartyState] = None
_kb_client: Optional[AgoraKBClient] = None
_msg_manager: Optional[SmartMessageManager] = None


class SetLocation(BaseModel):
    """
    Travel to a new location. Provide the full KB path.
    This will:
      1. Summarize events at the current location
      2. Update the current location's document with what happened
      3. Load the new location's context into the game

    Use list_locations first to discover available paths.
    """
    location_path: str = Field(
        ...,
        description=(
            "Full path to the location document in the KB. "
            "E.g., 'worlds/baldurs-gate/lower-city/market.md'"
        ),
    )

    def run(self) -> str:
        if _world_ctx is None:
            return "Error: World context not initialized."

        # Summarize and depart from current location
        if _world_ctx.current_location_path and _msg_manager:
            active = _msg_manager.get_active_messages()
            summary = _world_ctx.summarize_and_depart(active)
            if summary:
                print(f"\n  📜 [Location Update] {_world_ctx.current_location_title}: {summary}")

        # Load new location
        result = _world_ctx.load_location(self.location_path)
        return result


class ListLocations(BaseModel):
    """
    List available locations under a path prefix.
    Use this to discover where the party can travel.

    Examples:
      - "worlds/" → list all world roots
      - "worlds/baldurs-gate/" → list areas in Baldur's Gate
      - "worlds/baldurs-gate/lower-city/" → list specific locations
    """
    path_prefix: str = Field(
        ...,
        description=(
            "Path prefix to list. Use trailing slash for directories. "
            "E.g., 'worlds/baldurs-gate/lower-city/'"
        ),
    )

    def run(self) -> str:
        if _world_ctx is None:
            return "Error: World context not initialized."

        locations = _world_ctx.list_available_locations(self.path_prefix)
        if not locations:
            return f"No locations found under '{self.path_prefix}'."

        lines = [f"Locations under '{self.path_prefix}':"]
        for loc in locations:
            lines.append(f"  - {loc['title']} → {loc['path']}")
        return "\n".join(lines)


class SearchWorld(BaseModel):
    """
    Search the entire world knowledge base for a term.
    Useful for finding NPCs, items, lore, or locations by keyword.
    """
    query: str = Field(
        ..., description="Search term (e.g., 'thieves guild', 'healing potion', 'Duke Ravengard')."
    )

    def run(self) -> str:
        if _kb_client is None:
            return "Error: KB client not initialized."

        results = _kb_client.search_documents(self.query)
        if not results:
            return f"No results found for '{self.query}'."

        lines = [f"Search results for '{self.query}':"]
        for r in results[:5]:  # Limit to top 5
            title = r.get("title", "Untitled")
            path = r.get("path", "unknown")
            snippet = r.get("snippet", "")[:150]
            lines.append(f"  - [{title}] ({path}): {snippet}")
        return "\n".join(lines)


class ReadDocument(BaseModel):
    """
    Read a specific document from the world knowledge base.
    Use this to look up detailed lore, NPC info, or location descriptions.
    Optionally read just one section by header name.
    """
    path: str = Field(
        ..., description="Full document path (e.g., 'worlds/baldurs-gate/npcs/duke-ravengard.md')."
    )
    section: Optional[str] = Field(
        None, description="Optional section header to extract (e.g., 'Personality')."
    )

    def run(self) -> str:
        if _kb_client is None:
            return "Error: KB client not initialized."

        doc = _kb_client.read_document(self.path, section=self.section)
        if not doc:
            return f"Document not found: '{self.path}'"

        title = doc.get("title", "Untitled")
        body = doc.get("body", "")
        if self.section:
            return f"## {title} — Section: {self.section}\n\n{body}"
        return f"## {title}\n\n{body}"


class UpdateParty(BaseModel):
    """
    Update the party's state. Use this to track HP changes,
    inventory updates, gold, quest progress, or conditions.
    Provide only the fields you want to change.
    """
    character_name: Optional[str] = Field(None, description="Character to update.")
    hp_change: Optional[int] = Field(None, description="HP change (positive=heal, negative=damage).")
    add_condition: Optional[str] = Field(None, description="Add a condition (e.g., 'poisoned', 'blessed').")
    remove_condition: Optional[str] = Field(None, description="Remove a condition.")
    add_item: Optional[str] = Field(None, description="Add an item to shared inventory.")
    remove_item: Optional[str] = Field(None, description="Remove an item from shared inventory.")
    gold_change: Optional[int] = Field(None, description="Gold change (positive=gain, negative=spend).")
    add_quest: Optional[str] = Field(None, description="Add a new quest (format: 'name|description').")
    complete_quest: Optional[str] = Field(None, description="Mark a quest as completed by name.")
    gm_note: Optional[str] = Field(None, description="Add a private GM note about plot/events.")

    def run(self) -> str:
        if _party is None:
            return "Error: Party state not initialized."

        results = []

        if self.character_name and self.hp_change is not None:
            if self.character_name in _party.characters:
                char = _party.characters[self.character_name]
                char["hp"] = max(0, min(char["max_hp"], char["hp"] + self.hp_change))
                results.append(
                    f"{self.character_name}: HP → {char['hp']}/{char['max_hp']}"
                )

        if self.character_name and self.add_condition:
            if self.character_name in _party.characters:
                _party.characters[self.character_name]["conditions"].append(self.add_condition)
                results.append(f"{self.character_name}: +condition '{self.add_condition}'")

        if self.character_name and self.remove_condition:
            if self.character_name in _party.characters:
                conds = _party.characters[self.character_name]["conditions"]
                if self.remove_condition in conds:
                    conds.remove(self.remove_condition)
                    results.append(f"{self.character_name}: -condition '{self.remove_condition}'")

        if self.add_item:
            _party.shared_inventory.append(self.add_item)
            results.append(f"Inventory: +'{self.add_item}'")

        if self.remove_item and self.remove_item in _party.shared_inventory:
            _party.shared_inventory.remove(self.remove_item)
            results.append(f"Inventory: -'{self.remove_item}'")

        if self.gold_change is not None:
            _party.gold = max(0, _party.gold + self.gold_change)
            results.append(f"Gold: {_party.gold} gp")

        if self.add_quest:
            parts = self.add_quest.split("|", 1)
            quest = {"name": parts[0], "description": parts[1] if len(parts) > 1 else "", "status": "active"}
            _party.active_quests.append(quest)
            results.append(f"Quest added: '{parts[0]}'")

        if self.complete_quest:
            for q in _party.active_quests:
                if q["name"].lower() == self.complete_quest.lower():
                    q["status"] = "completed"
                    results.append(f"Quest completed: '{q['name']}'")

        if self.gm_note:
            _party.notes.append(self.gm_note)
            results.append("GM note recorded.")

        return "\n".join(results) if results else "No changes applied."


# ═══════════════════════════════════════════════════════════════════
# PART 5: World Seeder — populate the KB with a sample world
# ═══════════════════════════════════════════════════════════════════

def seed_world(kb: AgoraKBClient):
    """
    Populate the Agora KB with a sample Baldur's Gate world.
    Run this once to set up the world data.
    """
    documents = [
        {
            "path": "worlds/baldurs-gate/overview.md",
            "title": "Baldur's Gate — City Overview",
            "tags": "city,baldurs-gate,forgotten-realms",
            "body": (
                "# Baldur's Gate\n\n"
                "A sprawling metropolis on the Sword Coast, Baldur's Gate is a city of\n"
                "stark contrasts. The wealthy Upper City gleams with marble and privilege,\n"
                "while the Lower City teems with merchants, sailors, and those seeking\n"
                "fortune. Beneath it all, the Undercity harbors secrets best left buried.\n\n"
                "## Factions\n"
                "- **The Flaming Fist** — mercenary company turned city guard\n"
                "- **The Guild** — thieves' guild controlling the shadows\n"
                "- **The Patriars** — noble families of the Upper City\n"
                "- **The Parliament of Peers** — governing council\n\n"
                "## Current Mood\n"
                "Tension simmers. Refugees from the north crowd the Outer City.\n"
                "The Flaming Fist tightens its grip. Something stirs in the temples."
            ),
        },
        {
            "path": "worlds/baldurs-gate/upper-city/overview.md",
            "title": "Upper City — District Overview",
            "tags": "district,upper-city,baldurs-gate",
            "body": (
                "# The Upper City\n\n"
                "Walled off from the rest of Baldur's Gate, the Upper City is where\n"
                "the Patriar families reside in their grand estates. The streets are\n"
                "clean, the guard presence heavy, and outsiders are watched carefully.\n"
                "At night, the gates close — those without a Patriar's invitation\n"
                "are locked out."
            ),
        },
        {
            "path": "worlds/baldurs-gate/upper-city/high-hall.md",
            "title": "High Hall — Seat of Government",
            "tags": "location,landmark,upper-city",
            "body": (
                "# High Hall\n\n"
                "The imposing seat of Baldur's Gate's government. Grand Duke Ulder\n"
                "Ravengard holds court here when he's not leading the Flaming Fist\n"
                "in the field. The Parliament of Peers convenes in the great chamber.\n\n"
                "## Notable NPCs\n"
                "- **Grand Duke Ulder Ravengard** — stern military leader, respected but feared\n"
                "- **Liara Portyr** — Duke, manages the city's defenses\n"
                "- **Clerks and functionaries** — the real wheels of bureaucracy\n\n"
                "## Atmosphere\n"
                "Marble floors echo with purposeful footsteps. Petitioners queue\n"
                "for audiences. Guards in polished armor stand at every doorway."
            ),
        },
        {
            "path": "worlds/baldurs-gate/lower-city/overview.md",
            "title": "Lower City — District Overview",
            "tags": "district,lower-city,baldurs-gate",
            "body": (
                "# The Lower City\n\n"
                "The beating heart of Baldur's Gate. Here merchants hawk their wares,\n"
                "sailors stumble from tavern to tavern, and everyone has an angle.\n"
                "The Flaming Fist patrols but can't be everywhere — the Guild fills\n"
                "the gaps with its own brand of order."
            ),
        },
        {
            "path": "worlds/baldurs-gate/lower-city/market.md",
            "title": "The Wide — Central Marketplace",
            "tags": "location,market,lower-city",
            "body": (
                "# The Wide\n\n"
                "Baldur's Gate's largest open market, a cacophony of commerce.\n"
                "Stalls sell everything from Calishite spices to dubious potions.\n"
                "Street performers compete with shouting merchants for attention.\n\n"
                "## Vendors\n"
                "- **Old Margha** — herbalist, sells common potions and components\n"
                "- **Drell the Fence** — 'antiques dealer', knows the Guild\n"
                "- **Sergeant Falk** — Flaming Fist presence, keeps the peace (mostly)\n\n"
                "## Atmosphere\n"
                "Crowded, loud, alive. The smell of roasting meat mixes with\n"
                "exotic incense. Pickpockets work the crowd. A good place to\n"
                "hear rumors or find trouble.\n\n"
                "## Rumors\n"
                "- Strange disappearances near the docks at night\n"
                "- A merchant claims to sell genuine Netherese artifacts\n"
                "- The Flaming Fist is recruiting — something big is coming"
            ),
        },
        {
            "path": "worlds/baldurs-gate/lower-city/elfsong-tavern.md",
            "title": "Elfsong Tavern",
            "tags": "location,tavern,lower-city",
            "body": (
                "# Elfsong Tavern\n\n"
                "A beloved Lower City institution, famous for the ghostly elven\n"
                "voice that sings mournful songs at unpredictable hours. The\n"
                "tavern is warm, dimly lit, and perpetually busy.\n\n"
                "## Notable NPCs\n"
                "- **Alan Alyth** — halfling owner, knows everyone's business\n"
                "- **Dead Man's Hand table** — where the Guild conducts quiet business\n"
                "- **Various adventurers** — this is THE gathering spot\n\n"
                "## Atmosphere\n"
                "Creaking wood, spilled ale, low conversation punctuated by\n"
                "sudden laughter. The elfsong, when it comes, silences the room.\n"
                "Everyone stops. Everyone listens. Then life resumes.\n\n"
                "## Services\n"
                "- Rooms: 5 sp/night (clean-ish)\n"
                "- Meals: 3 cp (stew), 1 sp (roast)\n"
                "- Ale: 2 cp, Wine: 5 cp\n"
                "- Information: price varies (ask Alan)"
            ),
        },
        {
            "path": "worlds/baldurs-gate/undercity/overview.md",
            "title": "The Undercity — District Overview",
            "tags": "district,undercity,baldurs-gate",
            "body": (
                "# The Undercity\n\n"
                "Beneath the streets of Baldur's Gate lies a labyrinth of sewers,\n"
                "ancient ruins, and forgotten temples. The Guild controls many of\n"
                "the passages. Deeper still, things lurk that predate the city\n"
                "entirely.\n\n"
                "## Danger Level\n"
                "HIGH. Even the Flaming Fist avoids the deeper tunnels.\n"
                "Light sources are essential. Trust no one you meet down here."
            ),
        },
        {
            "path": "worlds/baldurs-gate/undercity/bhaal-temple.md",
            "title": "Temple of Bhaal — The Murder Shrine",
            "tags": "location,temple,undercity,bhaal,dangerous",
            "body": (
                "# Temple of Bhaal\n\n"
                "A blood-stained altar to the Lord of Murder, hidden deep\n"
                "beneath the city. Though Bhaal was slain during the Time of\n"
                "Troubles, his worship persists in dark corners. The temple\n"
                "radiates malice.\n\n"
                "## Notable Features\n"
                "- **The Altar** — black stone, perpetually damp with something red\n"
                "- **Bone Pit** — offerings of the faithful, centuries deep\n"
                "- **The Whispering Walls** — the stone itself seems to murmur\n\n"
                "## Dangers\n"
                "- Bhaalspawn cultists (fanatical, well-armed)\n"
                "- Traps: pressure plates, poison darts, pit traps\n"
                "- The temple itself may be semi-sentient\n\n"
                "## Atmosphere\n"
                "Cold. Dark. The air tastes of copper. Your torchlight seems\n"
                "dimmer here, as if the darkness pushes back."
            ),
        },
    ]

    print("Seeding world data into Agora KB...")
    for doc in documents:
        success = kb.write_document(
            path=doc["path"],
            title=doc["title"],
            body=doc["body"],
            tags=doc.get("tags", ""),
        )
        status = "✓" if success else "✗"
        print(f"  {status} {doc['path']}")

    print(f"Seeded {len(documents)} locations.")


# ═══════════════════════════════════════════════════════════════════
# PART 6: Main — assemble and run the game
# ═══════════════════════════════════════════════════════════════════

def main():
    global _world_ctx, _party, _kb_client, _msg_manager

    # ── Configuration ──
    AGORA_URL = os.getenv("AGORA_URL", "http://127.0.0.1:8321")
    PROJECT_SLUG = os.getenv("GAME_PROJECT", "game-world")

    # ── LLM Provider ──
    # Groq (fast, free tier)
    api = GroqChatAPI(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )

    # OpenRouter alternative:
    # api = OpenAIChatAPI(
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    #     base_url="https://openrouter.ai/api/v1",
    #     model="openai/gpt-4o-mini",
    # )

    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = 0.7  # Higher for creative GM responses
    settings.top_p = 0.95

    # Summarizer uses lower temperature for factual summaries
    summarizer_settings = api.get_default_settings()
    summarizer_settings.temperature = 0.2

    # ── Agora KB Client ──
    _kb_client = AgoraKBClient(base_url=AGORA_URL, project_slug=PROJECT_SLUG)

    # ── Party State ──
    _party = PartyState()
    _party.add_character("Kael", "Fighter", level=3, hp=28)
    _party.add_character("Lyra", "Wizard", level=3, hp=16)
    _party.add_character("Theren", "Rogue", level=3, hp=22)
    _party.gold = 150
    _party.shared_inventory = ["Healing Potion x2", "Rope (50ft)", "Torches x5"]

    # ── Smart Message Manager ──
    _msg_manager = SmartMessageManager()

    # Pinned safety/rules message
    pinned_msg = ChatMessage.create_system_message(
        "[SYSTEM] You are a Game Master. Stay in character. Never break the fourth wall "
        "unless the player uses a meta-command. Track party state with update_party tool."
    )
    _msg_manager.add_message(
        pinned_msg,
        lifecycle=MessageLifecycle(pinned=True),
    )

    # ── Prompt Composer ──
    composer = PromptComposer()

    composer.add_module(
        name="gm_instructions",
        position=0,
        content=(
            "You are an expert Game Master running a tabletop RPG set in the Forgotten Realms.\n\n"
            "CRITICAL RULES:\n"
            "1. ALWAYS begin your response with 'Current Location: <location title>' on its own line.\n"
            "2. When the players want to travel to a new location, use list_locations to show\n"
            "   available destinations, then use set_location to travel there.\n"
            "3. Use update_party to track ALL mechanical changes (HP, items, gold, quests).\n"
            "4. Use search_world and read_document to look up lore, NPCs, and details.\n"
            "5. Describe scenes vividly. Use all senses. Make NPCs feel alive.\n"
            "6. Present meaningful choices. Not everything is combat — social, exploration,\n"
            "   and mystery are equally important.\n"
            "7. When the party enters a new area, ALWAYS use set_location first, then\n"
            "   describe what they see based on the loaded location data.\n\n"
            "LOCATION NAVIGATION:\n"
            "The world is organized as a hierarchy of paths:\n"
            "  worlds/baldurs-gate/              → the city\n"
            "  worlds/baldurs-gate/lower-city/   → a district\n"
            "  worlds/baldurs-gate/lower-city/market.md → a specific location\n\n"
            "Use list_locations with a path prefix to discover what's available.\n"
            "Use set_location with the full .md path to travel there.\n"
            "The system will automatically load location context into your prompt."
        ),
    )

    # World overview (dynamic — refreshed each turn)
    composer.add_module(
        name="world_overview",
        position=5,
        content_fn=lambda: (
                _kb_client.read_document("worlds/baldurs-gate/overview.md") or {}
        ).get("body", "World data unavailable."),
        prefix="### World Overview",
        suffix="### End World Overview",
    )

    # Location context (updated by WorldContextManager on travel)
    composer.add_module(
        name="location_context",
        position=10,
        content="No location loaded yet. Use set_location to begin.",
        prefix="### Current Location: None",
        suffix="### End Location Context",
    )

    # Party state (dynamic — re-rendered each turn)
    composer.add_module(
        name="party_state",
        position=15,
        content_fn=lambda: _party.build_context(),
        prefix="### Party State",
        suffix="### End Party State",
    )

    # ── World Context Manager ──
    _world_ctx = WorldContextManager(
        kb_client=_kb_client,
        composer=composer,
        summarizer_agent=agent,
        summarizer_settings=summarizer_settings,
        world_root="worlds",
    )

    # ── Tool Registry ──
    tool_registry = ToolRegistry()
    tool_registry.add_tools([
        FunctionTool(SetLocation),
        FunctionTool(ListLocations),
        FunctionTool(SearchWorld),
        FunctionTool(ReadDocument),
        FunctionTool(UpdateParty),
    ])

    # ── Seed world data (optional — comment out after first run) ──
    print("Checking world data...")
    test_doc = _kb_client.read_document("worlds/baldurs-gate/overview.md")
    if not test_doc:
        print("World data not found. Seeding...")
        seed_world(_kb_client)
    else:
        print("World data found. Skipping seed.")

    # ── Load starting location ──
    start_result = _world_ctx.load_location("worlds/baldurs-gate/lower-city/market.md")
    print(f"Starting location: {start_result}")

    # ── Inject opening ephemeral context ──
    opening = ChatMessage.create_system_message(
        "[Ephemeral Context] This is the start of a new adventure. "
        "Set the scene dramatically. The party has just arrived in Baldur's Gate "
        "after a long journey from the north. They are tired, low on supplies, "
        "and have heard rumors of trouble in the city."
    )
    _msg_manager.add_message(
        opening,
        lifecycle=MessageLifecycle(ttl=2, on_expire=ExpiryAction.REMOVE),
    )

    # ── Message TTL config ──
    USER_TTL = 10
    ASSISTANT_TTL = 10

    print()
    print("=" * 60)
    print("  ⚔️  Virtual Game Master — Baldur's Gate")
    print("  Powered by ToolAgents + Agora KB + PromptComposer")
    print("=" * 60)
    print()
    print("Commands:")
    print("  quit          — End the session")
    print("  /party        — Show party state")
    print("  /location     — Show current location path")
    print("  /archive      — Show archived messages")
    print("  /status       — Show system status")
    print("  /seed         — Re-seed the world data")
    print()
    print("Type your actions as a player. The GM will respond.")
    print()

    while True:
        try:
            user_input = input("⚔️  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nThe adventure ends... for now.")
            break

        if not user_input:
            continue

        # ── Meta-commands ──
        if user_input.lower() == "quit":
            print("The adventure ends... for now.")
            break

        elif user_input.lower() == "/party":
            print(f"\n{_party.build_context()}")
            continue

        elif user_input.lower() == "/location":
            if _world_ctx.current_location_path:
                print(f"\n📍 {_world_ctx.current_location_title}")
                print(f"   Path: {_world_ctx.current_location_path}")
                print(f"   History: {' → '.join(_world_ctx.location_history)}")
            else:
                print("\n📍 No location loaded.")
            continue

        elif user_input.lower() == "/archive":
            print(f"\n📦 Archive ({len(_msg_manager.archive)} messages):")
            for i, msg in enumerate(_msg_manager.archive):
                preview = msg.get_as_text()[:100]
                print(f"  [{i}] {preview}")
            if not _msg_manager.archive:
                print("  (empty)")
            continue

        elif user_input.lower() == "/status":
            print(f"\n📊 Status:")
            print(f"  Location: {_world_ctx.current_location_title}")
            print(f"  Active messages: {_msg_manager.message_count}")
            print(f"  Archived: {len(_msg_manager.archive)}")
            print(f"  Party members: {len(_party.characters)}")
            print(f"  Locations visited: {len(_world_ctx.location_history)}")
            continue

        elif user_input.lower() == "/seed":
            seed_world(_kb_client)
            continue

        # ── Tick message lifecycles ──
        tick_result = _msg_manager.tick()
        if tick_result.removed:
            for m in tick_result.removed:
                print("  🗑️  [Expired] ephemeral context removed")
        if tick_result.archived:
            for m in tick_result.archived:
                preview = m.get_as_text()[:60]
                print(f"  📦 [Archived] \"{preview}...\"")

        # ── Add user message with TTL ──
        user_msg = ChatMessage.create_user_message(user_input)
        _msg_manager.add_message(
            user_msg,
            lifecycle=MessageLifecycle(
                ttl=USER_TTL,
                on_expire=ExpiryAction.ARCHIVE,
            ),
        )

        # ── Compile system prompt (dynamic modules re-render here) ──
        system_prompt = composer.compile()

        # ── Build messages: system + active managed messages ──
        messages = [
            ChatMessage.create_system_message(system_prompt),
            *_msg_manager.get_active_messages(),
        ]

        # ── Call the GM agent ──
        try:
            chat_response = agent.get_response(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
            )

            response_text = chat_response.response.strip()
            print(f"\n🎭 GM > {response_text}")

            # Add assistant response with TTL
            assistant_msg = ChatMessage.create_assistant_message(response_text)
            _msg_manager.add_message(
                assistant_msg,
                lifecycle=MessageLifecycle(
                    ttl=ASSISTANT_TTL,
                    on_expire=ExpiryAction.ARCHIVE,
                ),
            )

            # Add tool-related messages
            for msg in chat_response.messages:
                role = msg.get_role() if hasattr(msg, "get_role") else None
                if role not in ("user", "assistant"):
                    _msg_manager.add_message(
                        msg,
                        lifecycle=MessageLifecycle(
                            ttl=ASSISTANT_TTL,
                            on_expire=ExpiryAction.ARCHIVE,
                        ),
                    )

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("  (The GM encountered an issue. Try again.)")

    # ── Cleanup ──
    _kb_client.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--seed-only":
        kb = AgoraKBClient()
        seed_world(kb)
        kb.close()
    else:
        main()
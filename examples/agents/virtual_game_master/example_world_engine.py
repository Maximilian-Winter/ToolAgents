"""
=============================================================================
example_world_engine.py — Virtual Game Master with Living World (Extended)
=============================================================================

A complete Virtual Game Master harness combining:
  - YAML Game Starters      → initial world state (characters, quests, setting)
  - XMLGameState             → structured game state, LLM-updatable via XML fragments
  - MessageTemplate          → customizable prompt templates with {placeholder} substitution
  - PromptComposer           → dynamic system prompt modules (re-rendered each turn)
  - SmartMessageManager      → message lifecycles (TTL, archival, pinning)
  - Agora Knowledge Base     → hierarchical world data (locations, NPCs, lore)
  - CoreMemory               → runtime GM notes and session memory

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    YAML Game Starter File                           │
  │  (setting, characters, companions, quests, inventory, factions)     │
  └──────────────────────┬──────────────────────────────────────────────┘
                         │ load at startup
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      XMLGameState                                   │
  │  Structured tree of all game data. The GM updates it by emitting    │
  │  XML fragments: <game-state><inventory>...</inventory></game-state> │
  │  which merge into the tree automatically.                           │
  └──────────────────────┬──────────────────────────────────────────────┘
                         │ rendered to string each turn
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      PromptComposer                                 │
  │  ┌────────────┬────────────┬───────────┬──────────┬──────────────┐ │
  │  │ GM Prompt  │ Game State │ Location  │ GM Notes │ Game Rules   │ │
  │  │ (template) │ (XML→text) │ (from KB) │ (memory) │ (static)     │ │
  │  │ pos=0      │ pos=5      │ pos=10    │ pos=15   │ pos=20       │ │
  │  └────────────┴────────────┴───────────┴──────────┴──────────────┘ │
  └──────────────────────┬──────────────────────────────────────────────┘
                         │ compile() each turn
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SmartMessageManager                                                │
  │  [pinned: rules] [archival: old turns TTL=12] [ephemeral: events]   │
  └──────────────────────┬──────────────────────────────────────────────┘
                         │ system_prompt + active_messages
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ChatToolAgent.get_response()                                       │
  │  Tools: set_location, list_locations, search_world,                 │
  │         update_game_state, read_document, add_gm_note,              │
  │         search_gm_notes                                             │
  └─────────────────────────────────────────────────────────────────────┘

Usage:
  1. Start Agora server:      python -m agora.runner
  2. Create a 'game-world' project in the Agora dashboard
  3. Optionally place a YAML game starter at game_starters/your_game.yaml
  4. Run:                     python example_world_engine.py
     Or seed only:            python example_world_engine.py --seed-only
     Or with custom starter:  python example_world_engine.py --starter game_starters/rpg_wudang.yaml
     Or with custom template: python example_world_engine.py --template gm_prompts/wuxia_gm.txt
"""

import json
import os
import sys
import re
import datetime as dt
from copy import copy
from typing import Optional, Dict, Any, List
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import xml.etree.ElementTree as ET

import httpx
import yaml
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
# PART 1: XMLGameState — structured, LLM-updatable game state
# ═══════════════════════════════════════════════════════════════════

def _clean_tag(tag: str) -> str:
    """Sanitize a string into a valid XML tag name."""
    return tag.replace(" ", "-").replace("_", "-").replace("'", "")


def _yaml_dict_to_xml(tag: str, data: dict) -> Element:
    """Recursively convert a dict (from YAML) into an XML element tree."""
    elem = Element(_clean_tag(tag))
    for key, val in data.items():
        key = _clean_tag(str(key))
        if isinstance(val, dict):
            child = SubElement(elem, key)
            for sub_key, sub_val in val.items():
                sub_elem = _yaml_dict_to_xml(sub_key, {sub_key: sub_val})
                found = sub_elem.find(_clean_tag(str(sub_key)))
                if found is not None:
                    child.append(found)
        elif isinstance(val, list):
            child = SubElement(elem, key)
            for item in val:
                if isinstance(item, dict):
                    child.append(_yaml_dict_to_xml("item", item))
                else:
                    list_item = SubElement(child, "item")
                    list_item.text = str(item).strip()
        else:
            child = SubElement(elem, key)
            child.text = str(val).strip()
    return elem


def _merge_xml_update(original_root: Element, update_string: str):
    """Merge an XML fragment string into the existing game state tree."""
    cleaned = update_string.replace("\n", "").replace("\r", "").replace("\t", "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    update_root = ET.fromstring(cleaned)

    def recursive_update(original_element: Element, update_element: Element):
        for update_child in update_element:
            matching_original = original_element.find(update_child.tag)
            if matching_original is None or matching_original.tag == "item":
                original_element.append(update_child)
            else:
                if len(update_child) == 0:
                    matching_original.text = update_child.text
                else:
                    recursive_update(matching_original, update_child)
        for attr, value in update_element.attrib.items():
            original_element.set(attr, value)

    recursive_update(original_root, update_root)


def _xml_to_string(root: Element) -> str:
    """Pretty-print an XML element tree."""
    rough_string = tostring(root, "utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class XMLGameState:
    """
    Structured game state stored as an XML tree.
    
    - Load initial state from a YAML game starter file
    - Load/save state from/to XML files
    - The LLM updates state by emitting XML fragments that merge in
    - Renders to a readable string for the system prompt
    """

    def __init__(self, initial_state_file: Optional[str] = None):
        if initial_state_file:
            self.xml_root = self._load_yaml(initial_state_file)
        else:
            self.xml_root = Element("game-state")

    def _load_yaml(self, file_path: str) -> Element:
        """Load a YAML game starter and convert to XML tree."""
        with open(file_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        return _yaml_dict_to_xml("game-state", yaml_data)

    def get_xml_string(self) -> str:
        """Get the full game state as a pretty-printed XML string."""
        return _xml_to_string(self.xml_root)

    def update_from_xml_string(self, xml_string: str) -> str:
        """
        Merge an XML fragment into the game state.
        The fragment must be wrapped in <game-state>...</game-state>.
        Returns a status message.
        """
        try:
            _merge_xml_update(self.xml_root, xml_string)
            return "Game state updated successfully."
        except ET.ParseError as e:
            return f"Error parsing XML update: {e}"

    def get_element_text(self, path: str) -> Optional[str]:
        """Get the text of an element by XPath-like path (e.g., 'setting')."""
        elem = self.xml_root.find(path)
        if elem is not None:
            if elem.text:
                return elem.text.strip()
            # If it has children, return the subtree as string
            return _xml_to_string(elem)
        return None

    def save_to_file(self, file_path: str):
        """Save current state to an XML file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(_xml_to_string(self.xml_root))

    def load_from_file(self, file_path: str):
        """Load state from an XML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.xml_root = ET.fromstring(f.read())

    @classmethod
    def from_dict(cls, data: dict) -> "XMLGameState":
        """Create a game state from a Python dictionary (no file needed)."""
        instance = cls()
        instance.xml_root = _yaml_dict_to_xml("game-state", data)
        return instance


# ═══════════════════════════════════════════════════════════════════
# PART 2: MessageTemplate — customizable prompt templates
# ═══════════════════════════════════════════════════════════════════

class MessageTemplate:
    """
    Prompt template with {placeholder} substitution.
    Load from a file or string, then call generate() with field values.
    Empty placeholders are removed (lines with only empty fields are dropped).
    """

    def __init__(self, template_file: Optional[str] = None, template_string: Optional[str] = None):
        if template_file:
            with open(template_file, "r", encoding="utf-8") as f:
                self.template = f.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError("Provide either template_file or template_string.")

    @classmethod
    def from_string(cls, template_string: str) -> "MessageTemplate":
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file: str) -> "MessageTemplate":
        return cls(template_file=template_file)

    def generate(self, fields: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Generate the prompt by replacing {placeholders} with values.
        Fields not provided are removed along with their containing line
        (if the line becomes empty after removal).
        """
        all_fields = {**(fields or {}), **kwargs}
        cleaned = {k: str(v) if not isinstance(v, str) else v for k, v in all_fields.items()}

        def replace_placeholder(match):
            key = match.group(1)
            value = cleaned.get(key, None)
            if value is not None:
                return value
            return "__EMPTY_TEMPLATE_FIELD__"

        result = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)

        # Remove lines that are only empty placeholders
        lines = result.split("\n")
        output_lines = []
        for line in lines:
            if "__EMPTY_TEMPLATE_FIELD__" in line:
                cleaned_line = line.replace("__EMPTY_TEMPLATE_FIELD__", "")
                if cleaned_line.strip():
                    output_lines.append(cleaned_line)
            else:
                output_lines.append(line)
        return "\n".join(output_lines)


# ═══════════════════════════════════════════════════════════════════
# PART 3: CoreMemory — runtime GM notes and session memory
# ═══════════════════════════════════════════════════════════════════

class CoreMemory:
    """
    Key-value memory for the GM to store runtime notes.
    Separate from the structured XMLGameState — this is for
    freeform observations, plot threads, and session notes.
    """

    def __init__(self, block_limit: int = 500):
        self.blocks: Dict[str, str] = {}
        self.block_limit = block_limit
        self.last_modified: str = "never"

    def set_block(self, name: str, content: str) -> str:
        if len(content) > self.block_limit:
            return f"Error: exceeds {self.block_limit} char limit (got {len(content)})."
        self.blocks[name] = content
        self.last_modified = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Note '{name}' saved."

    def append_block(self, name: str, content: str) -> str:
        current = self.blocks.get(name, "")
        new = current + content if current else content
        if len(new) > self.block_limit:
            return f"Error: appending would exceed {self.block_limit} chars."
        self.blocks[name] = new
        self.last_modified = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Appended to note '{name}'."

    def get_block(self, name: str) -> str:
        return self.blocks.get(name, f"Note '{name}' not found.")

    def delete_block(self, name: str) -> str:
        if name in self.blocks:
            del self.blocks[name]
            self.last_modified = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"Note '{name}' deleted."
        return f"Note '{name}' not found."

    def search(self, query: str) -> str:
        results = []
        for name, content in self.blocks.items():
            if query.lower() in name.lower() or query.lower() in content.lower():
                results.append(f"[{name}]: {content[:100]}")
        if results:
            return f"Found {len(results)} note(s):\n" + "\n".join(results)
        return f"No notes matching '{query}'."

    def build_context(self) -> str:
        if not self.blocks:
            return "No GM notes stored yet."
        lines = []
        for name, content in self.blocks.items():
            lines.append(f"<{name}>\n{content}\n</{name}>")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"blocks": self.blocks.copy(), "last_modified": self.last_modified}

    def load_from_dict(self, data: dict):
        self.blocks = data.get("blocks", {})
        self.last_modified = data.get("last_modified", "never")


# ═══════════════════════════════════════════════════════════════════
# PART 4: Agora KB Client — REST interface to the world database
# ═══════════════════════════════════════════════════════════════════

class AgoraKBClient:
    """Synchronous client for Agora's Knowledge Base REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8321", project_slug: str = "game-world"):
        self.base_url = base_url.rstrip("/")
        self.project_slug = project_slug
        self.client = httpx.Client(timeout=30.0)

    @property
    def kb_url(self) -> str:
        return f"{self.base_url}/api/projects/{self.project_slug}/kb"

    def read_document(self, path: str, section: Optional[str] = None) -> Optional[dict]:
        url = f"{self.kb_url}/documents/{path}"
        params = {"section": section} if section else {}
        try:
            resp = self.client.get(url, params=params)
            return resp.json() if resp.status_code == 200 else None
        except httpx.HTTPError:
            return None

    def write_document(self, path: str, title: str, body: str, tags: str = "") -> bool:
        url = f"{self.kb_url}/documents"
        try:
            resp = self.client.put(url, json={"path": path, "title": title, "body": body, "tags": tags})
            return resp.status_code in (200, 201)
        except httpx.HTTPError:
            return False

    def list_documents(self, prefix: str = "") -> list[dict]:
        url = f"{self.kb_url}/documents"
        params = {"prefix": prefix} if prefix else {}
        try:
            resp = self.client.get(url, params=params)
            return resp.json() if resp.status_code == 200 else []
        except httpx.HTTPError:
            return []

    def search_documents(self, query: str) -> list[dict]:
        url = f"{self.kb_url}/search"
        try:
            resp = self.client.get(url, params={"q": query})
            return resp.json() if resp.status_code == 200 else []
        except httpx.HTTPError:
            return []

    def get_tree(self) -> dict:
        url = f"{self.kb_url}/tree"
        try:
            resp = self.client.get(url)
            return resp.json() if resp.status_code == 200 else {}
        except httpx.HTTPError:
            return {}

    def close(self):
        self.client.close()


# ═══════════════════════════════════════════════════════════════════
# PART 5: WorldContextManager — location tracking + auto-context
# ═══════════════════════════════════════════════════════════════════

class WorldContextManager:
    """
    Bridges the Agora KB with PromptComposer.
    When the GM changes location:
      1. Summarizes what happened (LLM call)
      2. Updates the old location's KB document
      3. Loads new location + parent overview
      4. Updates the PromptComposer location module
    """

    def __init__(
        self,
        kb_client: AgoraKBClient,
        composer: PromptComposer,
        summarizer_agent: Optional[ChatToolAgent] = None,
        summarizer_settings=None,
    ):
        self.kb = kb_client
        self.composer = composer
        self.summarizer_agent = summarizer_agent
        self.summarizer_settings = summarizer_settings

        self.current_location_path: Optional[str] = None
        self.current_location_title: str = "Unknown"
        self.current_location_body: str = ""
        self.location_history: list[str] = []

    def _get_parent_overview(self, path: str) -> Optional[str]:
        parts = path.rsplit("/", 1)
        if len(parts) > 1:
            return f"{parts[0]}/overview.md"
        return None

    def _get_area_prefix(self, path: str) -> str:
        parts = path.rsplit("/", 1)
        return parts[0] + "/" if len(parts) > 1 else "worlds/"

    def list_locations(self, prefix: str) -> list[dict]:
        docs = self.kb.list_documents(prefix=prefix)
        return [
            {"path": d.get("path", ""), "title": d.get("title", "")}
            for d in docs if d.get("path", "").endswith(".md")
        ]

    def load_location(self, path: str) -> str:
        doc = self.kb.read_document(path)
        if not doc:
            return f"Error: Location '{path}' not found in knowledge base."

        self.current_location_path = path
        self.current_location_title = doc.get("title", "Unknown")
        self.current_location_body = doc.get("body", "")

        # Build layered context
        context_parts = []

        # Parent overview
        parent_path = self._get_parent_overview(path)
        if parent_path and parent_path != path:
            parent = self.kb.read_document(parent_path)
            if parent:
                context_parts.append(
                    f"## Area: {parent.get('title', 'Unknown')}\n{parent.get('body', '')}"
                )

        # Current location
        context_parts.append(
            f"## Current Location: {self.current_location_title}\n"
            f"Path: {path}\n\n{self.current_location_body}"
        )

        # Nearby locations
        area = self._get_area_prefix(path)
        siblings = self.list_locations(area)
        nearby = [s for s in siblings if s["path"] != path and not s["path"].endswith("overview.md")]
        if nearby:
            nearby_list = "\n".join(f"  - {s['title']} ({s['path']})" for s in nearby)
            context_parts.append(f"\n## Nearby Locations:\n{nearby_list}")

        full_context = "\n\n---\n\n".join(context_parts)

        self.composer.update_module(
            "location_context",
            content=full_context,
            prefix=f"### Current Location: {self.current_location_title}",
            suffix="### End Location Context",
        )

        self.location_history.append(path)
        return f"Arrived at: {self.current_location_title}"

    def summarize_and_depart(self, recent_messages: list[ChatMessage]) -> str:
        if not self.current_location_path or not self.summarizer_agent:
            return ""

        conversation_text = "\n".join(
            f"{m.get_role()}: {m.get_as_text()}" for m in recent_messages[-10:]
        )

        summary_messages = [
            ChatMessage.create_system_message(
                "You are a concise note-taker for a tabletop RPG. "
                "Summarize what happened at this location in 2-4 sentences. "
                "Focus on: key events, NPC interactions, items gained/lost, consequences. "
                "Write in past tense, third person. Output ONLY the summary."
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
                tool_registry=ToolRegistry(),
            )
            summary = response.response.strip()
        except Exception as e:
            summary = f"(Summary unavailable: {e})"

        # Append event log to KB document
        if summary and self.current_location_path:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            doc = self.kb.read_document(self.current_location_path)
            if doc:
                updated_body = doc.get("body", "") + f"\n\n### Event Log — {timestamp}\n{summary}"
                self.kb.write_document(
                    path=self.current_location_path,
                    title=doc.get("title", self.current_location_title),
                    body=updated_body,
                    tags=doc.get("tags", ""),
                )

        return summary


# ═══════════════════════════════════════════════════════════════════
# PART 6: Default GM Prompt Template
# ═══════════════════════════════════════════════════════════════════

DEFAULT_GM_TEMPLATE = """You are an expert Game Master running a tabletop RPG.
{setting_context}

CRITICAL RULES:
1. ALWAYS begin your response with "Current Location: <location title>" on its own line.
2. When players want to travel, use list_locations to discover paths, then set_location to travel.
3. Use update_game_state to track ALL mechanical changes (HP, items, gold, quests, relationships).
   You MUST wrap updates in <game-state>...</game-state> XML tags.
4. Use search_world and read_document to look up lore, NPCs, and details from the knowledge base.
5. Use add_gm_note to record plot threads, secrets, and observations that should persist.
6. Describe scenes vividly. Use all senses. Make NPCs feel alive with distinct voices.
7. Present meaningful choices — combat, social, exploration, and mystery are all valid.
8. Respect the setting's tone and period. {genre_notes}

LOCATION NAVIGATION:
The world is organized as a hierarchy of paths in the knowledge base:
  worlds/<world-name>/                    → top-level world
  worlds/<world-name>/<district>/         → an area or district
  worlds/<world-name>/<district>/place.md → a specific location

Use list_locations with a path prefix to discover what's available.
Use set_location with the full .md path to travel there.
Location context loads automatically into your prompt.

GAME STATE UPDATES:
To update game state, call update_game_state with an XML fragment like:
  <game-state>
    <inventory>
      <item>New item gained</item>
    </inventory>
    <active-quests>
      <item>New quest description</item>
    </active-quests>
  </game-state>
The fragment merges into the existing state tree.
{additional_instructions}"""


# ═══════════════════════════════════════════════════════════════════
# PART 7: GM Tools — what the Game Master LLM can call
# ═══════════════════════════════════════════════════════════════════

# Module-level references bound in main()
_world_ctx: Optional[WorldContextManager] = None
_game_state: Optional[XMLGameState] = None
_gm_memory: Optional[CoreMemory] = None
_kb_client: Optional[AgoraKBClient] = None
_msg_manager: Optional[SmartMessageManager] = None


class SetLocation(BaseModel):
    """
    Travel to a new location. Provide the full KB path.
    This will summarize events at the current location, update its document,
    then load the new location's context.
    Use list_locations first to discover available paths.
    """
    location_path: str = Field(
        ..., description="Full KB path to the location (e.g., 'worlds/han-dynasty/wudang-mountains/monastery.md')."
    )

    def run(self) -> str:
        if _world_ctx is None:
            return "Error: World context not initialized."
        if _world_ctx.current_location_path and _msg_manager:
            active = _msg_manager.get_active_messages()
            summary = _world_ctx.summarize_and_depart(active)
            if summary:
                print(f"\n  📜 [Location Update] {_world_ctx.current_location_title}: {summary}")
        return _world_ctx.load_location(self.location_path)


class ListLocations(BaseModel):
    """
    List available locations under a path prefix.
    Examples: "worlds/" → all worlds, "worlds/han-dynasty/wudang-mountains/" → specific places.
    """
    path_prefix: str = Field(
        ..., description="Path prefix with trailing slash (e.g., 'worlds/han-dynasty/')."
    )

    def run(self) -> str:
        if _world_ctx is None:
            return "Error: World context not initialized."
        locations = _world_ctx.list_locations(self.path_prefix)
        if not locations:
            return f"No locations found under '{self.path_prefix}'."
        lines = [f"Locations under '{self.path_prefix}':"]
        for loc in locations:
            lines.append(f"  - {loc['title']} → {loc['path']}")
        return "\n".join(lines)


class SearchWorld(BaseModel):
    """Search the world knowledge base for NPCs, items, lore, or locations."""
    query: str = Field(..., description="Search term (e.g., 'Master Chen', 'artifact', 'bandits').")

    def run(self) -> str:
        if _kb_client is None:
            return "Error: KB client not initialized."
        results = _kb_client.search_documents(self.query)
        if not results:
            return f"No results for '{self.query}'."
        lines = [f"Search results for '{self.query}':"]
        for r in results[:5]:
            lines.append(f"  - [{r.get('title', '?')}] ({r.get('path', '?')}): {r.get('snippet', '')[:120]}")
        return "\n".join(lines)


class ReadDocument(BaseModel):
    """Read a specific document from the knowledge base. Optionally extract one section."""
    path: str = Field(..., description="Full document path.")
    section: Optional[str] = Field(None, description="Optional section header to extract.")

    def run(self) -> str:
        if _kb_client is None:
            return "Error: KB client not initialized."
        doc = _kb_client.read_document(self.path, section=self.section)
        if not doc:
            return f"Document not found: '{self.path}'"
        title = doc.get("title", "Untitled")
        body = doc.get("body", "")
        header = f"## {title} — Section: {self.section}" if self.section else f"## {title}"
        return f"{header}\n\n{body}"


class UpdateGameState(BaseModel):
    """
    Update the structured game state with an XML fragment.
    The fragment MUST be wrapped in <game-state>...</game-state>.
    It merges into the existing state tree — matching elements are updated,
    new elements are appended.
    
    Example:
      <game-state>
        <inventory><Li-Wei><item>Ancient scroll</item></Li-Wei></inventory>
        <active-quests><item>Find the hidden temple</item></active-quests>
      </game-state>
    """
    xml_fragment: str = Field(
        ..., description="XML fragment wrapped in <game-state>...</game-state>."
    )

    def run(self) -> str:
        if _game_state is None:
            return "Error: Game state not initialized."
        return _game_state.update_from_xml_string(self.xml_fragment)


class AddGMNote(BaseModel):
    """
    Save a GM note to persistent memory. Use for plot threads,
    NPC secrets, player tendencies, and session observations.
    """
    note_name: str = Field(..., description="Short name/key for the note (e.g., 'plot_thread_1', 'npc_secret').")
    content: str = Field(..., description="The note content (max 500 chars).")

    def run(self) -> str:
        if _gm_memory is None:
            return "Error: GM memory not initialized."
        return _gm_memory.set_block(self.note_name, self.content)


class SearchGMNotes(BaseModel):
    """Search through GM notes for a keyword."""
    query: str = Field(..., description="Search term.")

    def run(self) -> str:
        if _gm_memory is None:
            return "Error: GM memory not initialized."
        return _gm_memory.search(self.query)


# ═══════════════════════════════════════════════════════════════════
# PART 8: World Seeder — populate Agora KB with sample world data
# ═══════════════════════════════════════════════════════════════════

def seed_wudang_world(kb: AgoraKBClient):
    """Seed the KB with a Wudang Mountains / Han Dynasty world."""
    documents = [
        {
            "path": "worlds/han-dynasty/overview.md",
            "title": "Han Dynasty China — World Overview",
            "tags": "world,han-dynasty,china",
            "body": (
                "# Han Dynasty China — circa 200 CE\n\n"
                "The Han Dynasty crumbles. Emperor Xian is a puppet, controlled by\n"
                "warlords who carve the empire into fiefdoms. The Yellow Turban\n"
                "Rebellion has shaken the foundations. Philosophical debates rage\n"
                "between Confucian scholars, Taoist mystics, and Legalist administrators.\n\n"
                "## The Silk Road\n"
                "Trade routes bring exotic goods, foreign ideas, and dangerous\n"
                "ambitions from the western regions.\n\n"
                "## Factions\n"
                "- **Imperial Court** — weakening but still the legitimate government\n"
                "- **Warlords** — Cao Cao, Sun Quan, Liu Bei, and others vie for supremacy\n"
                "- **Yellow Turbans** — remnants operating in secret cells\n"
                "- **Taoist Sects** — mystical groups with hidden knowledge and martial arts\n"
                "- **Wu Xia** — wandering martial artists bound by codes of honor"
            ),
        },
        {
            "path": "worlds/han-dynasty/wudang-mountains/overview.md",
            "title": "Wudang Mountains — Region Overview",
            "tags": "region,wudang,mountains",
            "body": (
                "# The Wudang Mountains\n\n"
                "Sacred peaks shrouded in mist, home to Taoist monasteries and\n"
                "martial arts schools. The mountains are both a spiritual refuge\n"
                "and a strategic waypoint — whoever controls the mountain passes\n"
                "controls trade between north and south.\n\n"
                "## Climate\n"
                "Cool and misty. Bamboo forests cling to steep slopes.\n"
                "Waterfalls cascade into deep valleys. The air is thin and clean.\n\n"
                "## Dangers\n"
                "- Bandits prey on pilgrims and traders\n"
                "- Wild animals: tigers, bears, venomous snakes\n"
                "- The mountains themselves — treacherous paths, sudden storms\n"
                "- Not all monks are what they seem"
            ),
        },
        {
            "path": "worlds/han-dynasty/wudang-mountains/monastery.md",
            "title": "Wudang Monastery — Temple of the Purple Cloud",
            "tags": "location,monastery,taoist,wudang",
            "body": (
                "# Wudang Monastery — Temple of the Purple Cloud\n\n"
                "A remote Taoist monastery perched on a cliff face, accessible only\n"
                "by a narrow stone stairway carved into the mountain. Known for its\n"
                "martial arts traditions and mystical practices.\n\n"
                "## Notable NPCs\n"
                "- **Master Chen (陳師父)** — the enigmatic head of the monastery.\n"
                "  Ancient, with piercing eyes that seem to see through deception.\n"
                "  Speaks in riddles but his words always prove wise.\n"
                "- **Brother Fang** — the monastery's martial arts instructor.\n"
                "  Gruff exterior, kind heart. Missing his left ear (bandit attack).\n"
                "- **Sister Yue** — the herbalist. Quiet, observant, keeps the\n"
                "  monastery's extensive library of scrolls.\n\n"
                "## Atmosphere\n"
                "Incense smoke curls through wooden halls. The sound of chanting\n"
                "mingles with the clash of practice swords. Dawn meditation on\n"
                "the cliff edge, watching clouds flow through the valley below.\n\n"
                "## Secrets\n"
                "The monastery guards an ancient text — the Taixuan Jing — said\n"
                "to contain prophecies about the dynasty's fall. Master Chen has\n"
                "allowed only fragments to be read."
            ),
        },
        {
            "path": "worlds/han-dynasty/wudang-mountains/mountain-path.md",
            "title": "Mountain Path — The Pilgrim's Ascent",
            "tags": "location,path,wudang,travel",
            "body": (
                "# The Pilgrim's Ascent\n\n"
                "A winding stone path that climbs from the foothills to the\n"
                "monastery above. Takes half a day on foot. The path passes\n"
                "through bamboo groves, over rope bridges, and along cliff edges\n"
                "with dizzying drops.\n\n"
                "## Waypoints\n"
                "- **First Gate** — stone archway with carved dragons, marks the\n"
                "  start of monastery territory\n"
                "- **Waterfall Rest** — a small shrine beside a waterfall,\n"
                "  traditional resting spot for pilgrims\n"
                "- **The Narrow** — a section where the path is barely two feet\n"
                "  wide with a sheer drop on one side\n\n"
                "## Encounters\n"
                "- Pilgrims and traders (usually peaceful)\n"
                "- Mountain bandits (especially at The Narrow — easy ambush point)\n"
                "- Occasionally, a wandering Wu Xia seeking the monastery"
            ),
        },
        {
            "path": "worlds/han-dynasty/wudang-mountains/hidden-cave.md",
            "title": "Hidden Cave — The Whispering Grotto",
            "tags": "location,cave,secret,wudang",
            "body": (
                "# The Whispering Grotto\n\n"
                "A natural cave system behind the waterfall on the Pilgrim's Ascent.\n"
                "Most travelers don't know it exists — the entrance is hidden behind\n"
                "the curtain of water. Inside, the cave walls are covered with\n"
                "ancient carvings that predate the Han Dynasty.\n\n"
                "## Features\n"
                "- **The Carvings** — astronomical charts and what appear to be\n"
                "  instructions for a ritual. Sister Yue has been studying them.\n"
                "- **The Inner Chamber** — deeper in, a perfectly circular room\n"
                "  with an altar made of jade. Something glows faintly beneath it.\n"
                "- **Underground Stream** — fresh water, but the sound it makes\n"
                "  sounds almost like whispered words\n\n"
                "## The Artifact\n"
                "Rumors speak of a bronze mirror hidden here — the Kunlun Mirror —\n"
                "said to reveal truths that the eye cannot see. Whether this is\n"
                "the artifact the party seeks remains to be discovered."
            ),
        },
        {
            "path": "worlds/han-dynasty/wudang-mountains/village.md",
            "title": "Foothill Village — Three Pines",
            "tags": "location,village,wudang",
            "body": (
                "# Three Pines Village\n\n"
                "A small farming village at the base of the Wudang Mountains.\n"
                "Named for three ancient pine trees in the village square.\n"
                "The villagers grow rice, raise silkworms, and trade with\n"
                "passing merchants on the road south.\n\n"
                "## Notable NPCs\n"
                "- **Elder Wu** — village headman. Worried about recent bandit raids.\n"
                "  Lost his grandson to the Yellow Turbans three years ago.\n"
                "- **Blacksmith Zhao** — can repair weapons. Secretly a former\n"
                "  soldier who deserted Cao Cao's army.\n"
                "- **Tea House Auntie** — proprietor of the only tea house.\n"
                "  Gossip hub. She knows everything that passes through.\n\n"
                "## Services\n"
                "- Rooms: 2 copper/night (basic but clean)\n"
                "- Meals: 1 copper (rice and vegetables)\n"
                "- Weapon repair: negotiable with Blacksmith Zhao\n"
                "- Supplies: basic traveling gear, rope, torches\n\n"
                "## Current Tensions\n"
                "Bandits from the mountain have been demanding 'protection fees'.\n"
                "The villagers can barely afford it. Elder Wu is desperate."
            ),
        },
    ]

    print("Seeding Wudang / Han Dynasty world into Agora KB...")
    for doc in documents:
        success = kb.write_document(doc["path"], doc["title"], doc["body"], doc.get("tags", ""))
        print(f"  {'✓' if success else '✗'} {doc['path']}")
    print(f"Seeded {len(documents)} locations.\n")


# ═══════════════════════════════════════════════════════════════════
# PART 9: Save / Load Session
# ═══════════════════════════════════════════════════════════════════

def save_session(
    filepath: str,
    game_state: XMLGameState,
    gm_memory: CoreMemory,
    world_ctx: WorldContextManager,
    msg_manager: SmartMessageManager,
):
    """Save the full session state to a JSON file + XML game state."""
    base = filepath.rsplit(".", 1)[0] if "." in filepath else filepath

    # Save game state as XML
    game_state.save_to_file(f"{base}_game_state.xml")

    # Save session metadata as JSON
    session = {
        "saved_at": dt.datetime.now().isoformat(),
        "current_location": world_ctx.current_location_path,
        "location_history": world_ctx.location_history,
        "gm_memory": gm_memory.to_dict(),
        "archive": [m.get_as_text() for m in msg_manager.archive],
        "active_messages": [m.get_as_text() for m in msg_manager.get_active_messages()],
    }
    with open(f"{base}_session.json", "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

    print(f"  💾 Saved: {base}_game_state.xml + {base}_session.json")


# ═══════════════════════════════════════════════════════════════════
# PART 10: Main — assemble and run the game
# ═══════════════════════════════════════════════════════════════════

def main():
    global _world_ctx, _game_state, _gm_memory, _kb_client, _msg_manager

    # ── Parse arguments ──
    starter_file = None
    template_file = None
    seed_only = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--starter" and i + 1 < len(args):
            starter_file = args[i + 1]
            i += 2
        elif args[i] == "--template" and i + 1 < len(args):
            template_file = args[i + 1]
            i += 2
        elif args[i] == "--seed-only":
            seed_only = True
            i += 1
        else:
            i += 1

    # ── Configuration ──
    AGORA_URL = os.getenv("AGORA_URL", "http://127.0.0.1:8321")
    PROJECT_SLUG = os.getenv("GAME_PROJECT", "game-world")

    # ── KB Client ──
    _kb_client = AgoraKBClient(base_url=AGORA_URL, project_slug=PROJECT_SLUG)

    if seed_only:
        seed_wudang_world(_kb_client)
        _kb_client.close()
        return

    # ── LLM Provider ──
    #api = GroqChatAPI(
    #    api_key=os.getenv("GROQ_API_KEY"),
    #    model="llama-3.3-70b-versatile",
    #)
    # Alternative: OpenRouter
    api = OpenAIChatAPI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
    )

    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = 0.7
    settings.top_p = 0.95

    summarizer_settings = api.get_default_settings()
    summarizer_settings.temperature = 0.2

    # ── Game State (from YAML or default) ──
    if starter_file and os.path.exists(starter_file):
        print(f"Loading game starter: {starter_file}")
        _game_state = XMLGameState(initial_state_file=starter_file)
    else:
        print("No starter file — using default Han Dynasty setting.")
        _game_state = XMLGameState.from_dict({
            "setting": (
                "Ancient China, late Han Dynasty (circa 200 CE). "
                "A time of political turmoil, philosophical debate, "
                "and the rise of powerful warlords."
            ),
            "player-character": (
                "Li Wei (李威): A human male Scholar-Warrior, age 32. "
                "Intelligent dark eyes, neatly trimmed beard. "
                "Wears scholar's robes with hidden armor. "
                "Carries a jian and philosophical scrolls."
            ),
            "companions": ["Mei Ling (梅玲): Wu Xia, age 29. Butterfly swords, acrobatics, dry humor."],
            "location": "Wudang Monastery — Temple of the Purple Cloud",
            "active-quests": [
                "Decode an ancient text predicting the fall of the Han Dynasty",
                "Investigate rumors of a powerful artifact in the Wudang Mountains",
                "Prevent an assassination attempt on a key political figure",
            ],
            "inventory": {
                "Li-Wei": [
                    "Jian (double-edged straight sword)",
                    "Philosophical and martial arts scrolls",
                    "Writing kit",
                    "Hidden light armor",
                    "Medicinal herbs",
                ],
                "Mei-Ling": [
                    "Pair of butterfly swords",
                    "Rope dart",
                    "Smoke pellets",
                    "Climbing claws",
                    "Disguise kit",
                ],
            },
            "key-npcs": [
                "Master Chen: Enigmatic head of the Taoist monastery",
                "Cao Cao: Cunning warlord rising to power",
                "Lady Sun: Noblewoman with hidden ties to rebel factions",
            ],
            "time-and-calendar": (
                "15th day of the 7th month, 5th year of Jian'an (200 CE). "
                "Mid-summer. Early morning."
            ),
            "world-state": [
                "The Han Dynasty's power is waning",
                "Philosophical debates between Confucianism, Taoism, and Legalism",
                "The Silk Road brings new ideas from distant lands",
            ],
        })

    # ── GM Memory ──
    _gm_memory = CoreMemory(block_limit=500)

    # ── Smart Message Manager ──
    _msg_manager = SmartMessageManager()

    pinned = ChatMessage.create_system_message(
        "[SYSTEM] You are the Game Master. Stay in character. Use tools to track state changes. "
        "Always begin responses with 'Current Location: <name>'."
    )
    _msg_manager.add_message(pinned, lifecycle=MessageLifecycle(pinned=True))

    # ── Prompt Composer ──
    composer = PromptComposer()

    # Load GM prompt template
    if template_file and os.path.exists(template_file):
        print(f"Loading custom GM template: {template_file}")
        gm_template = MessageTemplate.from_file(template_file)
        gm_prompt_text = gm_template.generate(
            setting_context=_game_state.get_element_text("setting") or "",
            genre_notes="This is a Wuxia martial arts setting. Respect Chinese cultural context.",
            additional_instructions="",
        )
    else:
        gm_template = MessageTemplate.from_string(DEFAULT_GM_TEMPLATE)
        gm_prompt_text = gm_template.generate(
            setting_context=_game_state.get_element_text("setting") or "",
            genre_notes="Respect the historical and cultural context of the setting.",
            additional_instructions="",
        )

    composer.add_module(name="gm_instructions", position=0, content=gm_prompt_text)

    # Game state (dynamic — XML re-rendered each turn)
    composer.add_module(
        name="game_state",
        position=5,
        content_fn=lambda: _game_state.get_xml_string(),
        prefix="### Game State (XML)",
        suffix="### End Game State",
    )

    # Location context (updated by WorldContextManager)
    composer.add_module(
        name="location_context",
        position=10,
        content="No location loaded yet. Use set_location to begin.",
        prefix="### Current Location: None",
        suffix="### End Location Context",
    )

    # GM notes (dynamic)
    composer.add_module(
        name="gm_notes",
        position=15,
        content_fn=lambda: _gm_memory.build_context(),
        prefix=f"### GM Notes [last modified: {_gm_memory.last_modified}]",
        suffix="### End GM Notes",
    )

    # ── World Context Manager ──
    _world_ctx = WorldContextManager(
        kb_client=_kb_client,
        composer=composer,
        summarizer_agent=agent,
        summarizer_settings=summarizer_settings,
    )

    # ── Tool Registry ──
    tool_registry = ToolRegistry()
    tool_registry.add_tools([
        FunctionTool(SetLocation),
        FunctionTool(ListLocations),
        FunctionTool(SearchWorld),
        FunctionTool(ReadDocument),
        FunctionTool(UpdateGameState),
        FunctionTool(AddGMNote),
        FunctionTool(SearchGMNotes),
    ])

    # ── Seed world if needed ──
    print("Checking world data in Agora KB...")
    test = _kb_client.read_document("worlds/han-dynasty/overview.md")
    if not test:
        print("World data not found. Seeding...")
        seed_wudang_world(_kb_client)
    else:
        print("World data found.")

    # ── Load starting location ──
    start = _world_ctx.load_location("worlds/han-dynasty/wudang-mountains/monastery.md")
    print(f"Starting: {start}")

    # ── Opening ephemeral context ──
    opening = ChatMessage.create_system_message(
        "[Ephemeral Context] This is the start of a new adventure. "
        "Li Wei and Mei Ling begin their morning training at the monastery. "
        "Set the scene with the mist-covered mountains, the sound of practice swords, "
        "and the scent of incense from the temple halls."
    )
    _msg_manager.add_message(opening, lifecycle=MessageLifecycle(ttl=2, on_expire=ExpiryAction.REMOVE))

    # ── Message TTL ──
    USER_TTL = 12
    ASSISTANT_TTL = 12

    print()
    print("=" * 64)
    print("  ⚔️  Virtual Game Master — Han Dynasty / Wudang Mountains")
    print("  ToolAgents + Agora KB + PromptComposer + XMLGameState")
    print("=" * 64)
    print()
    print("Commands:")
    print("  quit           — End the session")
    print("  /state         — Show game state (XML)")
    print("  /party         — Show party from game state")
    print("  /location      — Show current location")
    print("  /notes         — Show GM notes")
    print("  /archive       — Show archived messages")
    print("  /status        — Show system status")
    print("  /save [name]   — Save session")
    print("  /seed          — Re-seed world data")
    print()

    while True:
        try:
            user_input = input("\n⚔️  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nThe story pauses... for now.")
            break

        if not user_input:
            continue

        # ── Meta-commands ──
        if user_input.lower() == "quit":
            print("The story pauses... for now.")
            break

        elif user_input.lower() == "/state":
            print(f"\n📜 Game State:\n{_game_state.get_xml_string()}")
            continue

        elif user_input.lower() == "/party":
            pc = _game_state.get_element_text("player-character") or "Unknown"
            companions = _game_state.get_element_text("companions") or "None"
            inventory = _game_state.get_element_text("inventory") or "Empty"
            print(f"\n🧑 Player Character:\n{pc}")
            print(f"\n🤝 Companions:\n{companions}")
            print(f"\n🎒 Inventory:\n{inventory}")
            continue

        elif user_input.lower() == "/location":
            if _world_ctx.current_location_path:
                print(f"\n📍 {_world_ctx.current_location_title}")
                print(f"   Path: {_world_ctx.current_location_path}")
                print(f"   History: {' → '.join(_world_ctx.location_history[-5:])}")
            else:
                print("\n📍 No location loaded.")
            continue

        elif user_input.lower() == "/notes":
            print(f"\n📝 GM Notes:\n{_gm_memory.build_context()}")
            continue

        elif user_input.lower() == "/archive":
            print(f"\n📦 Archive ({len(_msg_manager.archive)} messages):")
            for i, msg in enumerate(_msg_manager.archive):
                print(f"  [{i}] {msg.get_as_text()[:100]}")
            if not _msg_manager.archive:
                print("  (empty)")
            continue

        elif user_input.lower() == "/status":
            print(f"\n📊 Status:")
            print(f"  Location: {_world_ctx.current_location_title}")
            print(f"  Active messages: {_msg_manager.message_count}")
            print(f"  Archived: {len(_msg_manager.archive)}")
            print(f"  GM notes: {len(_gm_memory.blocks)}")
            print(f"  Locations visited: {len(_world_ctx.location_history)}")
            continue

        elif user_input.lower().startswith("/save"):
            parts = user_input.split(maxsplit=1)
            name = parts[1] if len(parts) > 1 else "autosave"
            save_session(name, _game_state, _gm_memory, _world_ctx, _msg_manager)
            continue

        elif user_input.lower() == "/seed":
            seed_wudang_world(_kb_client)
            continue

        # ── Tick message lifecycles ──
        tick_result = _msg_manager.tick()
        if tick_result.removed:
            for m in tick_result.removed:
                print("  🗑️  [Expired] ephemeral context removed")
        if tick_result.archived:
            for m in tick_result.archived:
                print(f"  📦 [Archived] \"{m.get_as_text()[:60]}...\"")

        # ── Add user message ──
        user_msg = ChatMessage.create_user_message(user_input)
        _msg_manager.add_message(
            user_msg,
            lifecycle=MessageLifecycle(ttl=USER_TTL, on_expire=ExpiryAction.ARCHIVE),
        )

        # ── Update dynamic module prefixes ──
        composer.update_module(
            "gm_notes",
            prefix=f"### GM Notes [last modified: {_gm_memory.last_modified}]",
        )

        # ── Compile system prompt ──
        system_prompt = composer.compile()

        # ── Build messages ──
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

            assistant_msg = ChatMessage.create_assistant_message(response_text)
            _msg_manager.add_message(
                assistant_msg,
                lifecycle=MessageLifecycle(ttl=ASSISTANT_TTL, on_expire=ExpiryAction.ARCHIVE),
            )

            for msg in chat_response.messages:
                role = msg.get_role() if hasattr(msg, "get_role") else None
                if role not in ("user", "assistant"):
                    _msg_manager.add_message(
                        msg,
                        lifecycle=MessageLifecycle(ttl=ASSISTANT_TTL, on_expire=ExpiryAction.ARCHIVE),
                    )

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("  (Try rephrasing your action.)")

    _kb_client.close()


if __name__ == "__main__":
    main()

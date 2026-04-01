"""
=============================================================================
example_vgm_navigable.py — Virtual Game Master with Navigable World Memory
=============================================================================

A complete tabletop RPG game master combining:
  - VirtualGameMaster     → core game loop with save/load and history offset
  - NavigableMemory       → location-based world knowledge (auto-loading context)
  - PromptComposer        → modular dynamic system prompt
  - SmartMessageManager   → message lifecycles (TTL, archival, pinning)
  - CoreMemory            → GM notes and session observations
  - CommandSystem          → player meta-commands (@help, @save, etc.)

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                    PromptComposer                             │
  │  ┌───────────┬──────────┬──────────┬─────────┬────────────┐ │
  │  │ GM Prompt │ Game     │ Location │ GM Notes│ Session    │ │
  │  │ pos=0     │ State    │ Context  │ pos=15  │ Metadata   │ │
  │  │ (static)  │ pos=5    │ pos=10   │(dynamic)│ pos=20     │ │
  │  │           │(dynamic) │(dynamic) │         │ (dynamic)  │ │
  │  └───────────┴──────────┴──────────┴─────────┴────────────┘ │
  └──────────────────────┬───────────────────────────────────────┘
                         │ compile() each turn
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │               SmartMessageManager                             │
  │  [pinned: rules] [user TTL=12] [assistant TTL=12]            │
  │  [departed locations TTL=8 → archive]                        │
  └──────────────────────┬───────────────────────────────────────┘
                         │ system_prompt + active_messages
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  ChatToolAgent.get_response()                                │
  │  Tools: set_location, list_locations, search_world,          │
  │         read_document, write_document, append_to_document,   │
  │         add_gm_note, search_gm_notes                         │
  └──────────────────────────────────────────────────────────────┘

Usage:
  1. pip install ToolAgents pydantic python-dotenv
  2. Set OPENROUTER_API_KEY (or GROQ_API_KEY) in .env
  3. python example_vgm_navigable.py

  Try:
    "Let's start the adventure!"
    "I want to explore the monastery grounds"
    "Let me meditate at the cliff edge"
    "I want to go down to the village"
    @help              — show all commands
    @save              — manual save
    @location          — current location + history
    @notes             — GM notes
    @tree              — world knowledge tree
"""

import json
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Tuple, Generator, Dict, Any, Optional, List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── ToolAgents imports ──────────────────────────────────────────────
from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage, TextContent
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.agent_harness.prompt_composer import PromptComposer
from ToolAgents.agent_harness.smart_messages import (
    SmartMessageManager,
    MessageLifecycle,
    ExpiryAction,
)
from ToolAgents.provider import OpenAIChatAPI

# ── NavigableMemory ─────────────────────────────────────────────────
from ToolAgents.agent_memory.navigable_memory import (
    NavigableMemory,
    InMemoryBackend,
    DepartureRecord,
)

load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# PART 1: CoreMemory — GM notes and session observations
# ═══════════════════════════════════════════════════════════════════

class CoreMemory:
    """Freeform key-value memory for the GM's observations and notes."""

    def __init__(self, block_limit: int = 500):
        self.blocks: Dict[str, str] = {}
        self.block_limit = block_limit
        self.last_modified: str = "never"

    def set_block(self, name: str, content: str) -> str:
        if len(content) > self.block_limit:
            return f"Error: exceeds {self.block_limit} char limit."
        self.blocks[name] = content
        self.last_modified = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Note '{name}' saved."

    def append_block(self, name: str, content: str) -> str:
        current = self.blocks.get(name, "")
        new = current + content
        if len(new) > self.block_limit:
            return f"Error: would exceed {self.block_limit} chars."
        self.blocks[name] = new
        self.last_modified = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Appended to '{name}'."

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
        return f"Found {len(results)} note(s):\n" + "\n".join(results) if results else f"No notes matching '{query}'."

    def build_context(self) -> str:
        if not self.blocks:
            return "No GM notes stored yet."
        return "\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in self.blocks.items())

    def to_dict(self) -> dict:
        return {"blocks": dict(self.blocks), "last_modified": self.last_modified}

    def from_dict(self, data: dict):
        self.blocks = data.get("blocks", {})
        self.last_modified = data.get("last_modified", "restored")


# ═══════════════════════════════════════════════════════════════════
# PART 2: Game State — structured XML-like state tracking
# ═══════════════════════════════════════════════════════════════════

class GameState:
    """Simple structured game state with XML-style rendering."""

    def __init__(self):
        self.fields: Dict[str, Any] = {}

    def set_field(self, key: str, value: Any):
        self.fields[key] = value

    def get_field(self, key: str, default=None):
        return self.fields.get(key, default)

    def update_from_dict(self, data: Dict[str, Any]):
        self.fields.update(data)

    def build_context(self) -> str:
        if not self.fields:
            return "<game-state>(empty)</game-state>"
        lines = ["<game-state>"]
        for key, value in self.fields.items():
            if isinstance(value, list):
                lines.append(f"  <{key}>")
                for item in value:
                    lines.append(f"    <item>{item}</item>")
                lines.append(f"  </{key}>")
            elif isinstance(value, dict):
                lines.append(f"  <{key}>")
                for k, v in value.items():
                    lines.append(f"    <{k}>{v}</{k}>")
                lines.append(f"  </{key}>")
            else:
                lines.append(f"  <{key}>{value}</{key}>")
        lines.append("</game-state>")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dict(self.fields)

    def from_dict(self, data: dict):
        self.fields = dict(data)


# ═══════════════════════════════════════════════════════════════════
# PART 3: Command System (simplified inline version)
# ═══════════════════════════════════════════════════════════════════

class CommandRegistry:
    """Simple command registry for player meta-commands."""

    def __init__(self, prefix: str = "@"):
        self.prefix = prefix
        self.commands: Dict[str, Dict] = {}

    def command(self, name: str, description: str = ""):
        def decorator(func):
            self.commands[name] = {"func": func, "description": description}
            return func
        return decorator

    def handle(self, vgm, raw_input: str) -> Optional[Tuple[str, bool]]:
        if not raw_input.startswith(self.prefix):
            return None
        parts = raw_input[len(self.prefix):].strip().split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        args_str = parts[1] if len(parts) > 1 else ""

        if cmd_name not in self.commands:
            return f"Unknown command: {cmd_name}. Type {self.prefix}help for available commands.", False

        func = self.commands[cmd_name]["func"]
        if args_str:
            # Simple arg splitting — split on spaces, respecting quoted strings
            args = args_str.split()
            return func(vgm, *args)
        return func(vgm)


# Global command registry
commands = CommandRegistry(prefix="@")


# ═══════════════════════════════════════════════════════════════════
# PART 4: Instances
# ═══════════════════════════════════════════════════════════════════

gm_memory = CoreMemory(block_limit=500)
game_state = GameState()
message_manager = SmartMessageManager()
backend = InMemoryBackend()


def on_location_depart(record: DepartureRecord):
    """Inject departed location as a TTL message for rolling context."""
    snippet = record.content[:200].replace("\n", " ")
    msg = ChatMessage.create_system_message(
        f"[Previous Location] {record.title} ({record.path})\n{snippet}..."
    )
    message_manager.add_message(
        msg,
        lifecycle=MessageLifecycle(ttl=8, on_expire=ExpiryAction.ARCHIVE),
    )
    print(f"  📍 [Departed] {record.title}")


nav_memory = NavigableMemory(
    backend=backend,
    on_depart=on_location_depart,
    context_window=3,
    include_siblings=True,
    include_parent=True,
)


# ═══════════════════════════════════════════════════════════════════
# PART 5: GM Tools — what the Game Master LLM can call
# ═══════════════════════════════════════════════════════════════════

class AddGMNote(BaseModel):
    """Save a GM note for plot threads, NPC secrets, or session observations."""
    note_name: str = Field(..., description="Short name (e.g., 'plot_thread_cave', 'npc_secret').")
    content: str = Field(..., description="Note content (max 500 chars).")

    def run(self) -> str:
        return gm_memory.set_block(self.note_name, self.content)


class SearchGMNotes(BaseModel):
    """Search through GM notes for a keyword."""
    query: str = Field(..., description="Search term.")

    def run(self) -> str:
        return gm_memory.search(self.query)


class UpdateGameState(BaseModel):
    """Update a game state field (inventory, quests, HP, etc.)."""
    field_name: str = Field(..., description="Field to update (e.g., 'active_quests', 'inventory').")
    value: str = Field(..., description="New value (string, will be stored as-is).")

    def run(self) -> str:
        game_state.set_field(self.field_name, self.value)
        return f"Game state '{self.field_name}' updated."


class ArchiveSearch(BaseModel):
    """Search archived messages for past conversations and departed locations."""
    query: str = Field(..., description="Search term.")

    def run(self) -> str:
        results = []
        for msg in message_manager.archive:
            text = msg.get_as_text()
            if self.query.lower() in text.lower():
                results.append(text[:200])
        if results:
            return f"Found {len(results)} archived item(s):\n" + "\n---\n".join(results[:5])
        return f"No archived items matching '{self.query}'."


# ═══════════════════════════════════════════════════════════════════
# PART 6: World Seeder
# ═══════════════════════════════════════════════════════════════════

def seed_world():
    """Seed the knowledge base with a Wudang Mountains / Han Dynasty world."""

    documents = [
        ("worlds/han-dynasty/overview.md", "Han Dynasty China",
         "# Han Dynasty China — circa 200 CE\n\n"
         "The Han Dynasty crumbles. Emperor Xian is a puppet, controlled by\n"
         "warlords. The Yellow Turban Rebellion has shaken the foundations.\n\n"
         "## Factions\n"
         "- **Imperial Court** — weakening but legitimate\n"
         "- **Warlords** — Cao Cao, Sun Quan, Liu Bei vie for supremacy\n"
         "- **Yellow Turbans** — remnants in secret cells\n"
         "- **Taoist Sects** — hidden knowledge, martial arts\n"
         "- **Wu Xia** — wandering martial artists",
         ["world", "han-dynasty"]),

        ("worlds/han-dynasty/wudang/overview.md", "Wudang Mountains",
         "# The Wudang Mountains\n\n"
         "Sacred peaks shrouded in mist, home to Taoist monasteries.\n"
         "Both spiritual refuge and strategic waypoint.\n\n"
         "## Dangers\n"
         "- Bandits prey on pilgrims and traders\n"
         "- Wild animals: tigers, bears, venomous snakes\n"
         "- Treacherous paths, sudden storms\n"
         "- Not all monks are what they seem",
         ["region", "wudang"]),

        ("worlds/han-dynasty/wudang/monastery.md", "Wudang Monastery — Temple of the Purple Cloud",
         "# Wudang Monastery\n\n"
         "A remote Taoist monastery on a cliff face, accessible only\n"
         "by a narrow stone stairway carved into the mountain.\n\n"
         "## Notable NPCs\n"
         "- **Master Chen** — enigmatic head, speaks in riddles\n"
         "- **Brother Fang** — martial arts instructor, gruff, missing left ear\n"
         "- **Sister Yue** — herbalist and scroll-keeper, quiet and observant\n\n"
         "## Locations Within\n"
         "- Main training courtyard\n"
         "- Hall of Eternal Harmony (worship hall)\n"
         "- Medicinal Garden (Sister Yue's domain)\n"
         "- Cliff Edge Meditation Platform\n"
         "- Library Wing (scrolls and bound texts)\n"
         "- Abbot's Quarters (Master Chen)\n\n"
         "## Secrets\n"
         "Guards the Taixuan Jing — prophecies about the dynasty's fall.\n"
         "Master Chen has allowed only fragments to be read.",
         ["location", "monastery", "wudang"]),

        ("worlds/han-dynasty/wudang/mountain-path.md", "The Pilgrim's Ascent",
         "# The Pilgrim's Ascent\n\n"
         "Winding stone path from foothills to monastery. Half a day on foot.\n"
         "Passes through bamboo groves, over rope bridges, along cliff edges.\n\n"
         "## Waypoints\n"
         "- **First Gate** — stone archway with carved dragons\n"
         "- **Waterfall Rest** — shrine beside a waterfall\n"
         "- **The Narrow** — two feet wide, sheer drop, ambush point\n\n"
         "## Encounters\n"
         "- Pilgrims and traders (peaceful)\n"
         "- Mountain bandits (especially at The Narrow)\n"
         "- Wandering Wu Xia seeking the monastery",
         ["location", "path", "wudang"]),

        ("worlds/han-dynasty/wudang/hidden-cave.md", "The Whispering Grotto",
         "# The Whispering Grotto\n\n"
         "Natural cave system behind the waterfall on the Pilgrim's Ascent.\n"
         "Entrance hidden behind the curtain of water.\n\n"
         "## Features\n"
         "- **Ancient Carvings** — astronomical charts, ritual instructions\n"
         "- **Inner Chamber** — circular room with jade altar, faint glow beneath\n"
         "- **Underground Stream** — sounds like whispered words\n\n"
         "## The Artifact\n"
         "Rumors of the Kunlun Mirror — a bronze mirror said to reveal\n"
         "truths the eye cannot see. Whether this is the artifact\n"
         "the party seeks remains to be discovered.",
         ["location", "cave", "secret", "wudang"]),

        ("worlds/han-dynasty/wudang/village.md", "Three Pines Village",
         "# Three Pines Village\n\n"
         "Small farming village at the base of the Wudang Mountains.\n"
         "Named for three ancient pine trees in the village square.\n\n"
         "## Notable NPCs\n"
         "- **Elder Wu** — village headman, worried about bandit raids\n"
         "- **Blacksmith Zhao** — repairs weapons, secretly a deserter from Cao Cao's army\n"
         "- **Tea House Auntie** — gossip hub, knows everything\n\n"
         "## Services\n"
         "- Rooms: 2 copper/night\n"
         "- Meals: 1 copper\n"
         "- Weapon repair: negotiate with Zhao\n\n"
         "## Current Tensions\n"
         "Bandits demanding 'protection fees'. Villagers can barely afford it.",
         ["location", "village", "wudang"]),

        ("worlds/han-dynasty/wudang/bandit-camp.md", "Mountain Bandit Camp",
         "# Mountain Bandit Camp\n\n"
         "Hidden camp in a ravine two hours east of the Pilgrim's Ascent.\n"
         "About 30 bandits led by 'Iron Claw' Zhang.\n\n"
         "## Iron Claw Zhang\n"
         "Former soldier, brutal but cunning. Missing three fingers\n"
         "on his right hand (replaced with iron prosthetics).\n"
         "Has a grudge against the monastery — was expelled as a novice.\n\n"
         "## Defenses\n"
         "- Sentries on the ridge above\n"
         "- Rope traps on approach paths\n"
         "- Attack dogs\n\n"
         "## Loot\n"
         "Stolen goods from traders, some monastery scrolls,\n"
         "a jade seal of unknown origin.",
         ["location", "bandits", "wudang", "combat"]),
    ]

    for path, title, content, tags in documents:
        nav_memory.write(path, title, content, tags)

    print(f"  Seeded {backend.document_count} world documents.")


# ═══════════════════════════════════════════════════════════════════
# PART 7: Default game state
# ═══════════════════════════════════════════════════════════════════

def seed_game_state():
    game_state.update_from_dict({
        "setting": "Ancient China, late Han Dynasty (circa 200 CE)",
        "player_character": (
            "Li Wei (李威): Scholar-Warrior, age 32. "
            "Intelligent dark eyes, trimmed beard. "
            "Scholar's robes with hidden armor. Carries a jian."
        ),
        "companions": [
            "Mei Ling (梅玲): Wu Xia, age 29. Butterfly swords, acrobatics, dry humor."
        ],
        "location": "Wudang Monastery — Temple of the Purple Cloud",
        "active_quests": [
            "Decode the Taixuan Jing prophecy",
            "Investigate the artifact in the Wudang Mountains",
            "Prevent an assassination of a political figure",
        ],
        "inventory": "Jian, scrolls, writing kit, hidden armor, medicinal herbs",
        "hp": "Li Wei: 100/100, Mei Ling: 85/85",
        "gold": "15 gold, 40 copper",
    })


# ═══════════════════════════════════════════════════════════════════
# PART 8: Prompt Composer
# ═══════════════════════════════════════════════════════════════════

def build_prompt_composer() -> PromptComposer:
    composer = PromptComposer()

    composer.add_module("gm_instructions", position=0, content=(
        "You are an expert Game Master running a tabletop RPG.\n\n"
        "RULES:\n"
        "1. Begin responses with 'Current Location: <name>' on its own line.\n"
        "2. Use navigate_to_document to travel — list_locations to discover paths.\n"
        "3. Use update_game_state to track HP, items, gold, quests.\n"
        "4. Use add_gm_note to record plot threads, NPC secrets, session observations.\n"
        "5. Use search_knowledge and read_document to look up world lore.\n"
        "6. Describe scenes vividly. All senses. NPCs have distinct voices.\n"
        "7. Present meaningful choices — combat, social, exploration, mystery.\n"
        "8. Respect the historical setting. This is Wuxia martial arts China.\n\n"
        "WORLD NAVIGATION:\n"
        "The world is a hierarchy of markdown documents:\n"
        "  worlds/<world>/<region>/place.md\n"
        "Use list_locations with a prefix to discover locations.\n"
        "Location context loads automatically when you navigate."
    ))

    composer.add_module("game_state", position=5,
                        content_fn=game_state.build_context,
                        prefix="### Game State",
                        suffix="### End Game State")

    composer.add_module("location_context", position=10,
                        content_fn=nav_memory.build_context,
                        prefix="### World — Current Location",
                        suffix="### End World")

    composer.add_module("gm_notes", position=15,
                        content_fn=gm_memory.build_context,
                        prefix=f"### GM Notes [modified: {gm_memory.last_modified}]",
                        suffix="### End GM Notes")

    turn_counter = {"n": 0}

    def metadata_fn():
        turn_counter["n"] += 1
        loc = nav_memory.current_title if nav_memory.current_path else "None"
        return (
            f"Turn: {turn_counter['n']}\n"
            f"Location: {loc}\n"
            f"Active messages: {message_manager.message_count}\n"
            f"Archived: {len(message_manager.archive)}"
        )

    composer.add_module("metadata", position=20,
                        content_fn=metadata_fn,
                        prefix="### Session", suffix="### End Session")

    return composer


# ═══════════════════════════════════════════════════════════════════
# PART 9: Commands
# ═══════════════════════════════════════════════════════════════════

@commands.command("exit", "Save and exit the game.")
def cmd_exit(vgm):
    save_session()
    return "Game saved. Goodbye!", True


@commands.command("save", "Manually save the game.")
def cmd_save(vgm):
    save_session()
    return "💾 Game saved!", False


@commands.command("state", "Show the current game state.")
def cmd_state(vgm):
    return f"\n📜 Game State:\n{game_state.build_context()}", False


@commands.command("party", "Show party members, HP, and inventory.")
def cmd_party(vgm):
    pc = game_state.get_field("player_character", "Unknown")
    companions = game_state.get_field("companions", [])
    hp = game_state.get_field("hp", "Unknown")
    gold = game_state.get_field("gold", "Unknown")
    inv = game_state.get_field("inventory", "Empty")
    comp_str = "\n".join(f"  - {c}" for c in companions) if isinstance(companions, list) else str(companions)
    return (
        f"\n🧑 Player: {pc}\n"
        f"\n🤝 Companions:\n{comp_str}\n"
        f"\n❤️  HP: {hp}\n"
        f"💰 Gold: {gold}\n"
        f"🎒 Inventory: {inv}"
    ), False


@commands.command("location", "Show current world location and history.")
def cmd_location(vgm):
    if nav_memory.current_path:
        history = nav_memory.history[-5:]
        history_str = " → ".join(p.split("/")[-1] for p in history)
        return (
            f"\n📍 {nav_memory.current_title}\n"
            f"   Path: {nav_memory.current_path}\n"
            f"   History: {history_str}"
        ), False
    return "\n📍 No location loaded.", False


@commands.command("notes", "Show GM notes.")
def cmd_notes(vgm):
    return f"\n📝 GM Notes:\n{gm_memory.build_context()}", False


@commands.command("archive", "Show archived messages.")
def cmd_archive(vgm):
    archive = message_manager.archive
    if not archive:
        return "\n📦 Archive: (empty)", False
    lines = [f"\n📦 Archive ({len(archive)} items):"]
    for i, msg in enumerate(archive[-10:]):
        lines.append(f"  [{i}] {msg.get_as_text()[:100]}")
    return "\n".join(lines), False


@commands.command("status", "Show system status.")
def cmd_status(vgm):
    sms = message_manager.get_smart_messages()
    pinned = sum(1 for sm in sms if sm.lifecycle.pinned)
    ephemeral = sum(1 for sm in sms if not sm.lifecycle.is_permanent and not sm.lifecycle.pinned)
    return (
        f"\n📊 Status:\n"
        f"  Location: {nav_memory.current_title}\n"
        f"  Active messages: {message_manager.message_count} (pinned: {pinned}, ephemeral: {ephemeral})\n"
        f"  Archived: {len(message_manager.archive)}\n"
        f"  GM notes: {len(gm_memory.blocks)}\n"
        f"  World documents: {backend.document_count}\n"
        f"  Locations visited: {len(nav_memory.history)}"
    ), False


@commands.command("tree", "Show world knowledge tree.")
def cmd_tree(vgm):
    docs = sorted(nav_memory.list_at(""), key=lambda d: d.path)
    lines = ["\n🌳 World:"]
    current_dir = ""
    for d in docs:
        parts = d.path.rsplit("/", 1)
        dir_part = parts[0] + "/" if len(parts) > 1 else ""
        if dir_part != current_dir:
            current_dir = dir_part
            lines.append(f"\n  📁 {current_dir}")
        marker = " ◀ HERE" if d.path == nav_memory.current_path else ""
        lines.append(f"      {d.path.split('/')[-1]:35s} {d.title}{marker}")
    return "\n".join(lines), False


@commands.command("history", "Show recent chat history.")
def cmd_history(vgm, count: str = "10"):
    n = int(count)
    msgs = message_manager.get_active_messages()[-n:]
    lines = [f"\nLast {min(n, len(msgs))} messages:"]
    for msg in msgs:
        role = msg.get_role()
        text = msg.get_as_text()[:120].replace("\n", " ")
        lines.append(f"  [{role}] {text}")
    return "\n".join(lines), False


@commands.command("help", "Show all commands.")
def cmd_help(vgm):
    lines = ["\nAvailable commands:"]
    for name, info in sorted(commands.commands.items()):
        lines.append(f"  {commands.prefix}{name:15s} — {info['description']}")
    return "\n".join(lines), False


# ═══════════════════════════════════════════════════════════════════
# PART 10: Persistence
# ═══════════════════════════════════════════════════════════════════

SAVE_DIR = "game_saves"
SAVE_FILE = os.path.join(SAVE_DIR, "session.json")
HISTORY_FILE = os.path.join(SAVE_DIR, "history.json")


def save_session():
    os.makedirs(SAVE_DIR, exist_ok=True)
    state = {
        "game_state": game_state.to_dict(),
        "gm_memory": gm_memory.to_dict(),
        "current_location": nav_memory.current_path,
        "location_history": nav_memory.history,
        "active_messages": [
            {
                "role": sm.message.role.value,
                "text": sm.message.get_as_text(),
                "ttl": sm.lifecycle.ttl,
                "turns_alive": sm.lifecycle.turns_alive,
                "pinned": sm.lifecycle.pinned,
                "on_expire": sm.lifecycle.on_expire.value,
            }
            for sm in message_manager.get_smart_messages()
        ],
        "archive": [m.get_as_text() for m in message_manager.archive],
        "saved_at": dt.datetime.now().isoformat(),
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_session() -> bool:
    if not os.path.exists(SAVE_FILE):
        return False
    try:
        with open(SAVE_FILE, "r") as f:
            state = json.load(f)

        game_state.from_dict(state["game_state"])
        gm_memory.from_dict(state["gm_memory"])

        message_manager.clear()
        for md in state["active_messages"]:
            if md["role"] == "user":
                msg = ChatMessage.create_user_message(md["text"])
            elif md["role"] == "assistant":
                msg = ChatMessage.create_assistant_message(md["text"])
            else:
                msg = ChatMessage.create_system_message(md["text"])
            lifecycle = MessageLifecycle(
                ttl=md["ttl"], turns_alive=md["turns_alive"],
                pinned=md["pinned"], on_expire=ExpiryAction(md["on_expire"]),
            )
            message_manager.add_message(msg, lifecycle)

        if state.get("current_location"):
            nav_memory.navigate(state["current_location"])

        print(f"  📂 Restored from {state.get('saved_at', 'unknown')}")
        print(f"  Game state: {len(game_state.fields)} fields")
        print(f"  GM notes: {len(gm_memory.blocks)} blocks")
        print(f"  Messages: {message_manager.message_count} active")
        print(f"  Location: {nav_memory.current_title or 'none'}")
        return True
    except Exception as e:
        print(f"  ⚠️ Failed to load save: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# PART 11: Message helpers
# ═══════════════════════════════════════════════════════════════════

USER_TTL = 12
ASSISTANT_TTL = 12


def add_user_msg(text: str):
    msg = ChatMessage.create_user_message(text)
    message_manager.add_message(msg, MessageLifecycle(ttl=USER_TTL, on_expire=ExpiryAction.ARCHIVE))


def add_assistant_msg(text: str):
    msg = ChatMessage.create_assistant_message(text)
    message_manager.add_message(msg, MessageLifecycle(ttl=ASSISTANT_TTL, on_expire=ExpiryAction.ARCHIVE))


def inject_ephemeral(text: str, ttl: int = 2):
    msg = ChatMessage.create_system_message(f"[Ephemeral] {text}")
    message_manager.add_message(msg, MessageLifecycle(ttl=ttl, on_expire=ExpiryAction.REMOVE))


# ═══════════════════════════════════════════════════════════════════
# PART 12: Main
# ═══════════════════════════════════════════════════════════════════

def main():
    # Seed world (always — InMemoryBackend doesn't persist)
    seed_world()

    # Try loading previous session
    restored = load_session()
    if not restored:
        seed_game_state()
        gm_memory.set_block("session_start", f"New game started {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

        message_manager.add_message(
            ChatMessage.create_system_message(
                "[SYSTEM] You are the Game Master. Stay in character. Use tools to track state. "
                "Always begin responses with 'Current Location: <name>'."
            ),
            MessageLifecycle(pinned=True),
        )

        inject_ephemeral(
            "New adventure begins. Li Wei and Mei Ling start their morning "
            "at the monastery. Set the scene with mist-covered mountains, "
            "practice swords, and incense.",
            ttl=2,
        )

        nav_memory.navigate("worlds/han-dynasty/wudang/monastery.md")

    # Provider
    api = OpenAIChatAPI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("MODEL", "openai/gpt-4o-mini"),
        base_url=os.getenv("API_URL", "https://openrouter.ai/api/v1"),
    )

    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    settings.top_p = float(os.getenv("TOP_P", "0.95"))

    # Tools
    tool_registry = ToolRegistry()
    nav_tools = [FunctionTool(t) for t in nav_memory.create_tools()]
    tool_registry.add_tools(nav_tools)
    tool_registry.add_tools([
        FunctionTool(AddGMNote),
        FunctionTool(SearchGMNotes),
        FunctionTool(UpdateGameState),
        FunctionTool(ArchiveSearch),
    ])

    # Composer
    composer = build_prompt_composer()

    # Banner
    loc = nav_memory.current_title or "Unknown"
    print()
    print("=" * 64)
    print(f"  ⚔️  Virtual Game Master — Han Dynasty / Wudang Mountains")
    print(f"  NavigableMemory + PromptComposer + SmartMessages")
    print(f"  World: {backend.document_count} documents | Session: {'restored' if restored else 'new'}")
    print("=" * 64)
    print(f"\n  Starting at: {loc}")
    print(f"\n  Type @help for commands, or just play!")
    print()

    while True:
        try:
            user_input = input("\n⚔️  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            save_session()
            print("\n  💾 Saved. The story pauses... for now.")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.startswith(commands.prefix):
            result = commands.handle(None, user_input)
            if result:
                text, should_exit = result
                print(text)
                if should_exit:
                    break
            continue

        # ── Tick message lifecycles ──
        tick_result = message_manager.tick()
        if tick_result.removed:
            for m in tick_result.removed:
                print("  🗑️  [Expired] ephemeral context removed")
        if tick_result.archived:
            for m in tick_result.archived:
                print(f"  📦 [Archived] \"{m.get_as_text()[:50]}...\"")

        # ── Add user message ──
        add_user_msg(user_input)

        # ── Update dynamic prefixes ──
        composer.update_module(
            "gm_notes",
            prefix=f"### GM Notes [modified: {gm_memory.last_modified}]",
        )

        # ── Compile and send ──
        system_prompt = composer.compile()
        messages = [
            ChatMessage.create_system_message(system_prompt),
            *message_manager.get_active_messages(),
        ]

        try:
            chat_response = agent.get_response(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
            )

            response_text = chat_response.response.strip()
            print(f"\n🎭 GM > {response_text}")

            add_assistant_msg(response_text)

            for msg in chat_response.messages:
                if msg.role.value not in ("user", "assistant"):
                    message_manager.add_message(
                        msg,
                        lifecycle=MessageLifecycle(ttl=ASSISTANT_TTL, on_expire=ExpiryAction.ARCHIVE),
                    )

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("  (Try rephrasing your action.)")


if __name__ == "__main__":
    main()

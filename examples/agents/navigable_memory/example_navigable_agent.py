"""
=============================================================================
example_navigable_agent.py — Personal Agent with Navigable Knowledge Space
=============================================================================

Demonstrates how to combine:
  - NavigableMemory      → location-based knowledge loading (from dao_framework)
  - CoreMemory           → self-editable persistent notes
  - PromptComposer       → modular, dynamic system prompt
  - SmartMessageManager  → messages with TTL, archival, pinning
  - ChatToolAgent        → LLM-driven tool calling

The agent ("Sage") can:
  1. Navigate a knowledge space — move between documents, context loads automatically
  2. Edit its own memory — store observations, user preferences, project state
  3. Search and read — find documents without navigating to them
  4. Write and append — create new knowledge entries or log events
  5. Archive and recall — old messages archive naturally, searchable later

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                    PromptComposer                             │
  │  ┌─────────────┬────────────┬──────────┬──────────┬────────┐│
  │  │ Instructions│ Core Memory│ Location │ History  │Metadata ││
  │  │ pos=0       │ pos=5      │ pos=10   │ pos=15   │pos=20  ││
  │  │ (static)    │ (dynamic)  │ (dynamic)│ (dynamic)│(dynamic)││
  │  └─────────────┴────────────┴──────────┴──────────┴────────┘│
  └──────────────────────┬───────────────────────────────────────┘
                         │ compile() → system message
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │               SmartMessageManager                             │
  │  ┌────────┬───────────┬──────────┬─────────┬───────────────┐ │
  │  │ Pinned │ Ephemeral │ Archival │ Normal  │ Departed Locs │ │
  │  │ (∞)    │ (TTL=2)   │ (TTL→📦) │         │ (TTL→📦)      │ │
  │  └────────┴───────────┴──────────┴─────────┴───────────────┘ │
  └──────────────────────┬───────────────────────────────────────┘
                         │ get_active_messages()
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │            ChatToolAgent.get_response()                       │
  │  Tools: navigate, navigate_up, list, search, read, write,    │
  │         append, core_memory_set/append/delete, archive_search │
  └──────────────────────────────────────────────────────────────┘

Usage:
  1. pip install ToolAgents pydantic python-dotenv
  2. Set GROQ_API_KEY or OPENROUTER_API_KEY in .env
  3. python example_navigable_agent.py

The agent starts with a pre-seeded knowledge base about your projects,
people, and interests. Navigate with natural language — the agent calls
the tools automatically.
"""

import json
import os
from datetime import datetime
from typing import Optional

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
from ToolAgents.provider import OpenAIChatAPI

from ToolAgents.agent_memory.navigable_memory import (
    NavigableMemory,
    InMemoryBackend,
    DepartureRecord,
    SQLiteBackend,
    AgoraBackend
)

load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# PART 1: Core Memory — self-editable persistent notes
# ═══════════════════════════════════════════════════════════════════

class CoreMemory:
    """Key-value memory for agent observations and user preferences."""

    def __init__(self, block_limit: int = 500):
        self.blocks: dict[str, str] = {}
        self.block_limit = block_limit
        self.last_modified: str = "never"

    def set_block(self, name: str, content: str) -> str:
        if len(content) > self.block_limit:
            return f"Error: exceeds {self.block_limit} char limit (got {len(content)})."
        self.blocks[name] = content
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Memory block '{name}' updated."

    def append_block(self, name: str, content: str) -> str:
        current = self.blocks.get(name, "")
        new = current + content if current else content
        if len(new) > self.block_limit:
            return f"Error: would exceed {self.block_limit} chars."
        self.blocks[name] = new
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Appended to '{name}'."

    def get_block(self, name: str) -> str:
        return self.blocks.get(name, f"Block '{name}' not found.")

    def delete_block(self, name: str) -> str:
        if name in self.blocks:
            del self.blocks[name]
            self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"Block '{name}' deleted."
        return f"Block '{name}' not found."

    def build_context(self) -> str:
        if not self.blocks:
            return "<empty>No memory blocks stored yet.</empty>"
        lines = []
        for key, value in self.blocks.items():
            lines.append(f"<{key}> ({len(value)}/{self.block_limit} chars)\n{value}\n</{key}>")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# PART 2: Create instances
# ═══════════════════════════════════════════════════════════════════

# Core memory
core_memory = CoreMemory(block_limit=500)

# Smart message manager
message_manager = SmartMessageManager()

# Navigable memory (in-memory backend for this example)
# To use Agora KB instead:
#   backend = AgoraBackend(project_slug="personal")
backend = InMemoryBackend()


def on_location_depart(record: DepartureRecord):
    """When leaving a location, inject it as a TTL message for context continuity."""
    snippet = record.content[:200].replace("\n", " ")
    msg = ChatMessage.create_system_message(
        f"[Previous Location] {record.title} ({record.path})\n{snippet}..."
    )
    message_manager.add_message(
        msg,
        lifecycle=MessageLifecycle(
            ttl=12,
            on_expire=ExpiryAction.ARCHIVE,
        ),
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
# PART 3: Memory Tools — Pydantic models for core memory editing
# ═══════════════════════════════════════════════════════════════════

class CoreMemorySet(BaseModel):
    """Set or overwrite a named block in core memory.
    Use to store user preferences, observations, or project state."""
    block_name: str = Field(..., description="Block name (e.g., 'user_info', 'preferences').")
    content: str = Field(..., description="Content to store (max 500 chars).")

    def run(self) -> str:
        return core_memory.set_block(self.block_name, self.content)


class CoreMemoryAppend(BaseModel):
    """Append text to an existing core memory block."""
    block_name: str = Field(..., description="Block name to append to.")
    content: str = Field(..., description="Text to append.")

    def run(self) -> str:
        return core_memory.append_block(self.block_name, self.content)


class CoreMemoryDelete(BaseModel):
    """Delete a core memory block that is no longer relevant."""
    block_name: str = Field(..., description="Block name to delete.")

    def run(self) -> str:
        return core_memory.delete_block(self.block_name)


class ArchiveSearch(BaseModel):
    """Search the message archive for past conversations and departed location context."""
    query: str = Field(..., description="Search term.")

    def run(self) -> str:
        results = []
        for msg in message_manager.archive:
            text = msg.get_as_text()
            if self.query.lower() in text.lower():
                results.append(text[:200])
        if results:
            return f"Found {len(results)} archived item(s):\n" + "\n---\n".join(results)
        return f"No archived items matching '{self.query}'."


# ═══════════════════════════════════════════════════════════════════
# PART 4: Seed the knowledge space
# ═══════════════════════════════════════════════════════════════════

def seed_knowledge_base():
    """Populate the in-memory backend with example knowledge.

    This seeds a fictional indie developer's knowledge space to
    demonstrate the navigation patterns. Replace with your own
    documents for real use.
    """

    documents = [
        # ── Projects overview ──
        ("projects/overview.md", "Projects Overview",
         "# Active Projects\n\n"
         "Three active projects, each at different stages:\n"
         "- **Starfall** — 2D roguelike game, Steam release in June\n"
         "- **API Gateway** — open-source API gateway library\n"
         "- **Blog** — technical blog on game dev and systems programming",
         ["projects"]),

        # ── Starfall (game project) ──
        ("projects/starfall/overview.md", "Starfall Game",
         "# Starfall — 2D Roguelike\n\n"
         "A procedurally generated space roguelike built with Godot 4.\n"
         "Steam page live. Release planned for June 2026.\n\n"
         "## Team\n"
         "- Lead dev: me\n"
         "- Art: freelancer (Mika)\n"
         "- Music: freelancer (Jordan)\n\n"
         "## Priorities\n"
         "1. Boss fight balancing — too easy right now\n"
         "2. Steam achievements integration\n"
         "3. Trailer for marketing push",
         ["game", "godot", "project"]),

        ("projects/starfall/boss-balance.md", "Boss Fight Balance",
         "# Boss Fight Balancing\n\n"
         "Current bosses are too easy after the weapon rework.\n\n"
         "## Issues\n"
         "- Phase 2 transitions too slow, player can burst down\n"
         "- Shield mechanic not punishing enough\n"
         "- Final boss lacks a desperation phase\n\n"
         "## Ideas\n"
         "- Add enrage timer to Phase 2\n"
         "- Shield reflects projectiles at 50% HP\n"
         "- Final boss spawns minions below 20% HP\n\n"
         "## Status: IN PROGRESS",
         ["game", "balance", "bosses"]),

        ("projects/starfall/steam-integration.md", "Steam Integration",
         "# Steam Integration\n\n"
         "Using GodotSteam plugin for achievements and leaderboards.\n\n"
         "## Status\n"
         "- Achievements: 12/20 implemented\n"
         "- Leaderboards: not started\n"
         "- Cloud saves: working\n\n"
         "## Blocked On\n"
         "Need final achievement list from design doc.",
         ["game", "steam", "integration"]),

        # ── API Gateway ──
        ("projects/api-gateway/overview.md", "API Gateway Library",
         "# API Gateway — Open Source Library\n\n"
         "A lightweight API gateway library in Rust.\n"
         "Handles rate limiting, auth, and request routing.\n\n"
         "## Stats\n"
         "- 340 GitHub stars\n"
         "- 12 contributors\n"
         "- Latest: v0.8.2\n\n"
         "## Roadmap\n"
         "1. WebSocket support (v0.9)\n"
         "2. Plugin system (v1.0)\n"
         "3. Dashboard UI (post-v1.0)",
         ["rust", "api", "opensource"]),

        ("projects/api-gateway/websocket.md", "WebSocket Support",
         "# WebSocket Support (v0.9)\n\n"
         "Adding WebSocket proxying and connection management.\n\n"
         "## Design\n"
         "- Connection upgrade handling\n"
         "- Per-connection rate limits\n"
         "- Heartbeat monitoring\n\n"
         "## Status: DESIGN PHASE\n"
         "Need to decide on connection pooling strategy.",
         ["rust", "websocket", "design"]),

        # ── Blog ──
        ("projects/blog/overview.md", "Technical Blog",
         "# Technical Blog\n\n"
         "Writing about game development and systems programming.\n"
         "Hosted on personal site. ~2,000 monthly readers.\n\n"
         "## Drafts\n"
         "- 'Procedural Generation with Wave Function Collapse'\n"
         "- 'Rust Error Handling Patterns I Actually Use'\n\n"
         "## Published Recently\n"
         "- 'ECS Architecture in Godot 4' (March 2026)",
         ["blog", "writing"]),

        # ── People ──
        ("people/overview.md", "People",
         "# Key People\n\n"
         "Collaborators, freelancers, and contacts.",
         ["people"]),

        ("people/mika.md", "Mika (Artist)",
         "# Mika — Freelance Artist\n\n"
         "Pixel art and animation for Starfall.\n"
         "Based in Tokyo. Works async (timezone difference).\n"
         "Communication: Discord, responds within 24h.\n"
         "Rate: $40/hr. Current contract through June.",
         ["freelancer", "art", "starfall"]),

        ("people/jordan.md", "Jordan (Composer)",
         "# Jordan — Music Composer\n\n"
         "Chiptune and ambient soundtrack for Starfall.\n"
         "Has delivered 8/12 tracks. Remaining 4 due by May.\n"
         "Communication: email, weekly check-ins on Fridays.",
         ["freelancer", "music", "starfall"]),

        # ── Personal ──
        ("personal/overview.md", "Personal Notes",
         "# Personal Notes\n\n"
         "Interests, learning goals, and personal projects.",
         ["personal"]),

        ("personal/learning.md", "Learning Goals",
         "# Current Learning\n\n"
         "- Rust async patterns (tokio deep dive)\n"
         "- Shader programming (GLSL for Godot)\n"
         "- Japanese (N3 level, studying for N2)\n\n"
         "## Resources\n"
         "- 'Rust for Rustaceans' by Jon Gjengset\n"
         "- 'The Book of Shaders' online\n"
         "- Wanikani + Bunpro for Japanese",
         ["learning", "personal"]),

        ("personal/setup.md", "Dev Setup",
         "# Development Setup\n\n"
         "- Main machine: Linux (Fedora), 32GB RAM, RTX 3080\n"
         "- Editor: Neovim + Godot\n"
         "- Terminal: Kitty + tmux\n"
         "- Dotfiles: github.com/example/dotfiles\n\n"
         "## Workflow\n"
         "Morning: deep work (coding). Afternoon: meetings, reviews.\n"
         "Friday: blog writing and open-source maintenance.",
         ["setup", "tools", "personal"]),
    ]

    for path, title, content, tags in documents:
        nav_memory.write(path, title, content, tags)

    print(f"Seeded {backend.document_count} documents into knowledge base.")


# ═══════════════════════════════════════════════════════════════════
# PART 5: Prompt Composer — assemble the system prompt each turn
# ═══════════════════════════════════════════════════════════════════

def build_prompt_composer() -> PromptComposer:
    composer = PromptComposer()

    # ── Module 0: Base instructions ──
    composer.add_module(
        name="instructions",
        position=0,
        content=(
            "You are Sage, a personal assistant with navigable knowledge and persistent memory.\n\n"
            "YOU HAVE TWO MEMORY SYSTEMS:\n"
            "1. **Core Memory** — key-value blocks you can read and edit (shown below).\n"
            "   Use core_memory_set/append/delete to manage it.\n"
            "   Store: user preferences, observations, action items, current focus.\n\n"
            "2. **Knowledge Space** — a navigable hierarchy of documents.\n"
            "   Use navigate to move to a document — its content loads into your context.\n"
            "   Use list_locations to discover what's available under a path.\n"
            "   Use search_knowledge to find documents by content.\n"
            "   Use write_document and append_to_document to create or update knowledge.\n"
            "   When you navigate away, the old location stays in your context briefly.\n\n"
            "IMPORTANT BEHAVIORS:\n"
            "- When the user asks about a topic, navigate to the relevant knowledge first.\n"
            "- When the user shares information, save it — either to core memory (for\n"
            "  preferences and quick notes) or to the knowledge base (for detailed records).\n"
            "- When you navigate somewhere, tell the user what you found there.\n"
            "- Use archive_search to find old conversations and departed location context.\n"
            "- Be proactive: suggest navigating to relevant knowledge when it would help."
        ),
    )

    # ── Module 5: Core memory (dynamic) ──
    composer.add_module(
        name="core_memory",
        position=5,
        content_fn=lambda: core_memory.build_context(),
        prefix=f"### Core Memory [last modified: {core_memory.last_modified}]",
        suffix="### End Core Memory",
    )

    # ── Module 10: Navigation context (dynamic — from NavigableMemory) ──
    composer.add_module(
        name="location_context",
        position=10,
        content_fn=nav_memory.build_context,
        prefix="### Knowledge Space — Current Location",
        suffix="### End Knowledge Space",
    )

    # ── Module 15: Recent location history (dynamic) ──
    composer.add_module(
        name="location_history",
        position=15,
        content_fn=nav_memory.build_history_context,
        prefix="### Recently Visited",
        suffix="### End Recently Visited",
    )

    # ── Module 20: Session metadata (dynamic) ──
    turn_counter = {"count": 0}

    def metadata_fn() -> str:
        turn_counter["count"] += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loc = nav_memory.current_title if nav_memory.current_path else "None"
        return (
            f"Time: {now}\n"
            f"Turn: {turn_counter['count']}\n"
            f"Location: {loc} ({nav_memory.current_path or 'none'})\n"
            f"Active messages: {message_manager.message_count}\n"
            f"Archived: {len(message_manager.archive)}\n"
            f"Knowledge documents: {backend.document_count}"
        )

    composer.add_module(
        name="metadata",
        position=20,
        content_fn=metadata_fn,
        prefix="### Session",
        suffix="### End Session",
    )

    return composer


# ═══════════════════════════════════════════════════════════════════
# PART 6: Message helpers
# ═══════════════════════════════════════════════════════════════════

def add_user_message(text: str, ttl: int = 10):
    msg = ChatMessage.create_user_message(text)
    message_manager.add_message(
        msg,
        lifecycle=MessageLifecycle(
            ttl=ttl,
            on_expire=ExpiryAction.ARCHIVE,
        ),
    )


def add_assistant_message(text: str, ttl: int = 10):
    msg = ChatMessage.create_assistant_message(text)
    message_manager.add_message(
        msg,
        lifecycle=MessageLifecycle(
            ttl=ttl,
            on_expire=ExpiryAction.ARCHIVE,
        ),
    )


def inject_ephemeral(text: str, ttl: int = 2):
    msg = ChatMessage.create_system_message(f"[Ephemeral] {text}")
    message_manager.add_message(
        msg,
        lifecycle=MessageLifecycle(ttl=ttl, on_expire=ExpiryAction.REMOVE),
    )


# ═══════════════════════════════════════════════════════════════════
# PART 7: Main Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Seed knowledge base ──
    seed_knowledge_base()
    print()

    # ── Seed core memory ──
    core_memory.set_block("persona",
        "I am Sage, a personal assistant with navigable knowledge and persistent memory.")
    core_memory.set_block("user_info", "No information about the user yet.")
    core_memory.set_block("current_focus", "No focus set. Ask the user what they're working on.")

    api = OpenAIChatAPI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="xiaomi/mimo-v2-pro",
        base_url="https://openrouter.ai/api/v1",
    )

    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = 0.35
    settings.top_p = 1.0

    # ── Build tool registry ──
    tool_registry = ToolRegistry()

    # Navigation tools (from NavigableMemory)
    nav_tools = [FunctionTool(t) for t in nav_memory.create_tools()]
    tool_registry.add_tools(nav_tools)

    # Core memory tools
    tool_registry.add_tools([
        FunctionTool(CoreMemorySet),
        FunctionTool(CoreMemoryAppend),
        FunctionTool(CoreMemoryDelete),
        FunctionTool(ArchiveSearch),
    ])

    # ── Build prompt composer ──
    composer = build_prompt_composer()

    # ── Pinned system message ──
    pinned = ChatMessage.create_system_message(
        "[SYSTEM] Always navigate to relevant knowledge before answering questions "
        "about projects, people, or topics. Save important user information to core memory."
    )
    message_manager.add_message(pinned, lifecycle=MessageLifecycle(pinned=True))

    # ── Opening context ──
    inject_ephemeral(
        "New session starting. Greet the user and ask what they'd like to work on. "
        "Suggest navigating to their projects if they want a status update.",
        ttl=2,
    )

    # ── Start at projects overview ──
    nav_memory.navigate("projects/overview.md")

    # ── Message TTL ──
    USER_TTL = 10
    ASSISTANT_TTL = 10

    print("=" * 64)
    print("  🧭 Sage — Personal Agent with Navigable Knowledge")
    print("  ToolAgents + NavigableMemory + CoreMemory + SmartMessages")
    print(f"  Knowledge: {backend.document_count} documents seeded")
    print("=" * 64)
    print()
    print("Commands:")
    print("  quit           — Exit")
    print("  /memory        — Show core memory blocks")
    print("  /location      — Show current knowledge location")
    print("  /archive       — Show archived messages")
    print("  /status        — Show system status")
    print("  /tree          — Show knowledge space tree")
    print("  /inject <msg>  — Inject ephemeral context")
    print("  /save          — Save state to JSON")
    print()

    while True:
        try:
            user_input = input("\n🧑 You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        # ── Meta-commands ──
        if user_input.lower() == "quit":
            print("Session ended.")
            break

        elif user_input.lower() == "/memory":
            print(f"\n📝 Core Memory:")
            print(core_memory.build_context())
            continue

        elif user_input.lower() == "/location":
            if nav_memory.current_path:
                print(f"\n📍 {nav_memory.current_title}")
                print(f"   Path: {nav_memory.current_path}")
                history = nav_memory.history[-5:]
                if history:
                    print(f"   History: {' → '.join(history)}")
            else:
                print("\n📍 No location loaded.")
            continue

        elif user_input.lower() == "/archive":
            archive = message_manager.archive
            print(f"\n📦 Archive ({len(archive)} items):")
            for i, msg in enumerate(archive[-10:]):
                print(f"  [{i}] {msg.get_as_text()[:100]}")
            if not archive:
                print("  (empty)")
            continue

        elif user_input.lower() == "/status":
            print(f"\n📊 Status:")
            print(f"  Location: {nav_memory.current_title} ({nav_memory.current_path})")
            print(f"  Active messages: {message_manager.message_count}")
            print(f"  Archived: {len(message_manager.archive)}")
            print(f"  Core memory blocks: {len(core_memory.blocks)}")
            print(f"  Knowledge documents: {backend.document_count}")
            print(f"  Locations visited: {len(nav_memory.history)}")
            continue

        elif user_input.lower() == "/tree":
            print(f"\n🌳 Knowledge Space:")
            docs = sorted(nav_memory.list_at(""), key=lambda d: d.path)
            current_dir = ""
            for d in docs:
                parts = d.path.rsplit("/", 1)
                dir_part = parts[0] + "/" if len(parts) > 1 else ""
                if dir_part != current_dir:
                    current_dir = dir_part
                    print(f"  {current_dir}")
                marker = " ← YOU ARE HERE" if d.path == nav_memory.current_path else ""
                print(f"    {d.path.split('/')[-1]:30s} — {d.title}{marker}")
            continue

        elif user_input.lower().startswith("/inject "):
            inject_ephemeral(user_input[8:].strip(), ttl=2)
            print("  💉 Ephemeral context injected (expires in 2 turns)")
            continue

        elif user_input.lower() == "/save":
            state = {
                "core_memory": core_memory.blocks,
                "current_location": nav_memory.current_path,
                "location_history": nav_memory.history,
                "archive": [m.get_as_text() for m in message_manager.archive],
            }
            with open("sage_state.json", "w") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            print("  💾 Saved to sage_state.json")
            continue

        # ── Tick message lifecycles ──
        tick_result = message_manager.tick()
        if tick_result.removed:
            for m in tick_result.removed:
                print("  🗑️  [Expired] ephemeral context removed")
        if tick_result.archived:
            for m in tick_result.archived:
                print(f"  📦 [Archived] \"{m.get_as_text()[:60]}...\"")

        # ── Add user message ──
        add_user_message(user_input, ttl=USER_TTL)

        # ── Update dynamic prefix ──
        composer.update_module(
            "core_memory",
            prefix=f"### Core Memory [last modified: {core_memory.last_modified}]",
        )

        # ── Compile system prompt ──
        system_prompt = composer.compile()

        # ── Build messages ──
        messages = [
            ChatMessage.create_system_message(system_prompt),
            *message_manager.get_active_messages(),
        ]

        # ── Call the agent ──
        try:
            chat_response = agent.get_response(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
            )

            response_text = chat_response.response.strip()
            print(f"\n🧭 Sage > {response_text}")

            add_assistant_message(response_text, ttl=ASSISTANT_TTL)

            # Add tool call messages with TTL
            for msg in chat_response.messages:
                role = msg.get_role()
                if role not in ("user", "assistant"):
                    message_manager.add_message(
                        msg,
                        lifecycle=MessageLifecycle(
                            ttl=ASSISTANT_TTL,
                            on_expire=ExpiryAction.ARCHIVE,
                        ),
                    )

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("  (Try rephrasing your request.)")


if __name__ == "__main__":
    main()
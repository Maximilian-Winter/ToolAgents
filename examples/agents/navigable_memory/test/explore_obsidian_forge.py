#!/usr/bin/env python3
# explore_obsidian_forge.py — Interactive demo with the Obsidian Forge knowledge base
#
# A studio manager assistant ("Forge") that can navigate the deep knowledge
# hierarchy, track context with smart messages, and persist state across sessions.
#
# Usage:
#   pip install ToolAgents pydantic python-dotenv
#   Set OPENROUTER_API_KEY in .env (or modify the provider below)
#   python explore_obsidian_forge.py
#
# Try these interactions:
#   "What are our critical bugs?"
#   "Navigate to the VFX backlog"
#   "What's the risk if Kai gets sick?"
#   "Compare Ashenmoor and Drift Protocol status"
#   "Show me the boss design for Act 3"
#   "What did we decide in the leadership meeting?"
#   "Save a note: remind Marcus about vacation planning"
#   "What's our financial runway?"

import json
import os
import sys
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

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
)

from seed_obsidian_forge import seed as seed_knowledge_base

load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# Core Memory
# ═══════════════════════════════════════════════════════════════════

class CoreMemory:
    def __init__(self, block_limit: int = 600):
        self.blocks: dict[str, str] = {}
        self.block_limit = block_limit
        self.last_modified = "never"

    def set_block(self, name: str, content: str) -> str:
        if len(content) > self.block_limit:
            return f"Error: exceeds {self.block_limit} char limit."
        self.blocks[name] = content
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Core memory '{name}' updated."

    def append_block(self, name: str, content: str) -> str:
        current = self.blocks.get(name, "")
        new = current + content
        if len(new) > self.block_limit:
            return f"Error: would exceed {self.block_limit} chars."
        self.blocks[name] = new
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Appended to '{name}'."

    def delete_block(self, name: str) -> str:
        if name in self.blocks:
            del self.blocks[name]
            self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"Block '{name}' deleted."
        return f"Block '{name}' not found."

    def build_context(self) -> str:
        if not self.blocks:
            return "(no memory blocks stored)"
        lines = []
        for k, v in self.blocks.items():
            lines.append(f"<{k}> ({len(v)}/{self.block_limit} chars)\n{v}\n</{k}>")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"blocks": dict(self.blocks), "last_modified": self.last_modified}

    def from_dict(self, data: dict):
        self.blocks = data.get("blocks", {})
        self.last_modified = data.get("last_modified", "restored")


# ═══════════════════════════════════════════════════════════════════
# Instances
# ═══════════════════════════════════════════════════════════════════

core_memory = CoreMemory()
message_manager = SmartMessageManager()
backend = InMemoryBackend()


def on_location_depart(record: DepartureRecord):
    snippet = record.content[:200].replace("\n", " ")
    msg = ChatMessage.create_system_message(
        f"[Previously at] {record.title} ({record.path})\n{snippet}..."
    )
    message_manager.add_message(
        msg,
        lifecycle=MessageLifecycle(ttl=8, on_expire=ExpiryAction.ARCHIVE),
    )
    print(f"  📍 Departed: {record.title}")


nav_memory = NavigableMemory(
    backend=backend,
    on_depart=on_location_depart,
    context_window=3,
    include_siblings=True,
    include_parent=True,
)


# ═══════════════════════════════════════════════════════════════════
# Memory Tools
# ═══════════════════════════════════════════════════════════════════

class CoreMemorySet(BaseModel):
    """Set or overwrite a core memory block. Use for user preferences, observations, action items."""
    block_name: str = Field(..., description="Block name (e.g. 'user_info', 'priorities').")
    content: str = Field(..., description="Content to store (max 600 chars).")
    def run(self) -> str:
        return core_memory.set_block(self.block_name, self.content)

class CoreMemoryAppend(BaseModel):
    """Append text to an existing core memory block."""
    block_name: str = Field(..., description="Block name.")
    content: str = Field(..., description="Text to append.")
    def run(self) -> str:
        return core_memory.append_block(self.block_name, self.content)

class CoreMemoryDelete(BaseModel):
    """Delete a core memory block."""
    block_name: str = Field(..., description="Block name.")
    def run(self) -> str:
        return core_memory.delete_block(self.block_name)

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
# Prompt Composer
# ═══════════════════════════════════════════════════════════════════

def build_prompt_composer() -> PromptComposer:
    composer = PromptComposer()

    composer.add_module("instructions", position=0, content=(
        "You are Forge, a studio manager assistant for Obsidian Forge Studios.\n\n"
        "YOU HAVE TWO MEMORY SYSTEMS:\n"
        "1. **Core Memory** — key-value blocks you can read/edit (shown below).\n"
        "   Use core_memory_set/append/delete. Store: priorities, observations, notes.\n\n"
        "2. **Knowledge Space** — navigable document hierarchy with studio information.\n"
        "   Use navigate_to_document to move — content loads into your context.\n"
        "   Use list_locations to discover documents under a path.\n"
        "   Use search_knowledge to find by content.\n"
        "   Use read_document to peek without navigating.\n"
        "   When you navigate away, old location lingers briefly then archives.\n\n"
        "BEHAVIORS:\n"
        "- Navigate to relevant knowledge BEFORE answering project questions.\n"
        "- When navigating deep, tell the user what you found.\n"
        "- Save important observations and action items to core memory.\n"
        "- Use archive_search to recall old conversations.\n"
        "- Be proactive: suggest related documents the user might want to check.\n"
        "- When you see risks or blockers, highlight them clearly.\n"
        "- Cross-reference: if a person is mentioned, navigate to their profile too."
    ))

    composer.add_module("core_memory", position=5,
                        content_fn=lambda: core_memory.build_context(),
                        prefix=f"### Core Memory [modified: {core_memory.last_modified}]",
                        suffix="### End Core Memory")

    composer.add_module("location", position=10,
                        content_fn=nav_memory.build_context,
                        prefix="### Knowledge Space — Current Location",
                        suffix="### End Knowledge Space")

    composer.add_module("history", position=15,
                        content_fn=nav_memory.build_history_context,
                        prefix="### Recently Visited",
                        suffix="### End Recently Visited")

    turn_counter = {"n": 0}
    def metadata_fn():
        turn_counter["n"] += 1
        loc = nav_memory.current_title if nav_memory.current_path else "None"
        return (
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Turn: {turn_counter['n']}\n"
            f"Location: {loc}\n"
            f"Active msgs: {message_manager.message_count}\n"
            f"Archived: {len(message_manager.archive)}\n"
            f"Documents: {backend.document_count}"
        )

    composer.add_module("metadata", position=20,
                        content_fn=metadata_fn,
                        prefix="### Session", suffix="### End Session")

    return composer


# ═══════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════

SAVE_FILE = "forge_state.json"


def save_state():
    state = {
        "core_memory": core_memory.to_dict(),
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
        "tick_count": message_manager.tick_count,
        "saved_at": datetime.now().isoformat(),
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return state


def load_state() -> bool:
    if not os.path.exists(SAVE_FILE):
        return False
    try:
        with open(SAVE_FILE, "r") as f:
            state = json.load(f)

        # Restore core memory
        core_memory.from_dict(state["core_memory"])

        # Restore messages
        message_manager.clear()
        for md in state["active_messages"]:
            if md["role"] == "user":
                msg = ChatMessage.create_user_message(md["text"])
            elif md["role"] == "assistant":
                msg = ChatMessage.create_assistant_message(md["text"])
            else:
                msg = ChatMessage.create_system_message(md["text"])

            lifecycle = MessageLifecycle(
                ttl=md["ttl"],
                turns_alive=md["turns_alive"],
                pinned=md["pinned"],
                on_expire=ExpiryAction(md["on_expire"]),
            )
            message_manager.add_message(msg, lifecycle)

        # Restore navigation
        if state.get("current_location"):
            nav_memory.navigate(state["current_location"])

        print(f"  Restored from {state.get('saved_at', 'unknown time')}")
        print(f"  Core memory: {len(core_memory.blocks)} blocks")
        print(f"  Messages: {message_manager.message_count} active, {len(state.get('archive', []))} archived")
        print(f"  Location: {nav_memory.current_title or 'none'}")
        return True
    except Exception as e:
        print(f"  Failed to load state: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# Message helpers
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
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    # Seed knowledge base (always — InMemoryBackend doesn't persist)
    seed_knowledge_base(nav_memory)
    print()

    # Try loading previous session
    restored = load_state()
    if not restored:
        # Fresh session defaults
        core_memory.set_block("persona",
            "I am Forge, studio manager assistant for Obsidian Forge Studios.")
        core_memory.set_block("priorities",
            "Key dates: Ashenmoor EA Aug 2026, Drift Protocol E3 Jun 2026.")
        core_memory.set_block("user_info", "No user information yet.")

        message_manager.add_message(
            ChatMessage.create_system_message(
                "[SYSTEM] Navigate to relevant knowledge before answering project questions. "
                "Cross-reference people profiles when names come up. "
                "Highlight risks and blockers proactively."
            ),
            MessageLifecycle(pinned=True),
        )

        inject_ephemeral(
            "New session. Greet the user and ask what they need. "
            "Suggest a studio status overview or specific project check.",
            ttl=2,
        )

        nav_memory.navigate("studio/overview.md")

    # Provider setup
    api = OpenAIChatAPI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="xiaomi/mimo-v2-pro",
        base_url="https://openrouter.ai/api/v1",
    )

    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = 0.35
    settings.top_p = 1.0

    # Tools
    tool_registry = ToolRegistry()
    nav_tools = [FunctionTool(t) for t in nav_memory.create_tools()]
    tool_registry.add_tools(nav_tools)
    tool_registry.add_tools([
        FunctionTool(CoreMemorySet),
        FunctionTool(CoreMemoryAppend),
        FunctionTool(CoreMemoryDelete),
        FunctionTool(ArchiveSearch),
    ])

    # Composer
    composer = build_prompt_composer()

    # Banner
    print("=" * 64)
    print("  🔨 Forge — Obsidian Forge Studios Manager Assistant")
    print(f"  Knowledge: {backend.document_count} documents")
    print(f"  Session: {'restored' if restored else 'new'}")
    print("=" * 64)
    print()
    print("Commands:")
    print("  quit            — Exit (auto-saves)")
    print("  /memory         — Show core memory")
    print("  /location       — Current knowledge location")
    print("  /archive        — Show archived messages")
    print("  /status         — System status")
    print("  /tree           — Knowledge space tree")
    print("  /inject <msg>   — Inject ephemeral context")
    print("  /save           — Save state")
    print("  /clear          — Clear messages and start fresh")
    print()
    print("Try: 'What are our critical bugs?'")
    print("     'Show me the VFX backlog and who owns it'")
    print("     'What's our financial situation?'")
    print()

    while True:
        try:
            user_input = input("\n🧑 You > ").strip()
        except (KeyboardInterrupt, EOFError):
            save_state()
            print("\n  💾 State saved. Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            save_state()
            print("  💾 State saved. Session ended.")
            break

        elif user_input.lower() == "/memory":
            print(f"\n📝 Core Memory (modified: {core_memory.last_modified}):")
            print(core_memory.build_context())
            continue

        elif user_input.lower() == "/location":
            if nav_memory.current_path:
                print(f"\n📍 {nav_memory.current_title}")
                print(f"   Path: {nav_memory.current_path}")
                if nav_memory.history:
                    recent = nav_memory.history[-5:]
                    print(f"   Recent: {' → '.join(recent)}")
            else:
                print("\n📍 Not at any location.")
            continue

        elif user_input.lower() == "/archive":
            archive = message_manager.archive
            print(f"\n📦 Archive ({len(archive)} items):")
            for i, msg in enumerate(archive[-10:]):
                text = msg.get_as_text()[:100].replace("\n", " ")
                print(f"  [{i}] {text}")
            if not archive:
                print("  (empty)")
            continue

        elif user_input.lower() == "/status":
            print(f"\n📊 Status:")
            print(f"  Location: {nav_memory.current_title} ({nav_memory.current_path})")
            print(f"  Active messages: {message_manager.message_count}")
            sms = message_manager.get_smart_messages()
            permanent = sum(1 for sm in sms if sm.lifecycle.is_permanent)
            ephemeral = sum(1 for sm in sms if not sm.lifecycle.is_permanent and not sm.lifecycle.pinned)
            pinned = sum(1 for sm in sms if sm.lifecycle.pinned)
            print(f"    Permanent: {permanent}, Ephemeral: {ephemeral}, Pinned: {pinned}")
            print(f"  Archived: {len(message_manager.archive)}")
            print(f"  Core memory: {len(core_memory.blocks)} blocks")
            print(f"  Documents: {backend.document_count}")
            print(f"  Ticks: {message_manager.tick_count}")
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
                    print(f"\n  📁 {current_dir}")
                marker = " ◀ HERE" if d.path == nav_memory.current_path else ""
                name = d.path.split("/")[-1]
                print(f"      {name:40s} {d.title}{marker}")
            continue

        elif user_input.lower().startswith("/inject "):
            inject_ephemeral(user_input[8:].strip(), ttl=3)
            print("  💉 Injected (expires in 3 turns)")
            continue

        elif user_input.lower() == "/save":
            save_state()
            print("  💾 State saved to forge_state.json")
            continue

        elif user_input.lower() == "/clear":
            message_manager.clear()
            message_manager.add_message(
                ChatMessage.create_system_message(
                    "[SYSTEM] Navigate to relevant knowledge before answering questions."
                ),
                MessageLifecycle(pinned=True),
            )
            print("  🧹 Messages cleared, pinned system message restored.")
            continue

        # ── Process turn ──
        tick_result = message_manager.tick()
        if tick_result.removed:
            print(f"  🗑️  {len(tick_result.removed)} ephemeral message(s) expired")
        if tick_result.archived:
            print(f"  📦 {len(tick_result.archived)} message(s) archived")

        add_user_msg(user_input)

        composer.update_module(
            "core_memory",
            prefix=f"### Core Memory [modified: {core_memory.last_modified}]",
        )

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
            print(f"\n🔨 Forge > {response_text}")

            add_assistant_msg(response_text)

            for msg in chat_response.messages:
                if msg.role.value not in ("user", "assistant"):
                    message_manager.add_message(
                        msg,
                        lifecycle=MessageLifecycle(
                            ttl=ASSISTANT_TTL,
                            on_expire=ExpiryAction.ARCHIVE,
                        ),
                    )

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

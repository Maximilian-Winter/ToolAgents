#!/usr/bin/env python3
"""
ada_assistant.py — Personal MemGPT-Style Assistant (AgentHarness)
=================================================================

A personal AI companion that remembers everything about you,
organizes knowledge into navigable documents, and grows more
helpful over time.

Uses AgentHarness with:
  - NavigableMemory (InMemoryBackend or SQLiteBackend for persistence)
  - CoreMemory for quick-access personal facts
  - PromptComposer with dynamic modules for memory and navigation
  - SmartMessageManager for conversation windowing
  - Custom IOHandler with meta-commands

Usage:
  pip install ToolAgents pydantic python-dotenv pyyaml
  Set your API key in .env (or modify the provider config below)
  python ada_assistant.py

  With persistent storage (survives restarts):
  python ada_assistant.py --persist

Try these interactions:
  "Hey, I'm Alex. I'm a software engineer working on a Rust project."
  "I love spicy food, especially Thai and Sichuan cuisine."
  "Remind me: my dentist appointment is next Thursday at 2pm."
  "What do you know about me so far?"
  "I'm starting a new side project — a CLI tool for managing dotfiles."
  "What projects am I working on?"
"""

import os
import sys
from datetime import datetime
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ToolAgents import FunctionTool
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.agent_harness import create_harness
from ToolAgents.agent_harness.smart_messages import MessageLifecycle, ExpiryAction
from ToolAgents.agent_memory.navigable_memory import (
    NavigableMemory,
    InMemoryBackend,
    DepartureRecord,
)
from ToolAgents.agent_tools.coding_tools import CodingTools
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.message_template import MessageTemplate

load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# CoreMemory — self-editable persistent notes
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
            lines.append(
                f"<{key}> ({len(value)}/{self.block_limit} chars)\n{value}\n</{key}>"
            )
        return "\n".join(lines)


# Module-level instances (referenced by tool classes)
core_memory = CoreMemory(block_limit=500)
harness_ref: list = [None]  # Filled after harness creation


# ═══════════════════════════════════════════════════════════════════
# Memory Tools — Pydantic models for core memory + archive search
# ═══════════════════════════════════════════════════════════════════

class CoreMemorySet(BaseModel):
    """Set or overwrite a named block in core memory.
    Use to store user preferences, observations, or project state."""

    block_name: str = Field(
        ..., description="Block name (e.g., 'user_info', 'preferences')."
    )
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
        if harness_ref[0] is None:
            return "Archive not available."
        results = []
        for msg in harness_ref[0].smart_messages.archive:
            text = msg.get_as_text()
            if self.query.lower() in text.lower():
                results.append(text[:200])
        if results:
            return f"Found {len(results)} archived item(s):\n" + "\n---\n".join(
                results
            )
        return f"No archived items matching '{self.query}'."


# ═══════════════════════════════════════════════════════════════════
# Knowledge Seeding
# ═══════════════════════════════════════════════════════════════════


def seed_knowledge(nav: NavigableMemory, starter_file: str = "ada_starter.yaml"):
    """Seed the knowledge space from the YAML starter."""
    with open(starter_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    seeds = data.get("knowledge_seeds", {})
    count = 0

    for category, content in seeds.items():
        if isinstance(content, dict):
            for doc_name, doc_content in content.items():
                path = f"{category}/{doc_name}.md"
                title = f"{category.title()} — {doc_name.replace('_', ' ').title()}"
                if isinstance(doc_content, str):
                    body = doc_content.strip()
                elif isinstance(doc_content, dict):
                    parts = []
                    for k, v in doc_content.items():
                        if isinstance(v, list):
                            items = "\n".join(f"- {i}" for i in v)
                            parts.append(f"## {k}\n{items}")
                        else:
                            parts.append(f"## {k}\n{v}")
                    body = "\n\n".join(parts)
                else:
                    body = str(doc_content)

                nav.write(path, title, f"# {title}\n\n{body}")
                count += 1
                print(f"  + {path}")
        elif isinstance(content, str):
            path = f"{category}.md"
            nav.write(
                path, category.title(), f"# {category.title()}\n\n{content.strip()}"
            )
            count += 1
            print(f"  + {path}")

    reminders = data.get("initial_reminders", [])
    if reminders:
        reminder_text = "\n".join(f"- {r}" for r in reminders)
        nav.write(
            "notes/initial-reminders.md",
            "Initial Reminders",
            f"# Initial Reminders\n\n{reminder_text}",
        )
        count += 1
        print(f"  + notes/initial-reminders.md")

    print(f"  Seeded {count} documents.\n")
    return count


# ═══════════════════════════════════════════════════════════════════
# Custom IOHandler — meta-commands + formatted output
# ═══════════════════════════════════════════════════════════════════


class AdaIOHandler:
    """IOHandler with meta-commands for inspecting memory and navigation."""

    def __init__(self, core_mem: CoreMemory, nav_mem: NavigableMemory, backend):
        self.core_memory = core_mem
        self.nav_memory = nav_mem
        self.backend = backend

    def get_input(self, prompt: str = "> ") -> Optional[str]:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            return None

        if not user_input:
            return self.get_input(prompt)

        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            return None

        # ── Meta-commands ──
        if user_input.lower() == "/memory":
            print(f"\nCore Memory:")
            print(self.core_memory.build_context())
            return self.get_input(prompt)

        if user_input.lower() == "/location":
            if self.nav_memory.current_path:
                print(f"\n  {self.nav_memory.current_title}")
                print(f"   Path: {self.nav_memory.current_path}")
                history = self.nav_memory.history[-5:]
                if history:
                    print(f"   History: {' -> '.join(history)}")
            else:
                print("\n  No location loaded.")
            return self.get_input(prompt)

        if user_input.lower() == "/tree":
            print(f"\n  Knowledge Space:")
            docs = sorted(self.nav_memory.list_at(""), key=lambda d: d.path)
            current_dir = ""
            for d in docs:
                parts = d.path.rsplit("/", 1)
                dir_part = parts[0] + "/" if len(parts) > 1 else ""
                if dir_part != current_dir:
                    current_dir = dir_part
                    print(f"  {current_dir}")
                marker = (
                    " <-- HERE"
                    if d.path == self.nav_memory.current_path
                    else ""
                )
                print(f"    {d.path.split('/')[-1]:30s} -- {d.title}{marker}")
            return self.get_input(prompt)

        if user_input.lower() == "/status":
            print(f"\n  Status:")
            loc = self.nav_memory.current_title or "None"
            print(f"  Location: {loc} ({self.nav_memory.current_path or 'none'})")
            print(f"  Core memory blocks: {len(self.core_memory.blocks)}")
            print(f"  Knowledge documents: {self.backend.document_count}")
            print(f"  Locations visited: {len(self.nav_memory.history)}")
            return self.get_input(prompt)

        if user_input.lower() == "/help":
            print("\nCommands:")
            print("  quit / exit  -- End session")
            print("  /memory      -- Show core memory blocks")
            print("  /location    -- Show current knowledge location")
            print("  /tree        -- Show knowledge space tree")
            print("  /status      -- Show system status")
            print("  /help        -- Show this help")
            return self.get_input(prompt)

        return user_input

    def on_text(self, text: str) -> None:
        print(f"\nAda > {text}")

    def on_chunk(self, chunk) -> None:
        if chunk.chunk:
            print(chunk.chunk, end="", flush=True)
        if chunk.finished:
            print()

    def on_error(self, error: Exception) -> None:
        print(f"\nError: {error}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    # ── Parse arguments ──
    use_persist = "--persist" in sys.argv
    working_directory = "./"

    # ── Determine paths ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    starter_file = os.path.join(script_dir, "ada_starter.yaml")
    prompt_file = os.path.join(script_dir, "system_prompt.txt")

    # ── Backend ──
    if use_persist:
        try:
            from ToolAgents.agent_memory.navigable_memory import SQLiteBackend

            db_path = os.path.join(script_dir, "ada_knowledge.db")
            backend = SQLiteBackend(db_path)
            print(f"  Using SQLite backend: {db_path}")
            needs_seed = backend.count() == 0
        except ImportError:
            print("  SQLiteBackend not available, falling back to InMemoryBackend.")
            backend = InMemoryBackend()
            needs_seed = True
    else:
        backend = InMemoryBackend()
        needs_seed = True

    # ── Departure callback (closure over harness_ref) ──
    def on_location_depart(record: DepartureRecord):
        if harness_ref[0] is not None:
            snippet = record.content[:200].replace("\n", " ")
            msg = ChatMessage.create_system_message(
                f"[Previous Location] {record.title} ({record.path})\n{snippet}..."
            )
            harness_ref[0].add_smart_message(
                msg,
                lifecycle=MessageLifecycle(
                    ttl=12,
                    on_expire=ExpiryAction.ARCHIVE,
                ),
            )

    # ── NavigableMemory ──
    nav = NavigableMemory(
        backend=backend,
        on_depart=on_location_depart,
        context_window=3,
        include_siblings=True,
        include_parent=True,
    )

    # ── Seed knowledge if needed ──
    if needs_seed:
        print("  Seeding knowledge space...")
        seed_knowledge(nav, starter_file)

    # ── Seed core memory ──
    core_memory.set_block(
        "persona",
        "I am Ada, a personal AI companion with persistent memory. "
        "I remember what users tell me and organize knowledge into "
        "navigable documents that grow over time.",
    )
    core_memory.set_block("user_info", "No information about the user yet.")
    core_memory.set_block("active_reminders", "")

    # ── System prompt from template ──
    system_template = MessageTemplate.from_file(prompt_file)
    system_prompt = system_template.generate_message_content(
        additional_instructions=""
    )

    # ── Provider ──
    api = OpenAIChatAPI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="xiaomi/mimo-v2-pro",
        base_url="https://openrouter.ai/api/v1",
    )
    settings = api.get_default_settings()
    settings.temperature = 0.45
    settings.top_p = 1.0

    # ── Tools ──
    coding_tools = CodingTools(working_directory=working_directory)
    nav_tools = [FunctionTool(t) for t in nav.create_tools()]
    memory_tools = [
        FunctionTool(CoreMemorySet),
        FunctionTool(CoreMemoryAppend),
        FunctionTool(CoreMemoryDelete),
        FunctionTool(ArchiveSearch),
    ]
    all_tools = nav_tools + memory_tools + coding_tools.get_tools()

    # ── Create harness ──
    harness = create_harness(
        provider=api,
        system_prompt=system_prompt,
        settings=settings,
        max_context_tokens=128000,
        tools=all_tools,
        streaming=True,
    )
    harness_ref[0] = harness

    # ── Dynamic prompt modules ──
    harness.prompt_composer.add_module(
        name="core_memory",
        position=5,
        content_fn=lambda: core_memory.build_context(),
        prefix=f"### Core Memory [last modified: {core_memory.last_modified}]",
        suffix="### End Core Memory",
    )

    harness.prompt_composer.add_module(
        name="location_context",
        position=10,
        content_fn=nav.build_context,
        prefix="### Knowledge Space -- Current Location",
        suffix="### End Knowledge Space",
    )

    harness.prompt_composer.add_module(
        name="location_history",
        position=15,
        content_fn=nav.build_history_context,
        prefix="### Recently Visited",
        suffix="### End Recently Visited",
    )

    # ── Pinned system instruction ──
    harness.add_pinned_message(
        ChatMessage.create_system_message(
            "[SYSTEM] When the user shares personal details, save them to core memory "
            "immediately using core_memory_set or core_memory_append. "
            "When they mention projects or topics worth tracking, create knowledge documents. "
            "Be warm and attentive. Reference past context naturally."
        )
    )

    # ── Ephemeral greeting ──
    harness.add_ephemeral_message(
        ChatMessage.create_system_message(
            "New user session. Greet them warmly, introduce yourself briefly, "
            "and ask their name. Be natural -- don't list your capabilities."
        ),
        ttl=2,
    )

    # ── Navigate to starting location ──
    nav.navigate("user/overview.md")

    # ── Banner ──
    print("=" * 64)
    print("  Ada -- Personal AI Companion with Persistent Memory")
    print("  AgentHarness + NavigableMemory + CoreMemory")
    print(f"  Knowledge: {backend.document_count} documents seeded")
    print("=" * 64)
    print()
    print("  Type /help for commands, quit to exit.")
    print()

    # ── Run ──
    io_handler = AdaIOHandler(core_memory, nav, backend)
    harness.run(io_handler=io_handler, streaming=True)


if __name__ == "__main__":
    main()

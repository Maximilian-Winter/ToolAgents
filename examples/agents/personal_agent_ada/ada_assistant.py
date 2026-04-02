#!/usr/bin/env python3
"""
ada_assistant.py — Personal MemGPT-Style Assistant
===================================================

A personal AI companion that remembers everything about you,
organizes knowledge into navigable documents, and grows more
helpful over time.

Uses VirtualGameMaster2 as the harness with:
  - NavigableMemory (InMemoryBackend or SQLiteBackend for persistence)
  - CoreMemory for quick-access personal facts
  - PromptComposer with a custom system prompt template
  - SmartMessageManager for conversation windowing
  - Session save/load for continuity across restarts

Usage:
  pip install ToolAgents pydantic python-dotenv pyyaml
  Set your API key in .env (or modify the provider config below)
  python ada_assistant.py

  With persistent storage (survives restarts):
  python ada_assistant.py --persist

  Resume a saved session:
  python ada_assistant.py --persist --resume

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

import yaml
from dotenv import load_dotenv

from ToolAgents.agent_memory.navigable_memory import NavigableMemory, InMemoryBackend
from ToolAgents.agent_tools.coding_tools import CodingTools
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.message_template import MessageTemplate

# Import harness
from virtual_game_master_v2 import (
    VirtualGameMaster2,
    HarnessConfig,
    run_cli,
)

load_dotenv()


def seed_knowledge(nav: NavigableMemory, starter_file: str = "ada_starter.yaml"):
    """Seed the knowledge space from the YAML starter."""
    with open(starter_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    seeds = data.get("knowledge_seeds", {})
    count = 0

    for category, content in seeds.items():
        if isinstance(content, dict):
            # Each key under the category becomes a document
            for doc_name, doc_content in content.items():
                path = f"{category}/{doc_name}.md"
                title = f"{category.title()} — {doc_name.replace('_', ' ').title()}"
                if isinstance(doc_content, str):
                    body = doc_content.strip()
                elif isinstance(doc_content, dict):
                    # Render dict as markdown
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
                print(f"  ✓ {path}")
        elif isinstance(content, str):
            path = f"{category}.md"
            nav.write(path, category.title(), f"# {category.title()}\n\n{content.strip()}")
            count += 1
            print(f"  ✓ {path}")

    # Seed reminders into a notes document
    reminders = data.get("initial_reminders", [])
    if reminders:
        reminder_text = "\n".join(f"- {r}" for r in reminders)
        nav.write(
            "notes/initial-reminders.md",
            "Initial Reminders",
            f"# Initial Reminders\n\n{reminder_text}",
        )
        count += 1
        print(f"  ✓ notes/initial-reminders.md")

    print(f"  Seeded {count} documents.\n")
    return count


def main():
    # ── Parse arguments ──
    use_persist = "--persist" in sys.argv
    do_resume = "--resume" in sys.argv
    working_directory = "./"
    # ── Determine paths ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    starter_file = os.path.join(script_dir, "ada_starter.yaml")
    prompt_file = os.path.join(script_dir, "system_prompt.txt")
    save_folder = os.path.join(script_dir, "saves")

    # ── Backend ──
    if use_persist:
        try:
            from ToolAgents.agent_memory.sqlite_backend import SQLiteBackend
            db_path = os.path.join(script_dir, "ada_knowledge.db")
            backend = SQLiteBackend(db_path)
            print(f"  📀 Using SQLite backend: {db_path}")
            needs_seed = backend.count() == 0
        except ImportError:
            print("  SQLiteBackend not available, falling back to InMemoryBackend.")
            backend = InMemoryBackend()
            needs_seed = True
    else:
        backend = InMemoryBackend()
        needs_seed = True

    # ── NavigableMemory ──
    nav = NavigableMemory(
        backend=backend,
        context_window=3,
        include_siblings=True,
        include_parent=True,
    )

    # ── Seed knowledge if needed ──
    if needs_seed:
        print("  Seeding knowledge space...")
        seed_knowledge(nav, starter_file)

    # ── Config ──
    config = HarnessConfig()

    # Provider — edit these or use .env
    #config.API_TYPE = os.getenv("API_TYPE", "groq")
    #config.API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
    #.API_URL = os.getenv("API_URL", "")
    #config.MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")

    # For OpenRouter, uncomment:
    config.API_TYPE = "openrouter"
    config.API_KEY = os.getenv("OPENROUTER_API_KEY")
    config.API_URL = "https://openrouter.ai/api/v1"
    config.MODEL = "xiaomi/mimo-v2-pro"

    config.TEMPERATURE = 0.45
    config.TOP_P = 1.0
    config.USER_MSG_TTL = 16
    config.ASSISTANT_MSG_TTL = 16
    config.LOCATION_MSG_TTL = 12
    config.SAVE_FOLDER = save_folder

    # ── System prompt template ──
    system_template = MessageTemplate.from_file(prompt_file)
    coding_tools = CodingTools(working_directory=working_directory)

    # Get the base tools (bash, read, write, edit, glob, grep, list_directory, diff_files)
    extra_tools = coding_tools.get_tools()

    # ── Build harness ──
    harness = VirtualGameMaster2(
        config=config,
        nav_memory=nav,
        system_template=system_template,
        extra_tools=extra_tools,
        initial_core_memory={
            "persona": (
                "I am Ada, a personal AI companion with persistent memory. "
                "I remember what users tell me and organize knowledge into "
                "navigable documents that grow over time."
            ),
            "user_info": "No information about the user yet.",
            "active_reminders": "",
        },
        pinned_messages=[
            "[SYSTEM] When the user shares personal details, save them to core memory "
            "immediately using core_memory_set or core_memory_append. "
            "When they mention projects or topics worth tracking, create knowledge documents. "
            "Be warm and attentive. Reference past context naturally."
        ],
        summarize_on_depart=False,
        debug_mode="--debug" in sys.argv,
    )

    # ── Resume previous session if requested ──
    if do_resume:
        if harness.load():
            print("  📂 Previous session restored.")
        else:
            print("  No saved session found — starting fresh.")
            nav.navigate("user/overview.md")
            harness.inject_ephemeral(
                "New user session. Greet them warmly, introduce yourself briefly, "
                "and ask their name. Be natural — don't list your capabilities.",
                ttl=2,
            )
    else:
        # Fresh session — navigate to user overview
        nav.navigate("user/overview.md")
        harness.inject_ephemeral(
            "New user session. Greet them warmly, introduce yourself briefly, "
            "and ask their name. Be natural — don't list your capabilities.",
            ttl=2,
        )

    # ── Run ──
    run_cli(
        harness,
        banner="🧠 Ada — Personal AI Companion with Persistent Memory",
        stream=True,
    )


if __name__ == "__main__":
    main()

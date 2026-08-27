#!/usr/bin/env python3
"""Extended interactive demo for the Obsidian Forge NavigableMemory corpus.

This keeps the original explore_obsidian_forge.py untouched and adds a larger
demo surface for the newer NavigableMemory features:

    - selectable backends: in-memory, JSON, SQLite, filesystem
    - persistent session state
    - document version/history commands
    - tag and reference graph commands
    - generated navigation/version/tag/reference/binary tools
    - optional migration/export command

Usage:
    python explore_obsidian_forge_extended.py
    python explore_obsidian_forge_extended.py --backend sqlite
    python explore_obsidian_forge_extended.py --backend json --reseed

Environment:
    OPENROUTER_API_KEY must be set for LLM turns. All slash commands work
    without contacting a model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv() -> bool:
        return False

from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_STATE_FILE = SCRIPT_DIR / "forge_extended_state.json"
DEFAULT_DATA_DIR = SCRIPT_DIR / "forge_extended_memory"
USER_TTL = 12
ASSISTANT_TTL = 12

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_local_imports() -> None:
    """Load just enough ToolAgents modules for local slash-command demos."""
    for name in list(sys.modules):
        if name == "ToolAgents" or name.startswith("ToolAgents."):
            del sys.modules[name]

    package_paths = {
        "ToolAgents": SRC_DIR / "ToolAgents",
        "ToolAgents.data_models": SRC_DIR / "ToolAgents" / "data_models",
        "ToolAgents.agent_harness": SRC_DIR / "ToolAgents" / "agent_harness",
        "ToolAgents.agent_memory": SRC_DIR / "ToolAgents" / "agent_memory",
        "ToolAgents.agent_memory.navigable_memory": (
            SRC_DIR / "ToolAgents" / "agent_memory" / "navigable_memory"
        ),
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    _load_module(
        "ToolAgents.data_models.messages",
        SRC_DIR / "ToolAgents" / "data_models" / "messages.py",
    )
    _load_module(
        "ToolAgents.agent_harness.prompt_composer",
        SRC_DIR / "ToolAgents" / "agent_harness" / "prompt_composer.py",
    )
    _load_module(
        "ToolAgents.agent_harness.smart_messages",
        SRC_DIR / "ToolAgents" / "agent_harness" / "smart_messages.py",
    )

    nav_root = SRC_DIR / "ToolAgents" / "agent_memory" / "navigable_memory"
    core = _load_module(
        "ToolAgents.agent_memory.navigable_memory.navigable_memory",
        nav_root / "navigable_memory.py",
    )
    nav_pkg = sys.modules["ToolAgents.agent_memory.navigable_memory"]
    for attr in (
        "NavigableMemory",
        "InMemoryBackend",
        "DepartureRecord",
        "Document",
        "DocumentVersion",
        "Reference",
        "RefType",
        "StorageBackend",
        "BinaryStorage",
        "VersionedStorage",
        "ReferenceStorage",
        "TagStorage",
    ):
        setattr(nav_pkg, attr, getattr(core, attr))

    json_backend = _load_module(
        "ToolAgents.agent_memory.navigable_memory.json_backend",
        nav_root / "json_backend.py",
    )
    sqlite_backend = _load_module(
        "ToolAgents.agent_memory.navigable_memory.sqlite_backend",
        nav_root / "sqlite_backend.py",
    )
    filesystem_backend = _load_module(
        "ToolAgents.agent_memory.navigable_memory.filesystem_backend",
        nav_root / "filesystem_backend.py",
    )
    migration = _load_module(
        "ToolAgents.agent_memory.navigable_memory.migration",
        nav_root / "migration.py",
    )

    nav_pkg.JSONBackend = json_backend.JSONBackend
    nav_pkg.SQLiteBackend = sqlite_backend.SQLiteBackend
    nav_pkg.FilesystemBackend = filesystem_backend.FilesystemBackend
    nav_pkg.migrate = migration.migrate
    nav_pkg.MigrationReport = migration.MigrationReport


try:
    from ToolAgents import FunctionTool, ToolRegistry
    from ToolAgents.agent_harness.prompt_composer import PromptComposer
    from ToolAgents.agent_harness.smart_messages import (
        ExpiryAction,
        MessageLifecycle,
        SmartMessageManager,
    )
    from ToolAgents.agent_memory.navigable_memory import (
        DepartureRecord,
        FilesystemBackend,
        InMemoryBackend,
        JSONBackend,
        NavigableMemory,
        RefType,
        SQLiteBackend,
        migrate,
    )
    from ToolAgents.agents import ChatToolAgent
    from ToolAgents.data_models.messages import ChatMessage
    from ToolAgents.provider import OpenAIChatAPI
    FULL_AGENT_AVAILABLE = True
    IMPORT_WARNING = ""
except Exception as import_error:
    _bootstrap_local_imports()
    from ToolAgents.agent_harness.prompt_composer import PromptComposer
    from ToolAgents.agent_harness.smart_messages import (
        ExpiryAction,
        MessageLifecycle,
        SmartMessageManager,
    )
    from ToolAgents.agent_memory.navigable_memory import (
        DepartureRecord,
        FilesystemBackend,
        InMemoryBackend,
        JSONBackend,
        NavigableMemory,
        RefType,
        SQLiteBackend,
        migrate,
    )
    from ToolAgents.data_models.messages import ChatMessage
    FunctionTool = None
    ToolRegistry = None
    ChatToolAgent = None
    OpenAIChatAPI = None
    FULL_AGENT_AVAILABLE = False
    IMPORT_WARNING = str(import_error)

from seed_obsidian_forge import seed as seed_knowledge_base


class CoreMemory:
    def __init__(self, block_limit: int = 800):
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
        separator = "\n" if current and not current.endswith("\n") else ""
        new_value = current + separator + content
        if len(new_value) > self.block_limit:
            return f"Error: would exceed {self.block_limit} chars."
        self.blocks[name] = new_value
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Appended to '{name}'."

    def delete_block(self, name: str) -> str:
        if name not in self.blocks:
            return f"Block '{name}' not found."
        del self.blocks[name]
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Block '{name}' deleted."

    def build_context(self) -> str:
        if not self.blocks:
            return "(no memory blocks stored)"
        lines = []
        for key, value in self.blocks.items():
            lines.append(
                f"<{key}> ({len(value)}/{self.block_limit} chars)\n"
                f"{value}\n</{key}>"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": dict(self.blocks),
            "last_modified": self.last_modified,
            "block_limit": self.block_limit,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        self.blocks = dict(data.get("blocks", {}))
        self.last_modified = data.get("last_modified", "restored")
        self.block_limit = int(data.get("block_limit", self.block_limit))


class DemoState:
    def __init__(
        self,
        *,
        backend_name: str,
        backend: Any,
        state_file: Path,
        reseed: bool = False,
    ):
        self.backend_name = backend_name
        self.backend = backend
        self.state_file = state_file
        self.core_memory = CoreMemory()
        self.message_manager = SmartMessageManager()
        self.nav_memory = NavigableMemory(
            backend=backend,
            on_depart=self.on_location_depart,
            context_window=4,
            include_siblings=True,
            include_parent=True,
        )
        self.turn_counter = 0
        self.reseed = reseed

    @property
    def document_count(self) -> int:
        count = getattr(self.backend, "count", None)
        if callable(count):
            return int(count())
        document_count = getattr(self.backend, "document_count", None)
        if isinstance(document_count, int):
            return document_count
        return len(self.nav_memory.list_at(""))

    def on_location_depart(self, record: DepartureRecord) -> None:
        snippet = record.content[:220].replace("\n", " ")
        msg = ChatMessage.create_system_message(
            f"[Previously at] {record.title} ({record.path})\n{snippet}..."
        )
        self.message_manager.add_message(
            msg,
            MessageLifecycle(ttl=8, on_expire=ExpiryAction.ARCHIVE),
        )
        print(f"  Departed: {record.title}")

    def seed_if_needed(self) -> None:
        if self.reseed:
            clear_backend(self.backend)
        if self.backend_name == "inmemory" or self.document_count == 0:
            seed_knowledge_base(self.nav_memory)
            add_demo_references(self.nav_memory)
            add_demo_asset(self.nav_memory)

    def save(self) -> dict[str, Any]:
        state = {
            "backend": self.backend_name,
            "core_memory": self.core_memory.to_dict(),
            "current_location": self.nav_memory.current_path,
            "location_history": self.nav_memory.history,
            "active_messages": [
                {
                    "role": sm.message.role.value,
                    "text": sm.message.get_as_text(),
                    "ttl": sm.lifecycle.ttl,
                    "turns_alive": sm.lifecycle.turns_alive,
                    "pinned": sm.lifecycle.pinned,
                    "on_expire": sm.lifecycle.on_expire.value,
                }
                for sm in self.message_manager.get_smart_messages()
            ],
            "archive": [msg.get_as_text() for msg in self.message_manager.archive],
            "tick_count": self.message_manager.tick_count,
            "saved_at": datetime.now().isoformat(),
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        close_backend(self.backend)
        return state

    def load(self) -> bool:
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.core_memory.from_dict(state.get("core_memory", {}))

            self.message_manager.clear()
            for msg_data in state.get("active_messages", []):
                self.message_manager.add_message(
                    message_from_role(msg_data["role"], msg_data["text"]),
                    MessageLifecycle(
                        ttl=msg_data["ttl"],
                        turns_alive=msg_data["turns_alive"],
                        pinned=msg_data["pinned"],
                        on_expire=ExpiryAction(msg_data["on_expire"]),
                    ),
                )

            current_location = state.get("current_location")
            if current_location:
                self.nav_memory.navigate(current_location)

            print(f"  Restored from {state.get('saved_at', 'unknown time')}")
            print(f"  Core memory: {len(self.core_memory.blocks)} blocks")
            print(
                "  Messages: "
                f"{self.message_manager.message_count} active, "
                f"{len(state.get('archive', []))} archived"
            )
            print(f"  Location: {self.nav_memory.current_title or 'none'}")
            return True
        except Exception as exc:
            print(f"  Failed to load state: {exc}")
            return False


class CoreMemorySet(BaseModel):
    """Set or overwrite a core memory block."""

    block_name: str = Field(..., description="Block name, e.g. priorities.")
    content: str = Field(..., description="Content to store.")

    def run(self) -> str:
        return ACTIVE_DEMO.core_memory.set_block(self.block_name, self.content)


class CoreMemoryAppend(BaseModel):
    """Append text to an existing core memory block."""

    block_name: str = Field(..., description="Block name.")
    content: str = Field(..., description="Text to append.")

    def run(self) -> str:
        return ACTIVE_DEMO.core_memory.append_block(self.block_name, self.content)


class CoreMemoryDelete(BaseModel):
    """Delete a core memory block."""

    block_name: str = Field(..., description="Block name.")

    def run(self) -> str:
        return ACTIVE_DEMO.core_memory.delete_block(self.block_name)


class ArchiveSearch(BaseModel):
    """Search archived messages for previous conversation context."""

    query: str = Field(..., description="Search term.")

    def run(self) -> str:
        return archive_search(ACTIVE_DEMO.message_manager, self.query)


class SessionStatus(BaseModel):
    """Summarize current memory, backend, location, and message state."""

    def run(self) -> str:
        return build_status(ACTIVE_DEMO)


ACTIVE_DEMO: DemoState


def create_backend(name: str, data_dir: Path) -> Any:
    if name == "inmemory":
        return InMemoryBackend()
    data_dir.mkdir(parents=True, exist_ok=True)
    if name == "json":
        return JSONBackend(str(data_dir / "obsidian_forge_memory.json"))
    if name == "sqlite":
        return SQLiteBackend(str(data_dir / "obsidian_forge_memory.db"))
    if name == "filesystem":
        return FilesystemBackend(str(data_dir / "obsidian_forge_files"))
    raise ValueError(f"Unknown backend: {name}")


def clear_backend(backend: Any) -> None:
    for doc in list(backend.list("")):
        backend.delete(doc.path)


def close_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def add_demo_references(nav: NavigableMemory) -> None:
    nav.add_reference(
        "studio/projects/ashenmoor/qa/critical-bugs.md",
        "studio/projects/ashenmoor/engineering/multiplayer.md",
        RefType.DEPENDS_ON,
        "Multiplayer desync must be checked with netcode docs.",
    )
    nav.add_reference(
        "studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md",
        "studio/projects/ashenmoor/art/vfx-backlog.md",
        RefType.DEPENDS_ON,
        "Thornqueen phase 3 is blocked on VFX.",
    )
    nav.add_reference(
        "studio/people/marcus-chen.md",
        "studio/projects/ashenmoor/design/combat/stagger-system.md",
        RefType.LINKS_TO,
        "Marcus owns combat and stagger tuning.",
    )


def add_demo_asset(nav: NavigableMemory) -> None:
    data = b"Forge architecture placeholder asset\n"
    nav.write_binary(
        "studio/assets/architecture-diagram.bin",
        "Architecture Diagram Placeholder",
        "application/octet-stream",
        data,
        caption="Placeholder binary asset used to demonstrate binary memory.",
        tags=["asset", "architecture", "demo"],
        metadata={"source": "explore_obsidian_forge_extended"},
    )


def message_from_role(role: str, text: str) -> ChatMessage:
    if role == "user":
        return ChatMessage.create_user_message(text)
    if role == "assistant":
        return ChatMessage.create_assistant_message(text)
    return ChatMessage.create_system_message(text)


def archive_search(manager: SmartMessageManager, query: str) -> str:
    results = []
    for msg in manager.archive:
        text = msg.get_as_text()
        if query.lower() in text.lower():
            results.append(text[:240])
    if not results:
        return f"No archived items matching '{query}'."
    return f"Found {len(results)} archived item(s):\n" + "\n---\n".join(results[:8])


def add_user_msg(demo: DemoState, text: str) -> None:
    demo.message_manager.add_message(
        ChatMessage.create_user_message(text),
        MessageLifecycle(ttl=USER_TTL, on_expire=ExpiryAction.ARCHIVE),
    )


def add_assistant_msg(demo: DemoState, text: str) -> None:
    demo.message_manager.add_message(
        ChatMessage.create_assistant_message(text),
        MessageLifecycle(ttl=ASSISTANT_TTL, on_expire=ExpiryAction.ARCHIVE),
    )


def inject_ephemeral(demo: DemoState, text: str, ttl: int = 2) -> None:
    demo.message_manager.add_message(
        ChatMessage.create_system_message(f"[Ephemeral] {text}"),
        MessageLifecycle(ttl=ttl, on_expire=ExpiryAction.REMOVE),
    )


def build_prompt_composer(demo: DemoState) -> PromptComposer:
    composer = PromptComposer()
    composer.add_module(
        "instructions",
        position=0,
        content=(
            "You are Forge, a studio manager assistant for Obsidian Forge "
            "Studios.\n\n"
            "Use the knowledge space actively: navigate before answering, "
            "read related documents, follow references, inspect tags, and use "
            "version history when a document may have changed.\n\n"
            "Memory systems available:\n"
            "1. Core Memory: compact user/session notes.\n"
            "2. Navigable Knowledge Space: persistent documents with tags, "
            "references, binary assets, and version history.\n"
            "3. Smart Message Archive: recently expired context and departed "
            "locations."
        ),
    )
    composer.add_module(
        "core_memory",
        position=5,
        content_fn=demo.core_memory.build_context,
        prefix=f"### Core Memory [modified: {demo.core_memory.last_modified}]",
        suffix="### End Core Memory",
    )
    composer.add_module(
        "location",
        position=10,
        content_fn=demo.nav_memory.build_context,
        prefix="### Knowledge Space - Current Location",
        suffix="### End Knowledge Space",
    )
    composer.add_module(
        "recent_locations",
        position=15,
        content_fn=demo.nav_memory.build_history_context,
        prefix="### Recently Visited",
        suffix="### End Recently Visited",
    )
    composer.add_module(
        "version_context",
        position=16,
        content_fn=lambda: demo.nav_memory.build_version_context(max_versions=2),
        prefix="### Previous Versions",
        suffix="### End Previous Versions",
    )
    composer.add_module(
        "reference_context",
        position=17,
        content_fn=lambda: current_reference_context(demo.nav_memory),
        prefix="### References",
        suffix="### End References",
    )
    composer.add_module(
        "session",
        position=20,
        content_fn=lambda: session_metadata(demo),
        prefix="### Session",
        suffix="### End Session",
    )
    return composer


def session_metadata(demo: DemoState) -> str:
    demo.turn_counter += 1
    location = demo.nav_memory.current_title if demo.nav_memory.current_path else "None"
    return (
        f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
        f"Turn: {demo.turn_counter}\n"
        f"Backend: {demo.backend_name}\n"
        f"Location: {location}\n"
        f"Active messages: {demo.message_manager.message_count}\n"
        f"Archived messages: {len(demo.message_manager.archive)}\n"
        f"Documents: {demo.document_count}\n"
        f"Tags: {', '.join(demo.nav_memory.list_tags()[:12])}"
    )


def current_reference_context(nav: NavigableMemory) -> str:
    if not nav.current_path:
        return "No current location."
    refs_from = nav.references_from(nav.current_path)
    refs_to = nav.references_to(nav.current_path)
    if not refs_from and not refs_to:
        return f"No references for {nav.current_path}."
    lines = []
    for ref in refs_from[:6]:
        lines.append(f"out [{ref.ref_type}] {ref.to_path}: {ref.note}")
    for ref in refs_to[:6]:
        lines.append(f"in [{ref.ref_type}] {ref.from_path}: {ref.note}")
    return "\n".join(lines)


def build_status(demo: DemoState) -> str:
    stats = getattr(demo.backend, "stats", lambda: {})()
    smart_messages = demo.message_manager.get_smart_messages()
    permanent = sum(1 for sm in smart_messages if sm.lifecycle.is_permanent)
    pinned = sum(1 for sm in smart_messages if sm.lifecycle.pinned)
    temporary = sum(
        1 for sm in smart_messages
        if not sm.lifecycle.is_permanent and not sm.lifecycle.pinned
    )
    return (
        f"Backend: {demo.backend_name}\n"
        f"Location: {demo.nav_memory.current_title} ({demo.nav_memory.current_path})\n"
        f"Documents: {demo.document_count}\n"
        f"Active messages: {demo.message_manager.message_count} "
        f"(permanent={permanent}, temporary={temporary}, pinned={pinned})\n"
        f"Archived messages: {len(demo.message_manager.archive)}\n"
        f"Core memory blocks: {len(demo.core_memory.blocks)}\n"
        f"Ticks: {demo.message_manager.tick_count}\n"
        f"Visited locations: {len(demo.nav_memory.history)}\n"
        f"Backend stats: {json.dumps(stats, ensure_ascii=False, default=str)}"
    )


def print_help() -> None:
    print("\nCommands:")
    print("  quit | /quit                 Exit and save")
    print("  /help                        Show this command list")
    print("  /memory                      Show core memory")
    print("  /location                    Show current location and recent history")
    print("  /status                      Show backend/session status")
    print("  /tree [prefix]               Show documents grouped by directory")
    print("  /archive [query]             Show or search archived messages")
    print("  /tags                        List tags")
    print("  /tag <tag>                   List docs tagged with tag")
    print("  /versions [path]             List previous versions")
    print("  /version <path> <n>          Read a historical version")
    print("  /diff <path> <from> [to]     Compare versions/current")
    print("  /rollback <path> <n>         Restore document to version")
    print("  /refs [path]                 Show references for path/current")
    print("  /walk [path] [direction] [depth]  Walk reference graph")
    print("  /binary <path>               Show binary document metadata")
    print("  /inject <text>               Inject ephemeral context")
    print("  /save                        Save session state")
    print("  /clear                       Clear active messages")
    print("  /export-json <path>          Export current backend to JSON")
    print()


def print_tree(nav: NavigableMemory, prefix: str = "") -> None:
    docs = sorted(nav.list_at(prefix), key=lambda doc: doc.path)
    if not docs:
        print(f"No documents under {prefix!r}.")
        return
    current_dir = None
    for doc in docs:
        parts = doc.path.rsplit("/", 1)
        dir_part = parts[0] + "/" if len(parts) > 1 else ""
        if dir_part != current_dir:
            current_dir = dir_part
            print(f"\n  [{current_dir}]")
        marker = " <-- HERE" if doc.path == nav.current_path else ""
        tags = f" [{', '.join(doc.tags[:4])}]" if doc.tags else ""
        print(f"    {doc.path.split('/')[-1]:42s} {doc.title}{tags}{marker}")


def handle_command(demo: DemoState, text: str) -> bool:
    nav = demo.nav_memory
    manager = demo.message_manager
    parts = shlex.split(text)
    command = parts[0].lower()
    args = parts[1:]

    if command in ("quit", "exit", "/quit", "/exit"):
        demo.save()
        print(f"State saved to {demo.state_file}")
        return False

    if command == "/help":
        print_help()
    elif command == "/memory":
        print(f"\nCore Memory (modified: {demo.core_memory.last_modified})")
        print(demo.core_memory.build_context())
    elif command == "/location":
        if nav.current_path:
            print(f"\n{nav.current_title}")
            print(f"Path: {nav.current_path}")
            print(f"Recent: {' -> '.join(nav.history[-6:])}")
        else:
            print("Not at any location.")
    elif command == "/status":
        print("\n" + build_status(demo))
    elif command == "/tree":
        print_tree(nav, args[0] if args else "")
    elif command == "/archive":
        if args:
            print(archive_search(manager, " ".join(args)))
        else:
            print(f"\nArchive ({len(manager.archive)} items)")
            for index, msg in enumerate(manager.archive[-12:]):
                print(f"  [{index}] {msg.get_as_text()[:140].replace(chr(10), ' ')}")
    elif command == "/tags":
        print("Tags:\n" + "\n".join(f"  - {tag}" for tag in nav.list_tags()))
    elif command == "/tag":
        if not args:
            print("Usage: /tag <tag>")
        else:
            for doc in nav.list_by_tag(args[0]):
                print(f"  - {doc.title} ({doc.path})")
    elif command == "/versions":
        path = args[0] if args else nav.current_path
        if not path:
            print("Usage: /versions [path]")
        else:
            versions = nav.list_history(path)
            if not versions:
                print(f"No previous versions for {path}.")
            for version in versions:
                note = f" - {version.change_note}" if version.change_note else ""
                print(f"  v{version.version} {version.created_at}{note}")
    elif command == "/version":
        if len(args) != 2:
            print("Usage: /version <path> <version>")
        else:
            print(nav.format_version(args[0], int(args[1])))
    elif command == "/diff":
        if len(args) not in (2, 3):
            print("Usage: /diff <path> <from_version> [to_version]")
        else:
            to_version = int(args[2]) if len(args) == 3 else None
            print(nav.compare_versions(args[0], int(args[1]), to_version))
    elif command == "/rollback":
        if len(args) != 2:
            print("Usage: /rollback <path> <version>")
        else:
            ok = nav.rollback(
                args[0],
                int(args[1]),
                author="interactive-demo",
                change_note=f"manual rollback to v{args[1]}",
            )
            print("Rolled back." if ok else "Rollback failed.")
    elif command == "/refs":
        path = args[0] if args else nav.current_path
        if not path:
            print("Usage: /refs [path]")
        else:
            refs = nav.references_from(path), nav.references_to(path)
            if not refs[0] and not refs[1]:
                print(f"No references for {path}.")
            for ref in refs[0]:
                print(f"  out [{ref.ref_type}] {ref.to_path} - {ref.note}")
            for ref in refs[1]:
                print(f"  in  [{ref.ref_type}] {ref.from_path} - {ref.note}")
    elif command == "/walk":
        path = args[0] if args else nav.current_path
        direction = args[1] if len(args) >= 2 else "both"
        depth = int(args[2]) if len(args) >= 3 else 2
        if not path:
            print("Usage: /walk [path] [outgoing|incoming|both] [depth]")
        else:
            print(nav.render_reference_walk(
                nav.walk_references(path, direction=direction, max_depth=depth)
            ))
    elif command == "/binary":
        if len(args) != 1:
            print("Usage: /binary <path>")
        else:
            doc = nav.read(args[0])
            if doc is None:
                print("Not found.")
            elif not doc.is_binary:
                print(f"{args[0]} is text ({doc.mime_type}).")
            else:
                print(f"{doc.title}\nType: {doc.mime_type}\nSize: {doc.human_size}")
                print(f"Caption: {doc.content}")
    elif command == "/inject":
        if not args:
            print("Usage: /inject <text>")
        else:
            inject_ephemeral(demo, " ".join(args), ttl=3)
            print("Injected ephemeral context for 3 turns.")
    elif command == "/save":
        demo.save()
        print(f"State saved to {demo.state_file}")
    elif command == "/clear":
        manager.clear()
        add_pinned_system_message(demo)
        print("Active messages cleared; pinned system message restored.")
    elif command == "/export-json":
        if len(args) != 1:
            print("Usage: /export-json <path>")
        else:
            destination = JSONBackend(args[0])
            report = migrate(demo.backend, destination, overwrite=True)
            destination.close()
            print(report)
    else:
        return True

    return True


def add_pinned_system_message(demo: DemoState) -> None:
    demo.message_manager.add_message(
        ChatMessage.create_system_message(
            "[SYSTEM] Navigate to relevant knowledge before answering. "
            "Use tags, references, and version history when helpful."
        ),
        MessageLifecycle(pinned=True),
    )


def build_agent(args: argparse.Namespace):
    if not FULL_AGENT_AVAILABLE:
        return None, None, f"full ToolAgents agent imports unavailable: {IMPORT_WARNING}"
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        return None, None, "missing API key"
    api = OpenAIChatAPI(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
    )
    agent = ChatToolAgent(chat_api=api)
    settings = api.get_default_settings()
    settings.temperature = args.temperature
    settings.top_p = 1.0
    return agent, settings, ""


def build_tool_registry(demo: DemoState) -> ToolRegistry:
    if not FULL_AGENT_AVAILABLE:
        return None
    registry = ToolRegistry()
    registry.add_tools([FunctionTool(tool) for tool in demo.nav_memory.create_tools()])
    registry.add_tools([
        FunctionTool(CoreMemorySet),
        FunctionTool(CoreMemoryAppend),
        FunctionTool(CoreMemoryDelete),
        FunctionTool(ArchiveSearch),
        FunctionTool(SessionStatus),
    ])
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["inmemory", "json", "sqlite", "filesystem"],
        default="sqlite",
        help="Knowledge backend to use. Default: sqlite.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory for persistent backend files.",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="JSON file for session/core-message state.",
    )
    parser.add_argument("--reseed", action="store_true", help="Clear and reseed knowledge.")
    parser.add_argument("--model", default="xiaomi/mimo-v2-pro")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    backend = create_backend(args.backend, Path(args.data_dir))
    demo = DemoState(
        backend_name=args.backend,
        backend=backend,
        state_file=Path(args.state_file),
        reseed=args.reseed,
    )

    global ACTIVE_DEMO
    ACTIVE_DEMO = demo

    demo.seed_if_needed()
    restored = demo.load()
    if not restored:
        demo.core_memory.set_block(
            "persona",
            "I am Forge, studio manager assistant for Obsidian Forge Studios.",
        )
        demo.core_memory.set_block(
            "priorities",
            "Key dates: Ashenmoor EA Aug 2026, Drift Protocol E3 Jun 2026.",
        )
        demo.core_memory.set_block("user_info", "No user information yet.")
        add_pinned_system_message(demo)
        inject_ephemeral(
            demo,
            "New session. Suggest a studio status overview or specific project check.",
            ttl=2,
        )
        demo.nav_memory.navigate("studio/overview.md")

    agent, settings, agent_error = build_agent(args)
    tool_registry = build_tool_registry(demo)
    composer = build_prompt_composer(demo)

    print("=" * 72)
    print("Forge Extended - Obsidian Forge Studios Manager Assistant")
    print(f"Backend: {args.backend}")
    print(f"Knowledge: {demo.document_count} documents")
    print(f"Session: {'restored' if restored else 'new'}")
    if agent is None:
        print(f"LLM turns disabled: {agent_error} ({args.api_key_env})")
    print("=" * 72)
    print_help()
    print("Try: What are our critical bugs and who should I talk to?")
    print("Try: Check Thornqueen blockers and follow related references.")
    print()

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            demo.save()
            print(f"\nState saved to {demo.state_file}. Session ended.")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/") or user_input.lower() in {"quit", "exit"}:
            keep_running = handle_command(demo, user_input)
            if not keep_running:
                return 0
            continue

        tick_result = demo.message_manager.tick()
        if tick_result.removed:
            print(f"  {len(tick_result.removed)} ephemeral message(s) expired")
        if tick_result.archived:
            print(f"  {len(tick_result.archived)} message(s) archived")

        add_user_msg(demo, user_input)
        composer.update_module(
            "core_memory",
            prefix=f"### Core Memory [modified: {demo.core_memory.last_modified}]",
        )
        messages = [
            ChatMessage.create_system_message(composer.compile()),
            *demo.message_manager.get_active_messages(),
        ]

        if agent is None:
            print(
                "\nForge > LLM is not configured. Use slash commands, or set "
                f"{args.api_key_env} and restart."
            )
            continue

        try:
            response = agent.get_response(
                messages=messages,
                settings=settings,
                tool_registry=tool_registry,
            )
            text = response.response.strip()
            print(f"\nForge > {text}")
            add_assistant_msg(demo, text)
            for msg in response.messages:
                if msg.role.value not in ("user", "assistant"):
                    demo.message_manager.add_message(
                        msg,
                        MessageLifecycle(
                            ttl=ASSISTANT_TTL,
                            on_expire=ExpiryAction.ARCHIVE,
                        ),
                    )
        except Exception as exc:
            print(f"\nError: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())

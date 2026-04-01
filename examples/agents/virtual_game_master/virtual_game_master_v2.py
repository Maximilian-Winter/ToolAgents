"""
=============================================================================
virtual_game_master_v2.py — Unified Agent Harness
=============================================================================

A fully generic agent harness that combines:
  - VirtualGameMasterConfig   → configuration from .env or JSON
  - PromptComposer            → modular, dynamic system prompt
  - SmartMessageManager       → TTL-based message windowing and archival
  - NavigableMemory           → hierarchical knowledge navigation
  - ContextAppState           → structured XML state (LLM-updatable)
  - CoreMemory                → agent-editable runtime notes
  - ChatToolAgent             → LLM with tool calling
  - CommandSystem             → pluggable slash/prefix commands
  - Save/Load                 → full session persistence

Works for any domain: RPGs, studio management, research assistants,
customer support — anything that benefits from navigable knowledge,
structured state, and agent self-memory.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                     VirtualGameMaster2                          │
  │                                                                 │
  │  Config ──► Provider ──► ChatToolAgent                          │
  │                              │                                  │
  │  PromptComposer ─────────────┤                                  │
  │  ├─ instructions (template)  │                                  │
  │  ├─ app_state (XML dynamic)  │  compile() + get_active_messages │
  │  ├─ location (NavigableMemory)│ ─────────────────────────────►  │
  │  ├─ core_memory (dynamic)    │       get_response()             │
  │  └─ metadata (dynamic)       │                                  │
  │                              │                                  │
  │  SmartMessageManager ────────┘                                  │
  │  ├─ pinned system messages                                      │
  │  ├─ TTL user/assistant messages                                 │
  │  └─ location departure messages (from NavigableMemory callback) │
  │                                                                 │
  │  NavigableMemory ──► on_depart callback ──► SmartMessageManager │
  │  CoreMemory ──► agent tools (set/append/delete/search)          │
  │  ContextAppState ──► agent tool (update_state via XML fragment) │
  └─────────────────────────────────────────────────────────────────┘
"""

import json
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple, Callable, List

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
from ToolAgents.agent_memory.navigable_memory import (
    NavigableMemory,
    DepartureRecord,
)
from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents.utilities.message_template import MessageTemplate


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

class HarnessConfig:
    """
    Configuration for VirtualGameMaster2.
    Load from .env file, JSON file, or construct programmatically.
    """

    def __init__(self):
        # ── Provider ──
        self.API_TYPE: str = "openai"
        self.API_KEY: Optional[str] = None
        self.API_URL: str = ""
        self.MODEL: str = ""

        # ── Sampling ──
        self.TEMPERATURE: float = 0.7
        self.TOP_P: float = 1.0
        self.MAX_TOKENS: int = 4096

        # ── Prompt ──
        self.SYSTEM_MESSAGE_FILE: str = ""
        self.INITIAL_STATE_FILE: str = ""

        # ── Message Management ──
        self.USER_MSG_TTL: int = 12
        self.ASSISTANT_MSG_TTL: int = 12
        self.LOCATION_MSG_TTL: int = 12

        # ── Navigation ──
        self.CONTEXT_WINDOW: int = 3
        self.INCLUDE_SIBLINGS: bool = True
        self.INCLUDE_PARENT: bool = True

        # ── Persistence ──
        self.SAVE_FOLDER: str = "saves"

        # ── Commands ──
        self.COMMAND_PREFIX: str = "/"

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "HarnessConfig":
        load_dotenv(env_file)
        config = cls()
        for key in vars(config):
            env_val = os.getenv(key)
            if env_val is not None:
                config._set_typed(key, env_val)
        return config

    @classmethod
    def from_json(cls, json_file: str) -> "HarnessConfig":
        with open(json_file, "r") as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                config._set_typed(key, value)
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessConfig":
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                config._set_typed(key, value)
        return config

    def _set_typed(self, key: str, value: Any):
        current = getattr(self, key)
        if isinstance(current, bool):
            setattr(self, key, str(value).lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(self, key, int(value))
        elif isinstance(current, float):
            setattr(self, key, float(value))
        elif current is None:
            setattr(self, key, value)
        else:
            setattr(self, key, type(current)(value))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items()}

    def to_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Core Memory
# ═══════════════════════════════════════════════════════════════════

class CoreMemory:
    """Agent-editable key-value memory for runtime notes and observations."""

    def __init__(self, block_limit: int = 600):
        self.blocks: Dict[str, str] = {}
        self.block_limit = block_limit
        self.last_modified: str = "never"

    def set_block(self, name: str, content: str) -> str:
        if len(content) > self.block_limit:
            return f"Error: exceeds {self.block_limit} char limit."
        self.blocks[name] = content
        self.last_modified = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Memory '{name}' updated."

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
            return f"Block '{name}' deleted."
        return f"Block '{name}' not found."

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
# Memory Tools (Pydantic models for agent tool calling)
# ═══════════════════════════════════════════════════════════════════

# Module-level reference, bound by VirtualGameMaster2.__init__
_core_memory: Optional[CoreMemory] = None
_app_state: Optional[ContextAppState] = None
_msg_manager: Optional[SmartMessageManager] = None


class CoreMemorySet(BaseModel):
    """Set or overwrite a core memory block. Use for observations, notes, action items."""
    block_name: str = Field(..., description="Block name (e.g. 'user_info', 'priorities').")
    content: str = Field(..., description="Content to store.")

    def run(self) -> str:
        if _core_memory is None:
            return "Error: memory not initialized."
        return _core_memory.set_block(self.block_name, self.content)


class CoreMemoryAppend(BaseModel):
    """Append text to an existing core memory block."""
    block_name: str = Field(..., description="Block name.")
    content: str = Field(..., description="Text to append.")

    def run(self) -> str:
        if _core_memory is None:
            return "Error: memory not initialized."
        return _core_memory.append_block(self.block_name, self.content)


class CoreMemoryDelete(BaseModel):
    """Delete a core memory block."""
    block_name: str = Field(..., description="Block name.")

    def run(self) -> str:
        if _core_memory is None:
            return "Error: memory not initialized."
        return _core_memory.delete_block(self.block_name)


class ArchiveSearch(BaseModel):
    """Search archived messages for past conversations and departed locations."""
    query: str = Field(..., description="Search term.")

    def run(self) -> str:
        if _msg_manager is None:
            return "Error: message manager not initialized."
        results = []
        for msg in _msg_manager.archive:
            text = msg.get_as_text()
            if self.query.lower() in text.lower():
                results.append(text[:200])
        if results:
            return f"Found {len(results)} archived item(s):\n" + "\n---\n".join(results[:5])
        return f"No archived items matching '{self.query}'."


class UpdateAppState(BaseModel):
    """
    Update the structured application state with an XML fragment.
    The fragment must be wrapped in the root element tag.
    Matching elements update in place, new elements are appended.
    """
    xml_fragment: str = Field(
        ..., description="XML fragment to merge into the application state."
    )

    def run(self) -> str:
        if _app_state is None:
            return "Error: app state not initialized."
        try:
            _app_state.update_from_xml(self.xml_fragment)
            return "Application state updated successfully."
        except Exception as e:
            return f"Error updating state: {e}"


# ═══════════════════════════════════════════════════════════════════
# Provider Factory
# ═══════════════════════════════════════════════════════════════════

def create_provider(config: HarnessConfig):
    """Create an LLM provider from config. Returns (api, settings)."""
    from ToolAgents.provider import (
        OpenAIChatAPI,
        AnthropicChatAPI,
        GroqChatAPI,
        MistralChatAPI,
    )

    api_type = config.API_TYPE.lower()

    if api_type in ("openai", "openrouter", "local", "vllm", "llamacpp"):
        kwargs = {"api_key": config.API_KEY, "model": config.MODEL}
        if config.API_URL:
            kwargs["base_url"] = config.API_URL
        api = OpenAIChatAPI(**kwargs)
    elif api_type == "anthropic":
        api = AnthropicChatAPI(api_key=config.API_KEY, model=config.MODEL)
    elif api_type == "groq":
        api = GroqChatAPI(api_key=config.API_KEY, model=config.MODEL)
    elif api_type == "mistral":
        api = MistralChatAPI(api_key=config.API_KEY, model=config.MODEL)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

    settings = api.get_default_settings()
    settings.temperature = config.TEMPERATURE
    settings.top_p = config.TOP_P
    return api, settings


# ═══════════════════════════════════════════════════════════════════
# VirtualGameMaster2
# ═══════════════════════════════════════════════════════════════════

class VirtualGameMaster2:
    """
    Unified agent harness combining all ToolAgents subsystems.

    Fully generic — works for RPGs, studio management, research,
    or any domain with navigable knowledge and structured state.

    Systems:
      - PromptComposer: modular system prompt, re-rendered each turn
      - SmartMessageManager: TTL-based message windowing and archival
      - NavigableMemory: hierarchical knowledge with auto-context loading
      - ContextAppState: structured XML state, LLM-updatable
      - CoreMemory: agent-editable runtime notes
      - Tool calling: NavigableMemory tools + memory tools + custom tools
    """

    def __init__(
        self,
        config: HarnessConfig,
        nav_memory: NavigableMemory,
        agent: Optional[ChatToolAgent] = None,
        settings=None,
        app_state: Optional[ContextAppState] = None,
        system_template: Optional[MessageTemplate] = None,
        initial_core_memory: Optional[Dict[str, str]] = None,
        extra_tools: Optional[List] = None,
        pinned_messages: Optional[List[str]] = None,
        summarize_on_depart: bool = False,
        summarizer_settings=None,
        debug_mode: bool = False,
    ):
        global _core_memory, _app_state, _msg_manager

        self.config = config
        self.debug_mode = debug_mode
        self.summarize_on_depart = summarize_on_depart

        # ── LLM Provider ──
        if agent is not None:
            self.agent = agent
            self.settings = settings
        else:
            api, self.settings = create_provider(config)
            self.agent = ChatToolAgent(chat_api=api)

        # Summarizer uses its own settings (lower temperature) or falls back to main
        self.summarizer_settings = summarizer_settings

        # ── Navigable Memory ──
        self.nav_memory = nav_memory

        # ── Smart Message Manager ──
        self.msg_manager = SmartMessageManager()
        _msg_manager = self.msg_manager

        # ── Core Memory ──
        self.core_memory = CoreMemory()
        _core_memory = self.core_memory

        if initial_core_memory:
            for name, content in initial_core_memory.items():
                self.core_memory.set_block(name, content)

        # ── App State (optional structured XML state) ──
        self.app_state: Optional[ContextAppState] = app_state
        _app_state = self.app_state

        # ── System Template ──
        self.system_template = system_template
        if system_template is None and config.SYSTEM_MESSAGE_FILE:
            if os.path.exists(config.SYSTEM_MESSAGE_FILE):
                self.system_template = MessageTemplate.from_file(config.SYSTEM_MESSAGE_FILE)

        # ── Prompt Composer ──
        self.composer = self._build_composer()

        # ── NavigableMemory departure callback ──
        # Wire the on_depart to inject location messages into SmartMessageManager
        self.nav_memory.on_depart = self._on_location_depart

        # ── Tool Registry ──
        self.tool_registry = ToolRegistry()

        # Navigation tools (auto-generated by NavigableMemory)
        nav_tools = [FunctionTool(t) for t in self.nav_memory.create_tools()]
        self.tool_registry.add_tools(nav_tools)

        # Memory tools
        self.tool_registry.add_tools([
            FunctionTool(CoreMemorySet),
            FunctionTool(CoreMemoryAppend),
            FunctionTool(CoreMemoryDelete),
            FunctionTool(ArchiveSearch),
        ])

        # App state update tool (if app_state is provided)
        if self.app_state is not None:
            self.tool_registry.add_tools([FunctionTool(UpdateAppState)])

        # Extra user-provided tools
        if extra_tools:
            self.tool_registry.add_tools(extra_tools)

        # ── Pinned Messages ──
        if pinned_messages:
            for text in pinned_messages:
                msg = ChatMessage.create_system_message(text)
                self.msg_manager.add_message(msg, MessageLifecycle(pinned=True))

        # ── Turn Counter ──
        self.turn_count = 0

        # ── Persistence ──
        self.save_folder = config.SAVE_FOLDER
        os.makedirs(self.save_folder, exist_ok=True)

    # ── Prompt Composer Setup ──────────────────────────────────────

    def _build_composer(self) -> PromptComposer:
        composer = PromptComposer()

        # Module 0: Instructions (from template or static)
        if self.system_template and self.app_state:
            instructions = self.system_template.generate_message_content(
                self.app_state.template_fields
            )
        elif self.system_template:
            instructions = self.system_template.generate_message_content()
        else:
            instructions = (
                "You are an intelligent assistant with navigable knowledge and self-editable memory.\n\n"
                "MEMORY SYSTEMS:\n"
                "1. Core Memory — key-value blocks you can read/edit (shown below).\n"
                "   Use core_memory_set/append/delete to manage it.\n"
                "2. Knowledge Space — navigable document hierarchy.\n"
                "   Use navigate_to_document, list_locations, search_knowledge, read_document.\n"
                "   When you navigate away, old location context lingers then archives.\n\n"
                "BEHAVIORS:\n"
                "- Navigate to relevant knowledge BEFORE answering questions.\n"
                "- Save important observations to core memory.\n"
                "- Use archive_search to recall old conversations.\n"
                "- Be proactive: suggest related documents and flag risks."
            )

        composer.add_module(name="instructions", position=0, content=instructions)

        # Module 5: App State (dynamic, if present)
        if self.app_state is not None:
            composer.add_module(
                name="app_state",
                position=5,
                content_fn=lambda: self.app_state.get_state_string(),
                prefix="### Application State",
                suffix="### End Application State",
            )

        # Module 10: Core Memory (dynamic)
        composer.add_module(
            name="core_memory",
            position=10,
            content_fn=lambda: self.core_memory.build_context(),
            prefix=f"### Core Memory [modified: {self.core_memory.last_modified}]",
            suffix="### End Core Memory",
        )

        # Module 15: Knowledge Location (dynamic, from NavigableMemory)
        composer.add_module(
            name="location",
            position=15,
            content_fn=self.nav_memory.build_context,
            prefix="### Knowledge Space — Current Location",
            suffix="### End Knowledge Space",
        )

        # Module 20: Navigation History (dynamic)
        composer.add_module(
            name="nav_history",
            position=20,
            content_fn=self.nav_memory.build_history_context,
            prefix="### Recently Visited",
            suffix="### End Recently Visited",
        )

        # Module 25: Session Metadata (dynamic)
        def metadata_fn() -> str:
            loc = self.nav_memory.current_title if self.nav_memory.current_path else "None"
            return (
                f"Time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Turn: {self.turn_count}\n"
                f"Location: {loc}\n"
                f"Active messages: {self.msg_manager.message_count}\n"
                f"Archived: {len(self.msg_manager.archive)}"
            )

        composer.add_module(
            name="metadata",
            position=25,
            content_fn=metadata_fn,
            prefix="### Session",
            suffix="### End Session",
        )

        return composer

    # ── NavigableMemory Departure Callback ─────────────────────────

    def _on_location_depart(self, record: DepartureRecord):
        """When leaving a location, optionally summarize and inject as TTL message.

        If summarize_on_depart is True and an agent is available:
          1. Makes a separate LLM call to summarize what happened
          2. Appends the summary to the location's document in the backend
          3. Injects the summary as a SmartMessage with TTL

        Otherwise, just injects the raw content snippet as a SmartMessage.
        """
        summary = None

        if self.summarize_on_depart:
            summary = self._summarize_location(record)

            # Write summary back to the KB document
            if summary:
                self.nav_memory.append(
                    record.path,
                    f"**Session event:** {summary}",
                )

        # Inject into message stream (summary if available, raw snippet otherwise)
        if summary:
            msg_text = f"[Departed] {record.title} ({record.path})\nSummary: {summary}"
        else:
            snippet = record.content[:300].replace("\n", " ")
            msg_text = f"[Previously at] {record.title} ({record.path})\n{snippet}..."

        msg = ChatMessage.create_system_message(msg_text)
        self.msg_manager.add_message(
            msg,
            lifecycle=MessageLifecycle(
                ttl=self.config.LOCATION_MSG_TTL,
                on_expire=ExpiryAction.ARCHIVE,
            ),
        )

        if self.debug_mode:
            if summary:
                print(f"  📜 Departed {record.title}: {summary[:80]}...")
            else:
                print(f"  📍 Departed: {record.title}")

    def _summarize_location(self, record: DepartureRecord) -> Optional[str]:
        """Make an LLM call to summarize what happened at a location.

        Uses the recent active messages as conversation context.
        Returns the summary string, or None on failure.
        """
        # Gather recent conversation for context
        active_messages = self.msg_manager.get_active_messages()
        conversation_text = "\n".join(
            f"{m.get_role()}: {m.get_as_text()}"
            for m in active_messages[-10:]
        )

        summary_messages = [
            ChatMessage.create_system_message(
                "You are a concise note-taker. Summarize what happened at this "
                "location in 2-3 sentences. Focus on: key events, decisions made, "
                "information discovered, and any action items. "
                "Write in past tense, third person. Output ONLY the summary."
            ),
            ChatMessage.create_user_message(
                f"Location: {record.title}\n"
                f"Path: {record.path}\n\n"
                f"Recent conversation:\n{conversation_text}\n\n"
                f"Summarize what happened here:"
            ),
        ]

        try:
            # Use summarizer settings (lower temp) if provided, else main settings
            settings = self.summarizer_settings or self.settings
            response = self.agent.get_response(
                messages=summary_messages,
                settings=settings,
                tool_registry=ToolRegistry(),  # no tools for summarization
            )
            return response.response.strip()
        except Exception as e:
            if self.debug_mode:
                print(f"  ⚠️ Summary failed: {e}")
            return None

    # ── Message Helpers ────────────────────────────────────────────

    def add_user_message(self, text: str):
        msg = ChatMessage.create_user_message(text)
        self.msg_manager.add_message(
            msg,
            MessageLifecycle(ttl=self.config.USER_MSG_TTL, on_expire=ExpiryAction.ARCHIVE),
        )

    def add_assistant_message(self, text: str):
        msg = ChatMessage.create_assistant_message(text)
        self.msg_manager.add_message(
            msg,
            MessageLifecycle(ttl=self.config.ASSISTANT_MSG_TTL, on_expire=ExpiryAction.ARCHIVE),
        )

    def inject_ephemeral(self, text: str, ttl: int = 2):
        msg = ChatMessage.create_system_message(f"[Ephemeral] {text}")
        self.msg_manager.add_message(
            msg,
            MessageLifecycle(ttl=ttl, on_expire=ExpiryAction.REMOVE),
        )

    # ── Core Loop ──────────────────────────────────────────────────

    def _build_messages(self) -> list[ChatMessage]:
        """Compile system prompt + active messages for the LLM call."""
        # Update dynamic prefixes
        self.composer.update_module(
            "core_memory",
            prefix=f"### Core Memory [modified: {self.core_memory.last_modified}]",
        )

        # If using template + app state, refresh instructions
        if self.system_template and self.app_state:
            instructions = self.system_template.generate_message_content(
                self.app_state.template_fields
            )
            self.composer.update_module("instructions", content=instructions)

        system_prompt = self.composer.compile()

        if self.debug_mode:
            print(f"\n[DEBUG] System prompt ({len(system_prompt)} chars)")

        return [
            ChatMessage.create_system_message(system_prompt),
            *self.msg_manager.get_active_messages(),
        ]

    def get_response(self, user_input: str) -> str:
        """Get a non-streaming response."""
        self.turn_count += 1
        tick_result = self.msg_manager.tick()
        self._log_tick(tick_result)

        self.add_user_message(user_input)
        messages = self._build_messages()

        chat_response = self.agent.get_response(
            messages=messages,
            settings=self.settings,
            tool_registry=self.tool_registry,
        )

        response_text = chat_response.response.strip()
        self.add_assistant_message(response_text)
        self._add_tool_messages(chat_response)

        return response_text

    def get_streaming_response(self, user_input: str) -> Generator[str, None, None]:
        """Get a streaming response, yielding chunks."""
        self.turn_count += 1
        tick_result = self.msg_manager.tick()
        self._log_tick(tick_result)

        self.add_user_message(user_input)
        messages = self._build_messages()

        full_response = ""
        stream = self.agent.get_streaming_response(
            messages=messages,
            settings=self.settings,
            tool_registry=self.tool_registry,
        )

        chat_response = None
        for res in stream:
            full_response += res.chunk
            yield res.chunk
            if res.finished:
                chat_response = res.finished_response

        if chat_response is not None:
            self.add_assistant_message(full_response.strip())
            self._add_tool_messages(chat_response)
        else:
            self.add_assistant_message(full_response.strip())

    def _add_tool_messages(self, chat_response):
        """Add tool-call/result messages from the agent response."""
        for msg in chat_response.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role not in ("user", "assistant"):
                self.msg_manager.add_message(
                    msg,
                    lifecycle=MessageLifecycle(
                        ttl=self.config.ASSISTANT_MSG_TTL,
                        on_expire=ExpiryAction.ARCHIVE,
                    ),
                )

    def _log_tick(self, tick_result):
        if tick_result.removed:
            print(f"  🗑️  {len(tick_result.removed)} ephemeral message(s) expired")
        if tick_result.archived:
            print(f"  📦 {len(tick_result.archived)} message(s) archived")

    # ── Persistence ────────────────────────────────────────────────

    def save(self, name: str = "autosave") -> str:
        """Save full session state."""
        filepath = os.path.join(self.save_folder, f"{name}.json")

        state = {
            "saved_at": dt.datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "turn_count": self.turn_count,
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
                for sm in self.msg_manager.get_smart_messages()
            ],
            "archive": [m.get_as_text() for m in self.msg_manager.archive],
        }

        # Save app state XML separately if present
        if self.app_state is not None:
            state["app_state_fields"] = self.app_state.template_fields

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        return filepath

    def load(self, name: str = "autosave") -> bool:
        """Load session state. Returns True if successful."""
        filepath = os.path.join(self.save_folder, f"{name}.json")
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Restore core memory
            self.core_memory.from_dict(state.get("core_memory", {}))

            # Restore turn count
            self.turn_count = state.get("turn_count", 0)

            # Restore app state
            if self.app_state and "app_state_fields" in state:
                self.app_state.template_fields = state["app_state_fields"]

            # Restore messages
            self.msg_manager.clear()
            for md in state.get("active_messages", []):
                role = md["role"]
                if role == "user":
                    msg = ChatMessage.create_user_message(md["text"])
                elif role == "assistant":
                    msg = ChatMessage.create_assistant_message(md["text"])
                else:
                    msg = ChatMessage.create_system_message(md["text"])

                lifecycle = MessageLifecycle(
                    ttl=md.get("ttl"),
                    turns_alive=md.get("turns_alive", 0),
                    pinned=md.get("pinned", False),
                    on_expire=ExpiryAction(md.get("on_expire", "archive")),
                )
                self.msg_manager.add_message(msg, lifecycle)

            # Restore navigation
            loc = state.get("current_location")
            if loc:
                self.nav_memory.navigate(loc)

            if self.debug_mode:
                print(f"  Restored from {state.get('saved_at', '?')}")
                print(f"  Core memory: {len(self.core_memory.blocks)} blocks")
                print(f"  Messages: {self.msg_manager.message_count} active")
                print(f"  Location: {self.nav_memory.current_title or 'none'}")

            return True

        except Exception as e:
            print(f"  Failed to load state: {e}")
            return False

    # ── Convenience Properties ─────────────────────────────────────

    @property
    def current_location(self) -> Optional[str]:
        return self.nav_memory.current_path

    @property
    def current_location_title(self) -> Optional[str]:
        return self.nav_memory.current_title

    @property
    def active_message_count(self) -> int:
        return self.msg_manager.message_count

    @property
    def archive_count(self) -> int:
        return len(self.msg_manager.archive)


# ═══════════════════════════════════════════════════════════════════
# Interactive CLI Runner
# ═══════════════════════════════════════════════════════════════════

def run_cli(
    harness: VirtualGameMaster2,
    banner: str = "Agent Harness",
    stream: bool = True,
):
    """
    Generic interactive CLI loop for VirtualGameMaster2.
    Works for any domain — just pass a configured harness.
    """
    # Try loading previous session
    restored = harness.load()

    print()
    print("=" * 64)
    print(f"  {banner}")
    print(f"  Session: {'restored' if restored else 'new'}")
    if harness.current_location:
        print(f"  Location: {harness.current_location_title}")
    print("=" * 64)
    print()
    print("Commands:")
    print(f"  quit            — Exit (auto-saves)")
    print(f"  /memory         — Show core memory")
    print(f"  /location       — Current knowledge location")
    print(f"  /archive        — Show archived messages")
    print(f"  /status         — System status")
    print(f"  /tree           — Knowledge space tree")
    print(f"  /inject <msg>   — Inject ephemeral context")
    print(f"  /save [name]    — Save state")
    print(f"  /clear          — Clear messages")
    if harness.app_state:
        print(f"  /state          — Show application state")
    print()

    while True:
        try:
            user_input = input("\n🧑 You > ").strip()
        except (KeyboardInterrupt, EOFError):
            harness.save()
            print("\n  💾 Saved. Session ended.")
            break

        if not user_input:
            continue

        # ── Built-in commands ──
        if user_input.lower() == "quit":
            harness.save()
            print("  💾 Saved. Session ended.")
            break

        elif user_input.lower() == "/memory":
            print(f"\n📝 Core Memory (modified: {harness.core_memory.last_modified}):")
            print(harness.core_memory.build_context())
            continue

        elif user_input.lower() == "/location":
            if harness.nav_memory.current_path:
                print(f"\n📍 {harness.nav_memory.current_title}")
                print(f"   Path: {harness.nav_memory.current_path}")
                if harness.nav_memory.history:
                    recent = harness.nav_memory.history[-5:]
                    print(f"   Recent: {' → '.join(recent)}")
            else:
                print("\n📍 Not at any location.")
            continue

        elif user_input.lower() == "/archive":
            archive = harness.msg_manager.archive
            print(f"\n📦 Archive ({len(archive)} items):")
            for i, msg in enumerate(archive[-10:]):
                text = msg.get_as_text()[:100].replace("\n", " ")
                print(f"  [{i}] {text}")
            if not archive:
                print("  (empty)")
            continue

        elif user_input.lower() == "/status":
            print(f"\n📊 Status:")
            print(f"  Location: {harness.nav_memory.current_title} ({harness.nav_memory.current_path})")
            print(f"  Turn: {harness.turn_count}")
            print(f"  Active messages: {harness.msg_manager.message_count}")
            print(f"  Archived: {len(harness.msg_manager.archive)}")
            print(f"  Core memory: {len(harness.core_memory.blocks)} blocks")
            print(f"  Locations visited: {len(harness.nav_memory.history)}")
            continue

        elif user_input.lower() == "/tree":
            print(f"\n🌳 Knowledge Space:")
            docs = sorted(harness.nav_memory.list_at(""), key=lambda d: d.path)
            current_dir = ""
            for d in docs:
                parts = d.path.rsplit("/", 1)
                dir_part = parts[0] + "/" if len(parts) > 1 else ""
                if dir_part != current_dir:
                    current_dir = dir_part
                    print(f"\n  📁 {current_dir}")
                marker = " ◀ HERE" if d.path == harness.nav_memory.current_path else ""
                name = d.path.split("/")[-1]
                print(f"      {name:40s} {d.title}{marker}")
            continue

        elif user_input.lower().startswith("/inject "):
            harness.inject_ephemeral(user_input[8:].strip(), ttl=3)
            print("  💉 Injected (expires in 3 turns)")
            continue

        elif user_input.lower().startswith("/save"):
            parts = user_input.split(maxsplit=1)
            name = parts[1] if len(parts) > 1 else "autosave"
            filepath = harness.save(name)
            print(f"  💾 Saved to {filepath}")
            continue

        elif user_input.lower() == "/clear":
            harness.msg_manager.clear()
            harness.msg_manager.add_message(
                ChatMessage.create_system_message("[SYSTEM] Messages cleared. Context reset."),
                MessageLifecycle(pinned=True),
            )
            print("  🧹 Messages cleared.")
            continue

        elif user_input.lower() == "/state" and harness.app_state:
            print(f"\n📜 Application State:\n{harness.app_state.get_state_string()}")
            continue

        # ── Agent response ──
        try:
            if stream:
                print(f"\n🤖 > ", end="", flush=True)
                for chunk in harness.get_streaming_response(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = harness.get_response(user_input)
                print(f"\n🤖 > {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if harness.debug_mode:
                import traceback
                traceback.print_exc()
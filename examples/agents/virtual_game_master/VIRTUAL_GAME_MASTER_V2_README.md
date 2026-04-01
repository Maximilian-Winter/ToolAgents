# VirtualGameMaster2 — Unified Agent Harness

A fully generic agent harness that combines navigable knowledge, structured state, self-editable memory, and TTL-based conversation management into a single system. Works for any domain: tabletop RPGs, studio management, research assistants, customer support — anything that benefits from an agent that can navigate a knowledge space and maintain persistent context.

## Architecture

```
VirtualGameMaster2
├── HarnessConfig              Configuration from .env, JSON, or dict
├── PromptComposer             Modular system prompt, re-rendered each turn
│   ├── instructions           Static or template-based (position 0)
│   ├── app_state              Structured XML state (position 5, optional)
│   ├── core_memory            Agent-editable key-value blocks (position 10)
│   ├── location               NavigableMemory context (position 15)
│   ├── nav_history            Recently visited locations (position 20)
│   └── metadata               Turn count, message stats (position 25)
├── SmartMessageManager        TTL-based message windowing
│   ├── Pinned messages        System rules, never expire
│   ├── TTL messages           User/assistant messages, archive after N turns
│   └── Location messages      Departed location context, rolling window
├── NavigableMemory            Hierarchical knowledge navigation
│   ├── StorageBackend         InMemory, SQLite, or Agora KB
│   ├── Auto-context loading   Navigate → content appears in prompt
│   ├── Departure callback     Old location → SmartMessage with TTL
│   └── Agent tools            Navigate, list, search, read, write, append
├── ContextAppState            Structured XML state (optional)
├── CoreMemory                 Agent-editable runtime notes
├── Tool Registry              Navigation + memory + state + custom tools
└── Save/Load                  Full session persistence to JSON
```

## How It Works

Each turn follows this flow:

1. **Tick** — `SmartMessageManager` advances all message lifecycles. Messages past their TTL are archived or removed. Expired location contexts fade naturally.

2. **User message** — Added to the message manager with a configurable TTL. After that many turns, it archives (stays searchable, leaves active context).

3. **Compile system prompt** — `PromptComposer` calls each module's `content_fn`. The XML game state re-renders. The NavigableMemory builds the current location context. Core memory blocks re-render. Metadata updates with turn count and message stats.

4. **LLM call** — The compiled system prompt + all active messages go to the agent. The agent responds and optionally calls tools.

5. **Tool execution** — Navigation tools change the agent's position in the knowledge space. Memory tools edit core memory blocks. State tools merge XML fragments. Write tools create new documents.

6. **Response** — Assistant message added with TTL. Tool-call messages added with TTL. Everything ages naturally.

When the agent navigates to a new location:

```
Agent calls navigate("world/locations/candlekeep.md")
  → NavigableMemory loads the document from the backend
  → Fires on_depart callback for the old location
    → If summarize_on_depart is True:
        Makes a separate LLM call to summarize what happened
        Appends the summary to the old location's document
        Injects the summary as a SmartMessage with TTL
    → Otherwise:
        Injects a raw content snippet as a SmartMessage with TTL
  → Updates internal location state
  → Next turn, PromptComposer renders the new location context
```

## Installation

```bash
pip install ToolAgents pydantic python-dotenv
```

For SQLite backend (persistent storage with full-text search):
```bash
# No extra dependencies — uses Python's built-in sqlite3
```

For Agora KB backend (full platform with REST API):
```bash
pip install httpx
```

## Quick Start

### Minimal Example

```python
from ToolAgents.agent_memory.navigable_memory import NavigableMemory, InMemoryBackend
from virtual_game_master_v2 import VirtualGameMaster2, HarnessConfig, run_cli

# Create backend and seed some knowledge
backend = InMemoryBackend()
nav = NavigableMemory(backend=backend)
nav.write("docs/overview.md", "Project Overview", "# My Project\n\nThis is a demo.")
nav.write("docs/api.md", "API Design", "# API\n\nREST endpoints for the service.")

# Configure
config = HarnessConfig()
config.API_TYPE = "groq"
config.API_KEY = "your-groq-key"
config.MODEL = "llama-3.3-70b-versatile"

# Create harness and run
harness = VirtualGameMaster2(
    config=config,
    nav_memory=nav,
    initial_core_memory={"persona": "I am a helpful project assistant."},
    pinned_messages=["[SYSTEM] Navigate to relevant docs before answering."],
)

run_cli(harness, banner="Project Assistant")
```

### RPG with YAML Game Starter

```python
from ToolAgents.agent_memory.navigable_memory import NavigableMemory, InMemoryBackend
from ToolAgents.agent_memory.context_app_state import ContextAppState
from virtual_game_master_v2 import (
    VirtualGameMaster2, HarnessConfig, seed_from_yaml, run_cli,
)

# Backend + seed from YAML
backend = InMemoryBackend()
nav = NavigableMemory(backend=backend, context_window=3)
seed_from_yaml(nav, yaml_file="game_starters/rpg_candlekeep.yaml")

# Structured game state (XML, LLM-updatable)
app_state = ContextAppState("game_starters/rpg_candlekeep.yaml")

# Config
config = HarnessConfig()
config.API_TYPE = "openrouter"
config.API_KEY = "your-key"
config.API_URL = "https://openrouter.ai/api/v1"
config.MODEL = "openai/gpt-4o-mini"
config.TEMPERATURE = 0.7

# Harness
harness = VirtualGameMaster2(
    config=config,
    nav_memory=nav,
    app_state=app_state,
    summarize_on_depart=True,
    initial_core_memory={
        "persona": "I am a Game Master for a D&D campaign set in Candlekeep.",
        "priorities": "Focus on narrative, track party state, use tools for mechanics.",
    },
    pinned_messages=[
        "[SYSTEM] Always begin responses with 'Current Location: <name>'. "
        "Navigate to relevant knowledge before answering. "
        "Use update_app_state for mechanical changes."
    ],
)

# Navigate to starting location
nav.navigate("world/locations/current-location.md")

run_cli(harness, banner="⚔️ Candlekeep Campaign")
```

### Studio Management (Obsidian Forge Style)

```python
from ToolAgents.agent_memory.navigable_memory import NavigableMemory, InMemoryBackend
from virtual_game_master_v2 import VirtualGameMaster2, HarnessConfig, run_cli

backend = InMemoryBackend()
nav = NavigableMemory(backend=backend, context_window=3)

# Seed studio knowledge (your own seeding function)
seed_studio_knowledge(nav)  # writes 48 documents about projects, people, etc.

config = HarnessConfig()
config.API_TYPE = "openrouter"
config.API_KEY = "your-key"
config.API_URL = "https://openrouter.ai/api/v1"
config.MODEL = "xiaomi/mimo-v2-pro"
config.TEMPERATURE = 0.35

harness = VirtualGameMaster2(
    config=config,
    nav_memory=nav,
    initial_core_memory={
        "persona": "I am Forge, studio manager assistant for Obsidian Forge Studios.",
        "priorities": "Key dates: Ashenmoor EA Aug 2026, Drift Protocol E3 Jun 2026.",
    },
    pinned_messages=[
        "[SYSTEM] Navigate to relevant knowledge before answering project questions. "
        "Cross-reference people profiles when names come up. "
        "Highlight risks and blockers proactively."
    ],
)

nav.navigate("studio/overview.md")
run_cli(harness, banner="🔨 Forge — Studio Manager")
```

### Persistent Storage with SQLite

```python
from ToolAgents.agent_memory.navigable_memory import NavigableMemory
from ToolAgents.agent_memory.sqlite_backend import SQLiteBackend
from virtual_game_master_v2 import VirtualGameMaster2, HarnessConfig, seed_from_yaml, run_cli

# SQLite backend — persists across restarts
backend = SQLiteBackend("game_world.db")
nav = NavigableMemory(backend=backend, context_window=3)

# Only seed if the database is empty
if backend.count() == 0:
    seed_from_yaml(nav, yaml_file="game_starters/rpg_candlekeep.yaml")

config = HarnessConfig.from_env()  # load from .env file
harness = VirtualGameMaster2(config=config, nav_memory=nav)
run_cli(harness, banner="⚔️ Campaign (SQLite)")
```

### Agora KB Backend

```python
from ToolAgents.agent_memory.navigable_memory import NavigableMemory
from ToolAgents.agent_memory.agora_backend import AgoraBackend
from virtual_game_master_v2 import VirtualGameMaster2, HarnessConfig, run_cli

# Agora KB — full platform with REST API, dashboard, and multi-agent support
backend = AgoraBackend(
    base_url="http://127.0.0.1:8321",
    project_slug="game-world",
    author="game-master",
)
nav = NavigableMemory(backend=backend, context_window=3)

config = HarnessConfig.from_json("config.json")
harness = VirtualGameMaster2(config=config, nav_memory=nav)
run_cli(harness, banner="⚔️ Campaign (Agora KB)")
```

## Configuration

### HarnessConfig

Configuration can be loaded from `.env` files, JSON files, or constructed programmatically.

```python
# From .env file
config = HarnessConfig.from_env(".env")

# From JSON file
config = HarnessConfig.from_json("config.json")

# From dict
config = HarnessConfig.from_dict({"API_TYPE": "groq", "MODEL": "llama-3.3-70b-versatile"})

# Programmatic
config = HarnessConfig()
config.API_TYPE = "openai"
config.API_KEY = "sk-..."
config.MODEL = "gpt-4o-mini"
```

### Configuration Fields

#### Provider

| Field | Default | Description |
|-------|---------|-------------|
| `API_TYPE` | `"openai"` | Provider type: `openai`, `openrouter`, `anthropic`, `groq`, `mistral`, `local`, `vllm`, `llamacpp` |
| `API_KEY` | `None` | API key for the provider |
| `API_URL` | `""` | Base URL override (for OpenRouter, local servers, etc.) |
| `MODEL` | `""` | Model identifier |

#### Sampling

| Field | Default | Description |
|-------|---------|-------------|
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `1.0` | Nucleus sampling threshold |
| `MAX_TOKENS` | `4096` | Maximum tokens per response |

#### Prompt

| Field | Default | Description |
|-------|---------|-------------|
| `SYSTEM_MESSAGE_FILE` | `""` | Path to a MessageTemplate file for system instructions |
| `INITIAL_STATE_FILE` | `""` | Path to a YAML/XML file for initial ContextAppState |

#### Message Management

| Field | Default | Description |
|-------|---------|-------------|
| `USER_MSG_TTL` | `12` | Turns before user messages archive |
| `ASSISTANT_MSG_TTL` | `12` | Turns before assistant messages archive |
| `LOCATION_MSG_TTL` | `12` | Turns before departed location messages archive |

#### Navigation

| Field | Default | Description |
|-------|---------|-------------|
| `CONTEXT_WINDOW` | `3` | Number of recent locations in NavigableMemory history |
| `INCLUDE_SIBLINGS` | `True` | Show sibling documents in location context |
| `INCLUDE_PARENT` | `True` | Show parent overview in location context |

#### Persistence

| Field | Default | Description |
|-------|---------|-------------|
| `SAVE_FOLDER` | `"saves"` | Directory for session save files |

#### Commands

| Field | Default | Description |
|-------|---------|-------------|
| `COMMAND_PREFIX` | `"/"` | Prefix for meta-commands in the CLI |

### Example .env File

```env
API_TYPE=openrouter
API_KEY=sk-or-v1-your-key
API_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-4o-mini
TEMPERATURE=0.7
TOP_P=1.0
MAX_TOKENS=4096
USER_MSG_TTL=12
ASSISTANT_MSG_TTL=12
LOCATION_MSG_TTL=12
SAVE_FOLDER=saves/my_campaign
SYSTEM_MESSAGE_FILE=prompts/gm_template.txt
```

### Example config.json

```json
{
  "API_TYPE": "groq",
  "API_KEY": "gsk_your-key",
  "MODEL": "llama-3.3-70b-versatile",
  "TEMPERATURE": 0.35,
  "USER_MSG_TTL": 16,
  "SAVE_FOLDER": "saves/studio"
}
```

## YAML Game Starter Seeding

The `seed_from_yaml()` function parses a YAML game starter file and creates a navigable knowledge space from its contents. It handles both deeply nested YAMLs (like the Candlekeep starter with `game_world_information`) and flat ones (like a simple setting with characters and quests).

### Supported Fields

| YAML Field | Documents Created | Path |
|------------|-------------------|------|
| `setting` | World overview | `world/overview.md` |
| `story_summary` | Merged into overview | `world/overview.md` |
| `world_state` | Merged into overview | `world/overview.md` |
| `time_and_calendar` | Merged into overview | `world/overview.md` |
| `location` | Starting location | `world/locations/current-location.md` |
| `game_world_information` | One doc per entry | `world/locations/<slug>.md` |
| `player_character` | Player doc | `world/characters/player.md` |
| `character_details` | One doc per character | `world/characters/<slug>.md` |
| `companions` | Companions list | `world/characters/companions.md` |
| `key_npcs` | NPCs doc or individual docs | `world/npcs/key-npcs.md` or `world/npcs/<slug>.md` |
| `factions` | Factions overview or individual docs | `world/factions/overview.md` or `world/factions/<slug>.md` |
| `active_quests` | Quests list | `world/quests/active.md` |
| `important_events` | Events list | `world/events/history.md` |
| `inventory` | Per-character inventory | `world/party/inventory.md` |
| `special_items` | Detailed item docs | `world/party/special-items.md` |
| `relationships` | Relationships list | `world/party/relationships.md` |

### Usage

```python
from ToolAgents.agent_memory.navigable_memory import NavigableMemory, InMemoryBackend
from virtual_game_master_v2 import seed_from_yaml

backend = InMemoryBackend()
nav = NavigableMemory(backend=backend)

# From file
seed_from_yaml(nav, yaml_file="game_starters/rpg_candlekeep.yaml")

# From dict
seed_from_yaml(nav, yaml_data={"setting": "A dark forest...", "location": "The clearing"})

# Custom path prefix
seed_from_yaml(nav, yaml_file="starter.yaml", world_prefix="campaign/forgotten-realms")

# Quiet mode
seed_from_yaml(nav, yaml_file="starter.yaml", verbose=False)
```

### Example: Candlekeep YAML → 16 Documents

```
world/
  overview.md                          ← setting + story_summary + calendar
  locations/
    current-location.md                ← "Secret underground dock beneath Candlekeep"
    candlekeep.md                      ← from game_world_information.Candlekeep
    arboreal-voyager.md                ← from game_world_information.Arboreal Voyager
    sword-coast.md                     ← from game_world_information.Sword Coast
  characters/
    player.md                          ← Elysia Thunderscribe description
    elysia-thunderscribe.md            ← detailed stats from character_details
    lyra-flameheart.md                 ← detailed stats from character_details
    companions.md                      ← Lyra + Zephyr descriptions
  npcs/
    key-npcs.md                        ← Keeper of Tomes, The Treant, Miirym, etc.
  factions/
    overview.md                        ← Candlekeep Avowed, Harpers, Zhentarim, etc.
  quests/
    active.md                          ← Activate the Arboreal Voyager, etc.
  events/
    history.md                         ← Discovery of the Arboreal Voyager, etc.
  party/
    inventory.md                       ← per-character inventory
    special-items.md                   ← Ring of Whispered Thoughts (with properties)
    relationships.md                   ← Elysia and Lyra are girlfriends, etc.
```

### Custom Fields

Any YAML field not explicitly handled is ignored by the seeder. You can add custom fields to your YAML and process them in your own seeding code before or after calling `seed_from_yaml()`.

### Emergent World-Building

The `WriteDocument` and `AppendToDocument` tools are automatically available to the agent via `NavigableMemory.create_tools()`. This means the agent can create new locations, NPCs, and lore documents on the fly as the story unfolds. The seeded YAML provides the foundation; the agent builds on top of it.

## System Components

### PromptComposer

The system prompt is assembled from ordered modules, each with a fixed position. Some modules have static content; others have a `content_fn` that re-renders every turn.

| Module | Position | Type | Source |
|--------|----------|------|--------|
| `instructions` | 0 | Static or template | System message file or default |
| `app_state` | 5 | Dynamic (optional) | `ContextAppState.get_state_string()` |
| `core_memory` | 10 | Dynamic | `CoreMemory.build_context()` |
| `location` | 15 | Dynamic | `NavigableMemory.build_context()` |
| `nav_history` | 20 | Dynamic | `NavigableMemory.build_history_context()` |
| `metadata` | 25 | Dynamic | Turn count, message stats, current location |

The instructions module supports `MessageTemplate` with `{placeholder}` substitution. If a `ContextAppState` is provided, its `template_fields` are passed to the template each turn, so the system prompt always reflects the current game state.

### SmartMessageManager

Messages have lifecycle policies that determine how long they stay in active context:

| Message Type | TTL | On Expire | Purpose |
|-------------|-----|-----------|---------|
| Pinned system | ∞ | Never | Core rules that always apply |
| User messages | Configurable (default 12) | Archive | Player input |
| Assistant messages | Configurable (default 12) | Archive | Agent responses |
| Tool messages | Same as assistant | Archive | Tool call/result pairs |
| Location departures | Configurable (default 12) | Archive | Rolling window of past locations |
| Ephemeral context | 1-3 | Remove | One-shot injections (opening scenes, notifications) |

Archived messages are never deleted — they remain searchable via the `ArchiveSearch` tool. The TTL values are configurable through `HarnessConfig`.

### NavigableMemory

A hierarchical knowledge space that the agent navigates using tools. When the agent moves to a new location, the document's content loads automatically into the system prompt via `PromptComposer`.

The system is backend-agnostic. Three backends are available:

| Backend | Storage | Search | Persistence | Use Case |
|---------|---------|--------|-------------|----------|
| `InMemoryBackend` | Dict | Substring match | None (lost on restart) | Testing, lightweight demos |
| `SQLiteBackend` | SQLite file | FTS5 with BM25 ranking | Single file | Production single-user |
| `AgoraBackend` | Agora REST API | FTS5 via server | Server-managed | Multi-agent, dashboard |

The agent gets 7 navigation tools automatically from `NavigableMemory.create_tools()`:

| Tool | Purpose |
|------|---------|
| `Navigate` | Move to a document path, loading its content |
| `NavigateUp` | Go to parent directory overview |
| `ListLocations` | Discover documents under a path prefix |
| `SearchKnowledge` | Full-text search across all documents |
| `ReadDocument` | Peek at a document without navigating |
| `WriteDocument` | Create or overwrite a document |
| `AppendToDocument` | Append a timestamped log entry to a document |

### ContextAppState (Optional)

Structured application state stored as an XML tree. Loaded from a YAML file at startup, rendered into the system prompt each turn, and updatable by the agent via XML fragment merging.

The agent calls `UpdateAppState` with an XML fragment, and it merges into the existing tree — matching elements update in place, new elements are appended.

This is optional. If you don't need structured state (e.g., for a studio management assistant), omit it and the harness works without it.

### CoreMemory

Simple key-value memory that the agent can read and edit via tools. Separate from the structured `ContextAppState` — this is for freeform observations, action items, and session notes.

| Tool | Purpose |
|------|---------|
| `CoreMemorySet` | Create or overwrite a named block |
| `CoreMemoryAppend` | Append text to an existing block |
| `CoreMemoryDelete` | Remove a block |
| `ArchiveSearch` | Search archived messages by keyword |

Each block has a configurable character limit (default 600) to encourage the agent to be concise.

### Departure Summarization (Optional)

When `summarize_on_depart=True`, leaving a location triggers an additional LLM call that summarizes what happened there. The summary is appended to the location's document in the backend (via `NavigableMemory.append()`) and injected as a SmartMessage.

This means the knowledge base accumulates history as the agent operates — locations gain event logs, documents grow richer over time.

```python
harness = VirtualGameMaster2(
    config=config,
    nav_memory=nav,
    summarize_on_depart=True,
    summarizer_settings=low_temp_settings,  # optional: lower temperature for factual summaries
)
```

If `summarize_on_depart` is False (default), departed locations are injected as raw content snippets — simpler, faster, no extra LLM calls.

## CLI Commands

The `run_cli()` function provides an interactive terminal interface with built-in commands:

| Command | Description |
|---------|-------------|
| `quit` | Exit (auto-saves session) |
| `/memory` | Show core memory blocks |
| `/location` | Show current knowledge location and recent history |
| `/archive` | Show archived messages (last 10) |
| `/status` | System status: turn count, message counts, location |
| `/tree` | Full knowledge space tree with current position marked |
| `/inject <msg>` | Inject ephemeral context (expires in 3 turns) |
| `/save [name]` | Save session state to JSON (default: "autosave") |
| `/clear` | Clear all messages, restore pinned system message |
| `/state` | Show application state XML (only if ContextAppState is present) |

## Saving and Loading

Session state is saved as a JSON file containing:

- Core memory blocks
- Turn count
- Current navigation location and history
- All active messages with their lifecycle metadata (TTL, turns alive, pinned status)
- Archived message texts
- Application state template fields (if ContextAppState is present)

```python
# Save
filepath = harness.save("my_session")  # → saves/my_session.json

# Load (returns True if successful)
restored = harness.load("my_session")

# Auto-save on quit (handled by run_cli)
```

The knowledge base itself is managed by the backend. `InMemoryBackend` loses data on restart; `SQLiteBackend` and `AgoraBackend` persist independently of the session save.

## Switching LLM Providers

The `create_provider()` factory supports all ToolAgents providers:

```python
# OpenAI
config.API_TYPE = "openai"
config.API_KEY = "sk-..."
config.MODEL = "gpt-4o-mini"

# OpenRouter (access to many models)
config.API_TYPE = "openrouter"
config.API_KEY = "sk-or-v1-..."
config.API_URL = "https://openrouter.ai/api/v1"
config.MODEL = "anthropic/claude-3.5-sonnet"

# Groq (fast inference)
config.API_TYPE = "groq"
config.API_KEY = "gsk_..."
config.MODEL = "llama-3.3-70b-versatile"

# Anthropic (direct)
config.API_TYPE = "anthropic"
config.API_KEY = "sk-ant-..."
config.MODEL = "claude-3-5-sonnet-20241022"

# Mistral
config.API_TYPE = "mistral"
config.API_KEY = "..."
config.MODEL = "mistral-small-latest"

# Local server (vllm, llama.cpp)
config.API_TYPE = "local"
config.API_KEY = "token-abc123"
config.API_URL = "http://127.0.0.1:8080/v1"
config.MODEL = "your-model"
```

## Adding Custom Tools

Pass extra tools via the `extra_tools` parameter. These are added to the tool registry alongside the built-in navigation and memory tools.

```python
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class RollDice(BaseModel):
    """Roll dice for the game. Supports standard notation like 2d6+3."""
    notation: str = Field(..., description="Dice notation (e.g., '2d6', '1d20+5').")

    def run(self) -> str:
        import random, re
        match = re.match(r'(\d+)d(\d+)([+-]\d+)?', self.notation)
        if not match:
            return f"Invalid notation: {self.notation}"
        count, sides, mod = int(match[1]), int(match[2]), int(match[3] or 0)
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls) + mod
        return f"🎲 {self.notation}: {rolls} + {mod} = {total}"

harness = VirtualGameMaster2(
    config=config,
    nav_memory=nav,
    extra_tools=[FunctionTool(RollDice)],
)
```

## Programmatic Usage (No CLI)

The harness works without `run_cli()` for integration into web apps, APIs, or other interfaces:

```python
harness = VirtualGameMaster2(config=config, nav_memory=nav)

# Non-streaming
response = harness.get_response("What are our critical bugs?")
print(response)

# Streaming
for chunk in harness.get_streaming_response("Compare project timelines"):
    print(chunk, end="", flush=True)

# Direct access to subsystems
print(harness.core_memory.build_context())
print(harness.nav_memory.current_title)
print(harness.active_message_count)
print(harness.archive_count)

# Save/load
harness.save("checkpoint")
harness.load("checkpoint")
```

## License

MIT — same as ToolAgents.

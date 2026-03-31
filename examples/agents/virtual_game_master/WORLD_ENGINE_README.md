# Virtual Game Master — World Engine

A complete Virtual Game Master harness built on [ToolAgents](https://github.com/Maximilian-Winter/ToolAgents) and [Agora](https://github.com/Maximilian-Winter/agora). The GM agent navigates a hierarchical world stored in Agora's Knowledge Base, maintains structured game state via XML, and manages conversation context automatically — so you can focus on playing.

## How It Works

```
  YAML Game Starter ──► XMLGameState (structured, LLM-updatable)
                              │
  Custom GM Template ──► PromptComposer (modular system prompt)
                              │
  Agora KB (locations) ──► WorldContextManager (auto-loads on travel)
                              │
  CoreMemory (GM notes) ──► SmartMessageManager (TTL, archival)
                              │
                         ChatToolAgent ──► LLM response + tool calls
```

Each turn, the system:

1. Advances message lifecycles — old messages archive, ephemeral contexts vanish
2. Recompiles the system prompt with live game state, current location, and GM notes
3. Sends everything to the LLM, which responds in character and calls tools
4. Tools update game state, change locations, or record notes — visible next turn

When the GM travels to a new location, the system automatically summarizes what happened at the old location (via a separate LLM call), appends the summary to the old location's KB document, and loads the new location's context into the prompt.

## Prerequisites

- Python 3.11+
- [ToolAgents](https://github.com/Maximilian-Winter/ToolAgents) installed (`pip install ToolAgents`)
- [Agora](https://github.com/Maximilian-Winter/agora) server running (`pip install -e .` then `python -m agora.runner`)
- An LLM provider API key (Groq, OpenRouter, OpenAI, or a local server)

Install additional dependencies:

```bash
pip install httpx pyyaml pydantic python-dotenv
```

## Quick Start

### 1. Start the Agora server

```bash
cd agora
python -m agora.runner
```

The server runs on `http://127.0.0.1:8321` by default.

### 2. Create a project

Open the Agora dashboard at `http://localhost:5173` and create a project called `game-world`, or use the API:

```bash
curl -X POST http://127.0.0.1:8321/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Game World", "description": "Virtual Game Master world data"}'
```

### 3. Set up environment variables

Create a `.env` file:

```env
# LLM Provider (pick one)
GROQ_API_KEY=your-groq-key
# OPENROUTER_API_KEY=your-openrouter-key
# OPENAI_API_KEY=your-openai-key

# Agora (optional, defaults shown)
AGORA_URL=http://127.0.0.1:8321
GAME_PROJECT=game-world
```

### 4. Run the game

```bash
# Basic — uses built-in Han Dynasty defaults and auto-seeds the KB
python example_world_engine.py

# With your YAML game starter
python example_world_engine.py --starter game_starters/rpg_wudang.yaml

# With a custom GM prompt template
python example_world_engine.py --template gm_prompts/wuxia_gm.txt

# Both
python example_world_engine.py --starter game_starters/rpg_wudang.yaml --template gm_prompts/wuxia_gm.txt

# Seed world data only (no game loop)
python example_world_engine.py --seed-only
```

On first run, the system checks if world data exists in the Agora KB. If not, it seeds 6 locations in the Wudang Mountains / Han Dynasty world automatically.

## Playing the Game

Type your actions as a player. The GM responds in character, describes scenes, and uses tools to track state.

```
⚔️  You > I approach Master Chen and bow respectfully. "Master, I seek wisdom
          about the ancient text."

🎭 GM > Current Location: Wudang Monastery — Temple of the Purple Cloud
        
        The old master looks up from his meditation, his piercing eyes studying
        you for a long moment. Incense smoke curls between you like a veil.
        
        "Ah, Li Wei," he says, his voice like stones shifting in a riverbed.
        "You seek the Taixuan Jing. Many have sought it. Few have been ready."
        He gestures toward the scroll library. "Sister Yue has been studying
        the fragments. Perhaps she can tell you what the stars foretold..."
```

### In-Game Commands

| Command | Description |
|---------|-------------|
| `quit` | End the session |
| `/state` | Show the full game state as XML |
| `/party` | Show player character, companions, and inventory |
| `/location` | Show current location path and travel history |
| `/notes` | Show GM notes (plot threads, secrets) |
| `/archive` | Show archived messages from earlier in the conversation |
| `/status` | Show system status (message counts, locations visited) |
| `/save [name]` | Save session to `<name>_game_state.xml` + `<name>_session.json` |
| `/seed` | Re-seed the world data in the Agora KB |

## YAML Game Starters

Game starters define the initial state of the world: characters, setting, quests, inventory, NPCs, and more. They are YAML files that get converted to a structured XML tree at startup.

### Example: `game_starters/rpg_wudang.yaml`

```yaml
setting: >
  Ancient China, late Han Dynasty (circa 200 CE). A time of political turmoil,
  philosophical debate, and the rise of powerful warlords.

player_character: >
  Li Wei (李威): A human male Scholar-Warrior, age 32...

companions:
  - >
    Mei Ling (梅玲): A human female Wu Xia, age 29...

active_quests:
  - Decode an ancient text that may predict the fall of the Han Dynasty
  - Investigate rumors of a powerful artifact in the Wudang Mountains

inventory:
  Li Wei:
    - Jian (double-edged straight sword)
    - Philosophical and martial arts scrolls
  Mei Ling:
    - Pair of butterfly swords
    - Smoke pellets

key_npcs:
  - "Master Chen: The enigmatic head of the Taoist monastery"
  - "Cao Cao: A cunning warlord rising to power"
```

The YAML is converted to XML internally. The GM updates it during play by calling the `update_game_state` tool with XML fragments that merge into the tree:

```xml
<game-state>
  <inventory>
    <Li-Wei>
      <item>Ancient bronze mirror</item>
    </Li-Wei>
  </inventory>
  <active-quests>
    <item>Return the mirror to Master Chen</item>
  </active-quests>
</game-state>
```

### Creating Your Own Game Starter

Any YAML file with key-value pairs works. Common fields: `setting`, `player_character`, `companions`, `active_quests`, `inventory`, `key_npcs`, `world_state`, `factions`, `time_and_calendar`, `relationships`, `character_details`. You can add any custom fields — they all become part of the XML game state tree.

## Custom GM Prompt Templates

The GM's base instructions are loaded from a template with `{placeholder}` substitution. You can customize the GM's personality, rules, and style by providing your own template file.

### Default Template Variables

| Variable | Description |
|----------|-------------|
| `{setting_context}` | Filled from the game state's `setting` field |
| `{genre_notes}` | Style/genre guidance (e.g., "This is a Wuxia setting") |
| `{additional_instructions}` | Any extra rules or instructions |

### Example: `gm_prompts/wuxia_gm.txt`

```
You are a Game Master specializing in Wuxia martial arts stories.
{setting_context}

Your narration should evoke the rhythms of classical Chinese literature.
Combat should be described with flowing, poetic language — leaping between
rooftops, strikes that split the air like calligraphy brushstrokes.

{genre_notes}

RULES:
1. Always begin responses with "Current Location: <location title>"
2. Use update_game_state for ALL mechanical changes
3. Use set_location when the party travels
4. Honor the codes of Wu Xia — justice, loyalty, and honor matter
{additional_instructions}
```

Unused placeholders are silently removed, so templates work even if not all variables are provided.

## World Data in Agora KB

The world is organized as a hierarchy of markdown documents in the Agora Knowledge Base:

```
worlds/
  han-dynasty/
    overview.md                    ← world-level lore, factions, mood
    wudang-mountains/
      overview.md                  ← region description, dangers
      monastery.md                 ← specific location with NPCs
      mountain-path.md             ← travel route with encounters
      hidden-cave.md               ← secret location with artifact
      village.md                   ← settlement with services
```

### How Location Navigation Works

The GM uses tools to navigate this hierarchy:

1. `list_locations("worlds/han-dynasty/wudang-mountains/")` — shows all locations in an area
2. `set_location("worlds/han-dynasty/wudang-mountains/village.md")` — travels there
3. The system automatically loads the location document + its parent overview + nearby locations into the prompt

When leaving a location, the system makes a separate LLM call to summarize what happened, then appends that summary to the location's KB document as an event log. This means the world accumulates history as you play.

### Adding New Locations

Add documents through the Agora dashboard, CLI, or REST API:

```bash
# Via Agora CLI
agora kb write worlds/han-dynasty/wudang-mountains/secret-garden.md \
  --title "The Jade Garden" \
  --tags "location,secret,wudang" \
  --body "# The Jade Garden\n\nA hidden courtyard behind the monastery..."

# Via REST API
curl -X PUT http://127.0.0.1:8321/api/projects/game-world/kb/documents \
  -H "Content-Type: application/json" \
  -d '{
    "path": "worlds/han-dynasty/wudang-mountains/secret-garden.md",
    "title": "The Jade Garden",
    "body": "# The Jade Garden\n\nA hidden courtyard...",
    "tags": "location,secret,wudang"
  }'
```

New locations are immediately discoverable by the GM via `list_locations` and `search_world`.

## GM Tools

The Game Master LLM has access to these tools:

| Tool | Purpose |
|------|---------|
| `set_location` | Travel to a new location (triggers summarize → update KB → load new context) |
| `list_locations` | Discover available locations under a path prefix |
| `search_world` | Full-text search across all KB documents |
| `read_document` | Read a specific document, optionally extracting one section |
| `update_game_state` | Merge an XML fragment into the game state tree |
| `add_gm_note` | Save a freeform note to persistent memory |
| `search_gm_notes` | Search through GM notes by keyword |

## System Components

### PromptComposer

Modular system prompt with named modules at ordered positions. Each module can be static text or a dynamic function that re-renders every turn.

| Module | Position | Type | Content |
|--------|----------|------|---------|
| `gm_instructions` | 0 | Static | GM prompt (from template) |
| `game_state` | 5 | Dynamic | Full XML game state |
| `location_context` | 10 | Swapped | Current location + parent + nearby |
| `gm_notes` | 15 | Dynamic | Freeform GM notes |

### SmartMessageManager

Messages with lifecycle policies:

- **Pinned** — system rules, never expire
- **TTL=12** — user and assistant messages archive after 12 turns
- **TTL=2, REMOVE** — ephemeral context (opening scene prompt) disappears after 2 turns
- **Archived** — expired messages remain searchable via `search_gm_notes` or manual `/archive` command

### XMLGameState

Structured game state as an XML tree. Loaded from YAML at startup, updated by the GM via XML fragment merging. The merge is recursive — matching elements update in place, new elements append.

### CoreMemory

Simple key-value store for freeform GM notes. Separate from the structured game state — used for plot threads, NPC secrets, and session observations that don't fit the XML schema.

## Saving and Loading

Use `/save` during play to snapshot the session:

```
⚔️  You > /save mysession
  💾 Saved: mysession_game_state.xml + mysession_session.json
```

This produces two files: the full XML game state and a JSON file with location history, GM notes, and archived messages.

## Switching LLM Providers

Edit the provider section in `main()`:

```python
# Groq (fast, free tier available)
api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# OpenRouter (many models)
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
)

# Local server (vllm, llama.cpp with --jinja flag)
api = OpenAIChatAPI(
    api_key="token-abc123",
    base_url="http://127.0.0.1:8080/v1",
    model="your-model-here",
)
```

## License

MIT — same as ToolAgents.

# Folder-Based Extension System Design

## Problem

The ToolAgents harness needs an extensible way to load capabilities from folder structures — starting with Agent Skills (SKILL.md-based knowledge injection) and later expanding to other folder types (tool folders, prompt folders, etc.). The system should follow the Agent Skills specification's progressive disclosure pattern: lightweight discovery at startup, full content loading on demand.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Protocol-based handlers + ExtensionManager | Clean contract, testable, extensible to future handler types |
| Manager placement | Peer module alongside harness | Keeps harness focused on conversation lifecycle; manager is independently testable |
| Activation mechanism | Dedicated tool + slash commands | Tool gives model reliable structured activation; slash commands are ergonomic for users |
| Tool folder handler | Deferred (YAGNI) | Build the abstraction with skills, validate the pattern before adding more handler types |
| Context protection | Configurable per-skill, default protected | Forward-thinking without complexity; uses existing `pin_message()` API |
| Directory convention | Separate dirs per type | `.agents/skills/`, `.agents/tools/`, etc. — clean separation, each with its own handler |

## Module Layout

```
src/ToolAgents/extensions/
├── __init__.py              # Public API exports, lazy imports
├── models.py                # ExtensionEntry, ActivationResult, ExtensionScanPath
├── handler.py               # FolderHandler protocol
├── manager.py               # ExtensionManager
└── skill_handler.py         # SkillFolderHandler
```

### Dependencies

- `extensions/` depends on: `ToolAgents.function_tool` (for `FunctionTool`)
- `extensions/` does NOT depend on: harness, context_manager, agents, providers
- Harness depends on `extensions/` only optionally (when `extension_manager` is passed)

## Core Abstractions

### ExtensionEntry

Lightweight metadata captured at discovery time.

```python
@dataclass
class ExtensionEntry:
    name: str                    # from frontmatter (e.g., "pdf-processing")
    description: str             # from frontmatter
    handler_type: str            # which handler owns this (e.g., "skills")
    path: Path                   # absolute path to the marker file
    scope: str                   # "project" | "user" | "builtin" — from the scan path
    priority: int                # from the scan path — used for collision resolution
    metadata: dict               # additional frontmatter fields (license, compatibility, etc.)
```

### ActivationResult

What gets returned when an extension is activated.

```python
@dataclass
class ActivationResult:
    content: str                 # full instructions/content to inject
    pin_in_context: bool = True  # whether to protect from context trimming
    tools: list[FunctionTool] | None = None   # optional tools to register on activation
    resources: list[str] | None = None        # bundled file paths (for model awareness)
```

### ExtensionScanPath

Configures where to scan for extensions.

```python
@dataclass
class ExtensionScanPath:
    path: Path                   # directory to scan
    scope: str = "project"       # "project" | "user" | "builtin"
    priority: int = 0            # higher priority wins on name collision
```

Convention: project scope uses `priority=10`, user scope uses `priority=0`. Project-level extensions override user-level ones on name collision.

### FolderHandler Protocol

```python
class FolderHandler(Protocol):
    name: str                    # handler type identifier (e.g., "skills")
    marker_file: str             # file to look for (e.g., "SKILL.md")

    def discover(self, path: Path) -> ExtensionEntry | None:
        """Parse a folder into an ExtensionEntry. Return None if invalid/malformed."""
        ...

    def build_catalog(self, entries: list[ExtensionEntry]) -> str:
        """Build a system prompt section listing available extensions of this type."""
        ...

    def activate(self, entry: ExtensionEntry) -> ActivationResult:
        """Load full extension content for injection into context."""
        ...

    def get_tools(self, manager: "ExtensionManager") -> list[FunctionTool]:
        """Return tools this handler wants registered (e.g., activate_skill).
        Receives the manager so tools can call manager.activate() via closure."""
        ...
```

## ExtensionManager

### Responsibilities

- Holds registered `FolderHandler` instances (keyed by `handler.name`)
- Scans configured `ExtensionScanPath` directories
- Maintains the extension catalog: `dict[str, ExtensionEntry]`
- Handles name collisions using priority (higher wins, warn on shadow)
- Routes slash commands to the correct handler
- Tracks active extensions for deduplication
- Provides: system prompt catalog text, tools to register, activation results

### Does NOT

- Touch the harness directly
- Manage messages or context
- Own LLM interaction

### Interface

```python
class ExtensionManager:
    def __init__(self) -> None: ...

    # --- Setup ---
    def register_handler(self, handler: FolderHandler) -> None:
        """Register a folder handler. Calls handler.get_tools(self) to collect tools."""

    def add_scan_path(self, scan_path: ExtensionScanPath) -> None:
        """Add a directory to scan for extensions."""

    # --- Discovery ---
    def discover(self) -> dict[str, list[ExtensionEntry]]:
        """Scan all paths with all handlers. Returns entries grouped by handler type.
        For each scan path, iterates subdirectories looking for marker files.
        Resolves name collisions via priority. Logs warnings on shadowed entries."""

    # --- Catalog ---
    def build_catalog(self) -> str:
        """Build combined system prompt section from all handlers.
        Delegates to each handler's build_catalog() with its entries.
        Returns empty string if no extensions discovered."""

    # --- Activation ---
    def activate(self, name: str) -> ActivationResult | None:
        """Activate by name. Returns None if not found.
        Returns cached result if already active (deduplication)."""

    def deactivate(self, name: str) -> None:
        """Deactivate an extension. Tracking-only: removes from active set so the
        extension can be re-activated if needed. Does NOT unpin messages or remove
        content from conversation — that content is already in the message history
        and managed by the context manager."""

    def is_active(self, name: str) -> bool: ...

    # --- Slash Commands ---
    def try_handle_command(self, command: str) -> ActivationResult | None:
        """If command matches an extension name, activate it.
        Returns None if not a recognized extension command."""

    # --- Tools ---
    def get_tools(self) -> list[FunctionTool]:
        """Collect tools from all registered handlers.
        Calls handler.get_tools(self) for each handler, passing self as manager."""

    # --- Introspection ---
    @property
    def entries(self) -> dict[str, ExtensionEntry]: ...

    @property
    def active_entries(self) -> dict[str, ExtensionEntry]: ...
```

### Discovery Algorithm

```
for each scan_path in sorted(scan_paths, by=priority descending):
    for each subdirectory in scan_path.path:
        skip directories in SKIP_DIRS
        for each registered handler:
            if subdirectory contains handler.marker_file:
                entry = handler.discover(subdirectory / handler.marker_file)
                entry.scope = scan_path.scope
                entry.priority = scan_path.priority
                if entry is not None:
                    if entry.name already in catalog:
                        if existing priority >= current priority:
                            log warning ("skill '{name}' from {scope} shadowed by {existing.scope}"), skip
                        else:
                            log warning, replace
                    else:
                        add to catalog
```

**Default skip directories**: `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `.tox`, `dist`, `build`, `.eggs`

**Name collision tiebreaker**: On equal priority, first-discovered wins (scan paths are iterated in registration order within the same priority level). A warning is logged so the user knows about the collision.

Max scan depth: 2 levels (skill dirs are immediate children of the scan path).

## SkillFolderHandler

Implements the Agent Skills specification for `SKILL.md` files.

### Discovery

1. Read `SKILL.md` file
2. Split YAML frontmatter from Markdown body
3. Extract required fields: `name`, `description`
4. Extract optional fields: `license`, `compatibility`, `metadata`, `allowed-tools`
5. Map custom field `pin_in_context` (defaults to `True`)
6. Lenient validation:
   - Missing description → skip (log error)
   - Unparseable YAML → skip (log error)
   - Name mismatch with directory → warn, load anyway
   - Name too long → warn, load anyway
7. Handle malformed YAML: retry with quoted values if initial parse fails
8. YAML parsing: use `pyyaml` (`yaml.safe_load`) with manual `---` delimiter splitting (no additional dependency — pyyaml is already a transitive dependency via pydantic)

### Catalog Generation

Returns XML-formatted catalog with behavioral instructions:

```xml
The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, call the activate_skill tool
with the skill's name to load its full instructions.
Users can also activate skills directly with /skill-name commands.

<available_skills>
  <skill name="pdf-processing" location="/path/to/SKILL.md">
    Extract PDF text, fill forms, merge files. Use when handling PDFs.
  </skill>
</available_skills>
```

Returns empty string if no skills discovered.

### Activation

1. Read full `SKILL.md` body content (after frontmatter)
2. Scan skill directory for `scripts/`, `references/`, `assets/` subdirectories
3. Enumerate bundled resource files (cap at 50 files, alphabetically sorted, log warning if truncated)
4. Wrap in structured tags:

```xml
<skill_content name="pdf-processing">
[SKILL.md body content]

Skill directory: /absolute/path/to/pdf-processing
Relative paths in this skill resolve against the skill directory.

<skill_resources>
  <file>scripts/extract.py</file>
  <file>references/pdf-spec.md</file>
</skill_resources>
</skill_content>
```

5. Return `ActivationResult` with `pin_in_context` from metadata (default `True`)

### Tools

Returns a single `activate_skill` tool:

```python
def activate_skill(skill_name: str) -> str:
    """Activate a skill by name to load its full instructions into context.

    Args:
        skill_name: Name of the skill to activate.
    """
```

- The tool's schema constrains `skill_name` to an enum of discovered skill names
- If no skills discovered, returns empty list (tool is not registered)
- The tool function calls `manager.activate(skill_name)` and returns the content string

## Harness Integration

### Factory Changes

`create_harness()` and `create_async_harness()` gain an optional parameter:

```python
def create_harness(
    provider, system_prompt, ...,
    extension_manager: ExtensionManager | None = None,
) -> AgentHarness:
```

If `extension_manager` is provided:
1. Append `manager.build_catalog()` to the system prompt
2. Register `manager.get_tools()` with the harness
3. Store as `harness.extension_manager`

### Convenience Factory

```python
def create_harness_with_extensions(
    provider, system_prompt, ...,
    skill_paths: list[Path | str] | None = None,
    scan_defaults: bool = True,
) -> AgentHarness:
```

Sets up `ExtensionManager` + `SkillFolderHandler` + default scan paths, then delegates to `create_harness()`.

### Context Pinning

The `activate_skill` tool function (inside SkillFolderHandler) calls `manager.activate(name)` which returns an `ActivationResult`. The tool function stores this result on the manager via `manager._pending_activations[name] = result` before returning `result.content` as the tool result string.

After each turn, in the harness `_process_agent_buffer()` method (which already walks the agent's `last_messages_buffer`), the harness checks: for each tool-result message, if the preceding tool call was `activate_skill`, look up the corresponding `ActivationResult` from `manager._pending_activations`. If `result.pin_in_context` is `True`, call `context_manager.pin_message(msg.id)`. If `result.tools` is non-empty, call `harness.add_tools(result.tools)`. Clear `_pending_activations` after processing.

This approach keeps the `ActivationResult` accessible to the harness without modifying `FunctionTool`'s return type.

### Slash Command Interception

In the harness `run()` loop (and in the IOHandler / TUI bridge):

1. Check if user input starts with `/`
2. Extract command name (strip the `/`)
3. Call `manager.try_handle_command(command)`
4. If `None`: treat as normal user input, proceed to `chat()` / `chat_stream()`
5. If `ActivationResult` returned:
   a. Create a system message: `ChatMessage.create_system_message(result.content)`
   b. Append to `self._messages`
   c. If `result.pin_in_context`: call `context_manager.pin_message(msg.id)`
   d. If `result.tools`: call `harness.add_tools(result.tools)`
   e. Display confirmation via `io_handler.on_text(f"Skill '{command}' activated.")`
   f. Do NOT increment turn counter, do NOT emit TURN_START/TURN_END events
   g. Do NOT call the LLM — the skill content is now in context for the next user message

### Async Considerations

All file I/O in `discover()` and `activate()` is synchronous. This is acceptable because:
- Discovery happens once at startup (before the event loop is busy)
- Activation reads small files (SKILL.md is recommended < 500 lines)
- The `AsyncAgentHarness` calls `activate_skill` inside the agent's tool-call loop, which already handles sync tool functions via `asyncio.to_thread`

No async variants of the `FolderHandler` protocol are needed.

## Testing Strategy

1. **Unit tests for models**: Construct `ExtensionEntry`, `ActivationResult`, verify fields
2. **Unit tests for SkillFolderHandler**: Create temp directories with SKILL.md files, test discover/activate/catalog
3. **Unit tests for ExtensionManager**: Register handlers, add paths, test discover/activate/deduplication/collision
4. **YAML edge cases**: Malformed YAML, missing fields, lenient parsing
5. **Integration test**: Wire manager to harness via factory, verify catalog in system prompt and tools registered
6. **Slash command test**: Verify `/skill-name` routes correctly
7. **Pinning test**: Verify activated skill messages get pinned in context manager

# Folder-Based Extension System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a folder-based extension system to ToolAgents, starting with Agent Skills (SKILL.md) support, using a `FolderHandler` protocol + `ExtensionManager` architecture.

**Architecture:** An `ExtensionManager` peer module scans configured directories, delegates to registered `FolderHandler` implementations (starting with `SkillFolderHandler`), and provides tools, catalog text, and activation results that the harness consumes. The harness integrates optionally via factory parameters.

**Tech Stack:** Python 3.10+, pydantic, pyyaml (transitive dep), dataclasses, typing.Protocol, logging

**Spec:** `docs/superpowers/specs/2026-03-13-folder-handler-extensions-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/ToolAgents/extensions/__init__.py` | Create | Public API exports with lazy imports |
| `src/ToolAgents/extensions/models.py` | Create | `ExtensionEntry`, `ActivationResult`, `ExtensionScanPath` dataclasses |
| `src/ToolAgents/extensions/handler.py` | Create | `FolderHandler` protocol |
| `src/ToolAgents/extensions/manager.py` | Create | `ExtensionManager` orchestrator |
| `src/ToolAgents/extensions/skill_handler.py` | Create | `SkillFolderHandler` implementing Agent Skills spec |
| `src/ToolAgents/agent_harness/harness.py` | Modify | Add extension pinning in `_process_agent_buffer`, slash commands in `run()`, `extension_manager` property |
| `src/ToolAgents/agent_harness/async_harness.py` | Modify | Same changes as sync harness |
| `tests/test_extensions.py` | Create | All extension system tests |

---

## Chunk 1: Data Models, Protocol, and Basic Tests

### Task 1: Data Models

**Files:**
- Create: `src/ToolAgents/extensions/models.py`
- Test: `tests/test_extensions.py`

- [ ] **Step 1: Write the failing test for data models**

```python
# tests/test_extensions.py
"""Tests for the ToolAgents extension system."""
import tempfile
from pathlib import Path

from ToolAgents.extensions.models import ExtensionEntry, ActivationResult, ExtensionScanPath


class TestExtensionEntry:
    def test_create_entry(self):
        entry = ExtensionEntry(
            name="test-skill",
            description="A test skill",
            handler_type="skills",
            path=Path("/tmp/skills/test-skill/SKILL.md"),
            scope="project",
            priority=10,
            metadata={"license": "MIT"},
        )
        assert entry.name == "test-skill"
        assert entry.description == "A test skill"
        assert entry.handler_type == "skills"
        assert entry.scope == "project"
        assert entry.priority == 10
        assert entry.metadata["license"] == "MIT"

    def test_entry_default_metadata(self):
        entry = ExtensionEntry(
            name="x", description="y", handler_type="skills",
            path=Path("/tmp/x/SKILL.md"), scope="user", priority=0,
            metadata={},
        )
        assert entry.metadata == {}


class TestActivationResult:
    def test_defaults(self):
        result = ActivationResult(content="# Instructions")
        assert result.pin_in_context is True
        assert result.tools is None
        assert result.resources is None

    def test_custom_values(self):
        result = ActivationResult(
            content="hello",
            pin_in_context=False,
            resources=["scripts/run.py"],
        )
        assert result.pin_in_context is False
        assert result.resources == ["scripts/run.py"]


class TestExtensionScanPath:
    def test_defaults(self):
        sp = ExtensionScanPath(path=Path("/tmp/skills"))
        assert sp.scope == "project"
        assert sp.priority == 0

    def test_custom(self):
        sp = ExtensionScanPath(path=Path("/home/.agents/skills"), scope="user", priority=5)
        assert sp.scope == "user"
        assert sp.priority == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestExtensionEntry -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ToolAgents.extensions'`

- [ ] **Step 3: Create the extensions package and models**

```python
# src/ToolAgents/extensions/__init__.py
# (empty for now — will add lazy imports in Task 3)
```

```python
# src/ToolAgents/extensions/models.py
"""Data models for the extension system."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ToolAgents.function_tool import FunctionTool

logger = logging.getLogger(__name__)


@dataclass
class ExtensionEntry:
    """Lightweight metadata for a discovered extension.

    Captured at discovery time from a marker file's frontmatter.

    Attributes:
        name: Extension identifier from frontmatter (e.g., "pdf-processing").
        description: What the extension does and when to use it.
        handler_type: Which handler owns this entry (e.g., "skills").
        path: Absolute path to the marker file (e.g., SKILL.md).
        scope: Where this was discovered — "project", "user", or "builtin".
        priority: From the scan path — higher wins on name collision.
        metadata: Additional frontmatter fields (license, compatibility, etc.).
    """

    name: str
    description: str
    handler_type: str
    path: Path
    scope: str
    priority: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ActivationResult:
    """What gets returned when an extension is activated.

    Attributes:
        content: Full instructions/content to inject into context.
        pin_in_context: Whether to protect this content from context trimming.
        tools: Optional tools to register with the harness on activation.
        resources: Bundled file paths for model awareness (not eagerly loaded).
    """

    content: str
    pin_in_context: bool = True
    tools: Optional[List["FunctionTool"]] = None
    resources: Optional[List[str]] = None


@dataclass
class ExtensionScanPath:
    """Configures a directory to scan for extensions.

    Attributes:
        path: Directory to scan.
        scope: "project", "user", or "builtin".
        priority: Higher priority wins on name collision (project=10, user=0).
    """

    path: Path
    scope: str = "project"
    priority: int = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestExtensionEntry tests/test_extensions.py::TestActivationResult tests/test_extensions.py::TestExtensionScanPath -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/extensions/__init__.py src/ToolAgents/extensions/models.py tests/test_extensions.py
git commit -m "feat(extensions): add data models — ExtensionEntry, ActivationResult, ExtensionScanPath"
```

---

### Task 2: FolderHandler Protocol

**Files:**
- Create: `src/ToolAgents/extensions/handler.py`
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write the failing test for the protocol**

Append to `tests/test_extensions.py`:

```python
from ToolAgents.extensions.handler import FolderHandler


class TestFolderHandlerProtocol:
    def test_protocol_is_runtime_checkable(self):
        """FolderHandler should be a runtime-checkable Protocol."""
        # A minimal class that satisfies the protocol
        class DummyHandler:
            name = "dummy"
            marker_file = "DUMMY.md"

            def discover(self, path):
                return None

            def build_catalog(self, entries):
                return ""

            def activate(self, entry):
                return ActivationResult(content="")

            def get_tools(self, manager):
                return []

        handler = DummyHandler()
        assert isinstance(handler, FolderHandler)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestFolderHandlerProtocol -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the protocol**

```python
# src/ToolAgents/extensions/handler.py
"""FolderHandler protocol — the contract for folder-based extension handlers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ToolAgents.function_tool import FunctionTool
    from .manager import ExtensionManager
    from .models import ActivationResult, ExtensionEntry


@runtime_checkable
class FolderHandler(Protocol):
    """Protocol for folder-based extension handlers.

    Each handler knows how to discover, catalog, and activate extensions
    from folders containing a specific marker file (e.g., SKILL.md).

    Attributes:
        name: Handler type identifier (e.g., "skills").
        marker_file: Filename to look for in directories (e.g., "SKILL.md").
    """

    name: str
    marker_file: str

    def discover(self, path: Path) -> Optional["ExtensionEntry"]:
        """Parse a folder's marker file into an ExtensionEntry.

        Args:
            path: Absolute path to the marker file.

        Returns:
            An ExtensionEntry if valid, None if invalid/malformed.
        """
        ...

    def build_catalog(self, entries: List["ExtensionEntry"]) -> str:
        """Build a system prompt section listing available extensions.

        Args:
            entries: All discovered entries for this handler type.

        Returns:
            Formatted string for injection into the system prompt.
            Empty string if no entries.
        """
        ...

    def activate(self, entry: "ExtensionEntry") -> "ActivationResult":
        """Load full extension content for injection into context.

        Args:
            entry: The extension to activate.

        Returns:
            ActivationResult with content, pinning preference, and optional tools/resources.
        """
        ...

    def get_tools(self, manager: "ExtensionManager") -> List["FunctionTool"]:
        """Return tools this handler wants registered with the harness.

        Args:
            manager: The ExtensionManager, so tools can call manager.activate() via closure.

        Returns:
            List of FunctionTool instances. Empty list if none.
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestFolderHandlerProtocol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/extensions/handler.py tests/test_extensions.py
git commit -m "feat(extensions): add FolderHandler protocol"
```

---

### Task 3: `__init__.py` with Lazy Imports

**Files:**
- Modify: `src/ToolAgents/extensions/__init__.py`
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write the failing test for public API imports**

Append to `tests/test_extensions.py`:

```python
class TestExtensionsPublicAPI:
    def test_import_models(self):
        from ToolAgents.extensions import ExtensionEntry, ActivationResult, ExtensionScanPath
        assert ExtensionEntry is not None
        assert ActivationResult is not None
        assert ExtensionScanPath is not None

    def test_import_handler(self):
        from ToolAgents.extensions import FolderHandler
        assert FolderHandler is not None

    def test_import_manager(self):
        from ToolAgents.extensions import ExtensionManager
        assert ExtensionManager is not None

    def test_import_skill_handler(self):
        from ToolAgents.extensions import SkillFolderHandler
        assert SkillFolderHandler is not None

    def test_dir_lists_all(self):
        import ToolAgents.extensions as ext
        public = dir(ext)
        for name in ["ExtensionEntry", "ActivationResult", "ExtensionScanPath",
                      "FolderHandler", "ExtensionManager", "SkillFolderHandler"]:
            assert name in public
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestExtensionsPublicAPI -v`
Expected: FAIL (imports not wired yet)

- [ ] **Step 3: Implement lazy imports**

```python
# src/ToolAgents/extensions/__init__.py
"""
ToolAgents Extensions — Folder-based extension system.

A protocol-based framework for loading capabilities from folder structures.
Starts with Agent Skills (SKILL.md) support, extensible to other folder types.

Models:
    ExtensionEntry      — Lightweight metadata for a discovered extension
    ActivationResult    — Content + config returned on activation
    ExtensionScanPath   — Configures a directory to scan

Protocol:
    FolderHandler       — Contract for folder-based extension handlers

Manager:
    ExtensionManager    — Orchestrator: scans, catalogs, activates, routes commands

Handlers:
    SkillFolderHandler  — Agent Skills (SKILL.md) handler

Requires: pyyaml (verify installed; add to project deps if missing)
"""

__all__ = [
    # Models
    "ExtensionEntry",
    "ActivationResult",
    "ExtensionScanPath",
    # Protocol
    "FolderHandler",
    # Manager
    "ExtensionManager",
    # Handlers
    "SkillFolderHandler",
]


def __getattr__(name: str):
    if name in {"ExtensionEntry", "ActivationResult", "ExtensionScanPath"}:
        from .models import ActivationResult, ExtensionEntry, ExtensionScanPath

        return {
            "ExtensionEntry": ExtensionEntry,
            "ActivationResult": ActivationResult,
            "ExtensionScanPath": ExtensionScanPath,
        }[name]

    if name == "FolderHandler":
        from .handler import FolderHandler

        return FolderHandler

    if name == "ExtensionManager":
        from .manager import ExtensionManager

        return ExtensionManager

    if name == "SkillFolderHandler":
        from .skill_handler import SkillFolderHandler

        return SkillFolderHandler

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
```

Note: This will still fail until `manager.py` and `skill_handler.py` exist. Create stub files:

```python
# src/ToolAgents/extensions/manager.py (stub)
"""ExtensionManager — orchestrator for folder-based extensions."""


class ExtensionManager:
    """Placeholder — implemented in Task 4."""
    pass
```

```python
# src/ToolAgents/extensions/skill_handler.py (stub)
"""SkillFolderHandler — Agent Skills (SKILL.md) handler."""


class SkillFolderHandler:
    """Placeholder — implemented in Task 5."""
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestExtensionsPublicAPI -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/extensions/__init__.py src/ToolAgents/extensions/manager.py src/ToolAgents/extensions/skill_handler.py tests/test_extensions.py
git commit -m "feat(extensions): wire up __init__.py with lazy imports and stubs"
```

---

## Chunk 2: ExtensionManager

### Task 4: ExtensionManager Implementation

**Files:**
- Modify: `src/ToolAgents/extensions/manager.py` (replace stub)
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write failing tests for ExtensionManager**

Append to `tests/test_extensions.py`:

```python
import os

from ToolAgents.extensions.manager import ExtensionManager
from ToolAgents.extensions.models import ExtensionEntry, ActivationResult, ExtensionScanPath


class DummySkillHandler:
    """A minimal handler for testing the manager."""
    name = "skills"
    marker_file = "SKILL.md"

    def discover(self, path):
        # Read just the name from the file
        text = path.read_text(encoding="utf-8")
        lines = text.strip().split("\n")
        name = path.parent.name  # default to dir name
        desc = "No description"
        for line in lines:
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                desc = line.split(":", 1)[1].strip()
        if not desc or desc == "No description":
            return None
        return ExtensionEntry(
            name=name,
            description=desc,
            handler_type=self.name,
            path=path,
            scope="",  # filled by manager
            priority=0,  # filled by manager
            metadata={},
        )

    def build_catalog(self, entries):
        if not entries:
            return ""
        lines = ["<skills>"]
        for e in entries:
            lines.append(f'  <skill name="{e.name}">{e.description}</skill>')
        lines.append("</skills>")
        return "\n".join(lines)

    def activate(self, entry):
        content = entry.path.read_text(encoding="utf-8")
        return ActivationResult(content=content)

    def get_tools(self, manager):
        return []


def _create_skill_dir(base_path, name, description="A test skill"):
    """Helper: create a skill directory with a SKILL.md file."""
    skill_dir = base_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nInstructions here.\n")
    return skill_dir


class TestExtensionManagerSetup:
    def test_register_handler(self):
        mgr = ExtensionManager()
        handler = DummySkillHandler()
        mgr.register_handler(handler)
        assert "skills" in mgr._handlers

    def test_add_scan_path(self):
        mgr = ExtensionManager()
        sp = ExtensionScanPath(path=Path("/tmp/test"), scope="project", priority=10)
        mgr.add_scan_path(sp)
        assert len(mgr._scan_paths) == 1


class TestExtensionManagerDiscovery:
    def test_discover_finds_skills(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "Does something useful")
        _create_skill_dir(tmp_path, "other-skill", "Does something else")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path, scope="project", priority=10))
        result = mgr.discover()

        assert "skills" in result
        assert len(result["skills"]) == 2
        names = {e.name for e in result["skills"]}
        assert names == {"my-skill", "other-skill"}

    def test_discover_sets_scope_and_priority(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "Test skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path, scope="user", priority=5))
        result = mgr.discover()

        entry = result["skills"][0]
        assert entry.scope == "user"
        assert entry.priority == 5

    def test_discover_skips_dotgit(self, tmp_path):
        _create_skill_dir(tmp_path / ".git", "hidden-skill", "Should be skipped")
        _create_skill_dir(tmp_path, "visible-skill", "Should be found")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        result = mgr.discover()

        names = {e.name for e in result.get("skills", [])}
        assert "visible-skill" in names
        assert "hidden-skill" not in names

    def test_discover_name_collision_higher_priority_wins(self, tmp_path):
        project_dir = tmp_path / "project_skills"
        user_dir = tmp_path / "user_skills"
        _create_skill_dir(project_dir, "my-skill", "Project version")
        _create_skill_dir(user_dir, "my-skill", "User version")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=project_dir, scope="project", priority=10))
        mgr.add_scan_path(ExtensionScanPath(path=user_dir, scope="user", priority=0))
        result = mgr.discover()

        # Only one entry, from project scope
        skills = result["skills"]
        assert len(skills) == 1
        assert skills[0].scope == "project"

    def test_discover_equal_priority_first_wins(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _create_skill_dir(dir_a, "same-skill", "Version A")
        _create_skill_dir(dir_b, "same-skill", "Version B")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=dir_a, scope="project", priority=10))
        mgr.add_scan_path(ExtensionScanPath(path=dir_b, scope="project", priority=10))
        result = mgr.discover()

        assert len(result["skills"]) == 1
        # First-discovered wins (dir_a was added first, so higher priority scan processed first)
        assert result["skills"][0].description == "Version A"

    def test_discover_empty_directory(self, tmp_path):
        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        result = mgr.discover()

        assert result.get("skills", []) == []

    def test_discover_nonexistent_path(self, tmp_path):
        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path / "does-not-exist"))
        result = mgr.discover()

        assert result.get("skills", []) == []


class TestExtensionManagerCatalog:
    def test_build_catalog(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "Does things")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        catalog = mgr.build_catalog()
        assert "my-skill" in catalog
        assert "Does things" in catalog

    def test_build_catalog_empty(self, tmp_path):
        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        catalog = mgr.build_catalog()
        assert catalog == ""


class TestExtensionManagerActivation:
    def test_activate_returns_result(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        result = mgr.activate("my-skill")
        assert result is not None
        assert "my-skill" in result.content

    def test_activate_unknown_returns_none(self, tmp_path):
        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        assert mgr.activate("nonexistent") is None

    def test_activate_deduplication(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        result1 = mgr.activate("my-skill")
        result2 = mgr.activate("my-skill")
        assert result1 is result2  # same cached object

    def test_is_active(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        assert not mgr.is_active("my-skill")
        mgr.activate("my-skill")
        assert mgr.is_active("my-skill")

    def test_deactivate(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        mgr.activate("my-skill")
        assert mgr.is_active("my-skill")
        mgr.deactivate("my-skill")
        assert not mgr.is_active("my-skill")

    def test_deactivate_allows_reactivation(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        mgr.activate("my-skill")
        mgr.deactivate("my-skill")
        result = mgr.activate("my-skill")
        assert result is not None
        assert mgr.is_active("my-skill")


class TestExtensionManagerSlashCommands:
    def test_try_handle_command_known(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        result = mgr.try_handle_command("my-skill")
        assert result is not None
        assert mgr.is_active("my-skill")

    def test_try_handle_command_unknown(self, tmp_path):
        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        result = mgr.try_handle_command("nonexistent")
        assert result is None


class TestExtensionManagerIntrospection:
    def test_entries_property(self, tmp_path):
        _create_skill_dir(tmp_path, "s1", "Skill 1")
        _create_skill_dir(tmp_path, "s2", "Skill 2")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        assert len(mgr.entries) == 2
        assert "s1" in mgr.entries
        assert "s2" in mgr.entries

    def test_active_entries_property(self, tmp_path):
        _create_skill_dir(tmp_path, "s1", "Skill 1")
        _create_skill_dir(tmp_path, "s2", "Skill 2")

        mgr = ExtensionManager()
        mgr.register_handler(DummySkillHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        assert len(mgr.active_entries) == 0
        mgr.activate("s1")
        assert len(mgr.active_entries) == 1
        assert "s1" in mgr.active_entries
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestExtensionManagerSetup -v`
Expected: FAIL (manager is a stub)

- [ ] **Step 3: Implement ExtensionManager**

Replace `src/ToolAgents/extensions/manager.py` with:

```python
# src/ToolAgents/extensions/manager.py
"""ExtensionManager — orchestrator for folder-based extensions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ToolAgents.function_tool import FunctionTool
    from .handler import FolderHandler

from .models import ActivationResult, ExtensionEntry, ExtensionScanPath

logger = logging.getLogger(__name__)

# Directories to skip during scanning.
SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".eggs",
})


class ExtensionManager:
    """Orchestrator for folder-based extensions.

    Holds registered FolderHandler instances, scans configured directories,
    maintains the extension catalog, and provides tools, catalog text, and
    activation results for harness consumption.

    Usage:
        manager = ExtensionManager()
        manager.register_handler(SkillFolderHandler())
        manager.add_scan_path(ExtensionScanPath(Path(".agents/skills"), scope="project", priority=10))
        manager.discover()

        catalog_text = manager.build_catalog()   # for system prompt
        tools = manager.get_tools()              # for harness tool registration
        result = manager.activate("my-skill")    # on-demand activation
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, "FolderHandler"] = {}
        self._scan_paths: List[ExtensionScanPath] = []
        self._entries: Dict[str, ExtensionEntry] = {}
        self._active: Dict[str, ActivationResult] = {}
        self._pending_activations: Dict[str, ActivationResult] = {}

    # --- Setup ---

    def register_handler(self, handler: "FolderHandler") -> None:
        """Register a folder handler.

        Args:
            handler: A FolderHandler implementation (e.g., SkillFolderHandler).
        """
        self._handlers[handler.name] = handler

    def add_scan_path(self, scan_path: ExtensionScanPath) -> None:
        """Add a directory to scan for extensions.

        Args:
            scan_path: An ExtensionScanPath with path, scope, and priority.
        """
        self._scan_paths.append(scan_path)

    # --- Discovery ---

    def discover(self) -> Dict[str, List[ExtensionEntry]]:
        """Scan all paths with all handlers. Returns entries grouped by handler type.

        Iterates scan paths sorted by priority (descending). Within each path,
        checks subdirectories for handler marker files. Resolves name collisions
        by priority (higher wins); on equal priority, first-discovered wins.

        Returns:
            Dict mapping handler type names to lists of discovered ExtensionEntry objects.
        """
        self._entries.clear()
        self._active.clear()
        self._pending_activations.clear()

        # Sort by priority descending — higher priority scanned first
        sorted_paths = sorted(self._scan_paths, key=lambda sp: sp.priority, reverse=True)

        for scan_path in sorted_paths:
            if not scan_path.path.is_dir():
                logger.debug("Scan path does not exist: %s", scan_path.path)
                continue

            for subdir in sorted(scan_path.path.iterdir()):
                if not subdir.is_dir():
                    continue
                if subdir.name in SKIP_DIRS:
                    continue

                for handler in self._handlers.values():
                    marker = subdir / handler.marker_file
                    if not marker.is_file():
                        continue

                    entry = handler.discover(marker)
                    if entry is None:
                        continue

                    # Set scope and priority from scan path
                    entry.scope = scan_path.scope
                    entry.priority = scan_path.priority

                    # Handle name collision
                    if entry.name in self._entries:
                        existing = self._entries[entry.name]
                        if existing.priority >= entry.priority:
                            logger.warning(
                                "Extension '%s' from %s (priority %d) shadowed by "
                                "existing from %s (priority %d)",
                                entry.name, scan_path.scope, scan_path.priority,
                                existing.scope, existing.priority,
                            )
                            continue
                        else:
                            logger.warning(
                                "Extension '%s' from %s (priority %d) replaces "
                                "existing from %s (priority %d)",
                                entry.name, scan_path.scope, scan_path.priority,
                                existing.scope, existing.priority,
                            )

                    self._entries[entry.name] = entry

        # Group by handler type for return value
        grouped: Dict[str, List[ExtensionEntry]] = {}
        for entry in self._entries.values():
            grouped.setdefault(entry.handler_type, []).append(entry)

        return grouped

    # --- Catalog ---

    def build_catalog(self) -> str:
        """Build combined system prompt section from all handlers.

        Delegates to each handler's build_catalog() with its entries.

        Returns:
            Combined catalog string. Empty string if no extensions discovered.
        """
        if not self._entries:
            return ""

        # Group entries by handler type
        grouped: Dict[str, List[ExtensionEntry]] = {}
        for entry in self._entries.values():
            grouped.setdefault(entry.handler_type, []).append(entry)

        sections = []
        for handler_name, entries in grouped.items():
            handler = self._handlers.get(handler_name)
            if handler is None:
                continue
            catalog = handler.build_catalog(entries)
            if catalog:
                sections.append(catalog)

        return "\n\n".join(sections)

    # --- Activation ---

    def activate(self, name: str) -> Optional[ActivationResult]:
        """Activate an extension by name.

        Returns cached result if already active (deduplication).

        Args:
            name: The extension name to activate.

        Returns:
            ActivationResult, or None if not found.
        """
        # Deduplication — return cached result
        if name in self._active:
            return self._active[name]

        entry = self._entries.get(name)
        if entry is None:
            return None

        handler = self._handlers.get(entry.handler_type)
        if handler is None:
            return None

        result = handler.activate(entry)
        self._active[name] = result
        return result

    def deactivate(self, name: str) -> None:
        """Deactivate an extension (tracking-only).

        Removes from active set so the extension can be re-activated if needed.
        Does NOT unpin messages or remove content from conversation history.

        Args:
            name: The extension name to deactivate.
        """
        self._active.pop(name, None)

    def is_active(self, name: str) -> bool:
        """Check if an extension is currently active.

        Args:
            name: The extension name to check.
        """
        return name in self._active

    # --- Slash Commands ---

    def try_handle_command(self, command: str) -> Optional[ActivationResult]:
        """If command matches an extension name, activate it.

        Args:
            command: The command string (without leading '/').

        Returns:
            ActivationResult if command matched an extension, None otherwise.
        """
        if command in self._entries:
            return self.activate(command)
        return None

    # --- Tools ---

    def get_tools(self) -> List["FunctionTool"]:
        """Collect tools from all registered handlers.

        Calls handler.get_tools(self) for each handler, passing self as manager.

        Returns:
            Combined list of FunctionTool instances from all handlers.
        """
        tools = []
        for handler in self._handlers.values():
            tools.extend(handler.get_tools(self))
        return tools

    # --- Introspection ---

    @property
    def entries(self) -> Dict[str, ExtensionEntry]:
        """All discovered extensions (copy)."""
        return dict(self._entries)

    @property
    def active_entries(self) -> Dict[str, ExtensionEntry]:
        """Currently active extensions."""
        return {name: self._entries[name] for name in self._active if name in self._entries}
```

- [ ] **Step 4: Run all manager tests**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py -k "Manager" -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/extensions/manager.py tests/test_extensions.py
git commit -m "feat(extensions): implement ExtensionManager with discovery, activation, and slash commands"
```

---

## Chunk 3: SkillFolderHandler

### Task 5: SkillFolderHandler Implementation

**Files:**
- Modify: `src/ToolAgents/extensions/skill_handler.py` (replace stub)
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write failing tests for SkillFolderHandler**

Append to `tests/test_extensions.py`:

```python
from ToolAgents.extensions.skill_handler import SkillFolderHandler


def _create_full_skill(base_path, name, description="A skill", body="# Instructions\nDo things.",
                       extra_frontmatter="", pin_in_context=None, add_scripts=False, add_references=False):
    """Helper: create a full skill directory with optional resources."""
    skill_dir = base_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    fm_lines = [f"name: {name}", f"description: {description}"]
    if pin_in_context is not None:
        fm_lines.append(f"pin_in_context: {str(pin_in_context).lower()}")
    if extra_frontmatter:
        fm_lines.append(extra_frontmatter)

    content = "---\n" + "\n".join(fm_lines) + "\n---\n" + body
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    if add_scripts:
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_text("print('hello')")

    if add_references:
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("# Guide")

    return skill_dir


class TestSkillFolderHandlerDiscovery:
    def test_discover_valid_skill(self, tmp_path):
        _create_full_skill(tmp_path, "my-skill", "Does things", "# My Skill\nInstructions.")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "my-skill" / "SKILL.md")

        assert entry is not None
        assert entry.name == "my-skill"
        assert entry.description == "Does things"
        assert entry.handler_type == "skills"

    def test_discover_missing_description_returns_none(self, tmp_path):
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: bad-skill\n---\n# Bad\n")

        handler = SkillFolderHandler()
        entry = handler.discover(skill_dir / "SKILL.md")
        assert entry is None

    def test_discover_unparseable_yaml_returns_none(self, tmp_path):
        skill_dir = tmp_path / "broken"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\n: : : broken\n---\n# Broken\n")

        handler = SkillFolderHandler()
        entry = handler.discover(skill_dir / "SKILL.md")
        assert entry is None

    def test_discover_extracts_optional_fields(self, tmp_path):
        _create_full_skill(tmp_path, "rich-skill", "Rich", extra_frontmatter="license: MIT")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "rich-skill" / "SKILL.md")

        assert entry is not None
        assert entry.metadata.get("license") == "MIT"

    def test_discover_pin_in_context_custom(self, tmp_path):
        _create_full_skill(tmp_path, "unpin-skill", "Unpinned", pin_in_context=False)
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "unpin-skill" / "SKILL.md")

        assert entry is not None
        assert entry.metadata.get("pin_in_context") is False

    def test_discover_pin_in_context_default_true(self, tmp_path):
        _create_full_skill(tmp_path, "default-skill", "Default pin")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "default-skill" / "SKILL.md")

        assert entry is not None
        # pin_in_context defaults to True when not in frontmatter
        assert entry.metadata.get("pin_in_context", True) is True

    def test_discover_name_mismatch_warns_but_loads(self, tmp_path):
        skill_dir = tmp_path / "dir-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: different-name\ndescription: Mismatched\n---\n# Skill\n"
        )

        handler = SkillFolderHandler()
        entry = handler.discover(skill_dir / "SKILL.md")
        assert entry is not None
        assert entry.name == "different-name"


class TestSkillFolderHandlerCatalog:
    def test_catalog_format(self, tmp_path):
        _create_full_skill(tmp_path, "skill-a", "Does A")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "skill-a" / "SKILL.md")

        catalog = handler.build_catalog([entry])
        assert "<available_skills>" in catalog
        assert 'name="skill-a"' in catalog
        assert "Does A" in catalog
        assert "activate_skill" in catalog

    def test_catalog_empty(self):
        handler = SkillFolderHandler()
        assert handler.build_catalog([]) == ""


class TestSkillFolderHandlerActivation:
    def test_activate_returns_content(self, tmp_path):
        _create_full_skill(tmp_path, "my-skill", "A skill", "# Instructions\nDo X then Y.")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "my-skill" / "SKILL.md")
        result = handler.activate(entry)

        assert '<skill_content name="my-skill">' in result.content
        assert "Do X then Y." in result.content
        assert "</skill_content>" in result.content

    def test_activate_pin_in_context_default(self, tmp_path):
        _create_full_skill(tmp_path, "pin-default", "Default")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "pin-default" / "SKILL.md")
        result = handler.activate(entry)
        assert result.pin_in_context is True

    def test_activate_pin_in_context_false(self, tmp_path):
        _create_full_skill(tmp_path, "no-pin", "No pin", pin_in_context=False)
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "no-pin" / "SKILL.md")
        result = handler.activate(entry)
        assert result.pin_in_context is False

    def test_activate_lists_resources(self, tmp_path):
        _create_full_skill(tmp_path, "rich", "Has resources", add_scripts=True, add_references=True)
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "rich" / "SKILL.md")
        result = handler.activate(entry)

        assert result.resources is not None
        resource_str = " ".join(result.resources)
        assert "scripts/run.py" in resource_str or "scripts\\run.py" in resource_str
        assert "references/guide.md" in resource_str or "references\\guide.md" in resource_str

        # Content should include resource listing
        assert "<skill_resources>" in result.content

    def test_activate_no_resources(self, tmp_path):
        _create_full_skill(tmp_path, "simple", "No resources")
        handler = SkillFolderHandler()
        entry = handler.discover(tmp_path / "simple" / "SKILL.md")
        result = handler.activate(entry)

        # No resource section when there are none
        assert "<skill_resources>" not in result.content


class TestSkillFolderHandlerTools:
    def test_get_tools_with_entries(self, tmp_path):
        _create_full_skill(tmp_path, "my-skill", "A skill")

        mgr = ExtensionManager()
        handler = SkillFolderHandler()
        mgr.register_handler(handler)
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        tools = handler.get_tools(mgr)
        assert len(tools) == 1
        assert tools[0].model.__name__ == "activate_skill"

    def test_get_tools_no_entries(self):
        mgr = ExtensionManager()
        handler = SkillFolderHandler()
        mgr.register_handler(handler)

        tools = handler.get_tools(mgr)
        assert tools == []

    def test_activate_skill_tool_works(self, tmp_path):
        _create_full_skill(tmp_path, "my-skill", "A skill", "# Skill Content\nDo things.")

        mgr = ExtensionManager()
        handler = SkillFolderHandler()
        mgr.register_handler(handler)
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        tools = handler.get_tools(mgr)
        tool = tools[0]

        # Simulate tool execution (pass enum value as string — FunctionTool handles conversion)
        result_str = tool.execute({"skill_name": "my-skill"})
        assert "Skill Content" in result_str
        assert mgr.is_active("my-skill")

        # Should also store pending activation for harness pinning
        assert "my-skill" in mgr._pending_activations
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestSkillFolderHandlerDiscovery::test_discover_valid_skill -v`
Expected: FAIL (skill_handler is a stub)

- [ ] **Step 3: Implement SkillFolderHandler**

Replace `src/ToolAgents/extensions/skill_handler.py` with:

```python
# src/ToolAgents/extensions/skill_handler.py
"""SkillFolderHandler — implements the Agent Skills specification for SKILL.md files."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import yaml

from ToolAgents.function_tool import FunctionTool

from .models import ActivationResult, ExtensionEntry

if TYPE_CHECKING:
    from .manager import ExtensionManager

logger = logging.getLogger(__name__)

# Subdirectories to scan for bundled resources.
_RESOURCE_DIRS = ("scripts", "references", "assets")

# Maximum number of resource files to enumerate.
_MAX_RESOURCES = 50


def _parse_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """Split YAML frontmatter from Markdown body.

    Args:
        text: Full file content.

    Returns:
        Tuple of (parsed YAML dict or None, body string).
    """
    if not text.startswith("---"):
        return None, text

    # Find closing ---
    end = text.find("---", 3)
    if end == -1:
        return None, text

    yaml_str = text[3:end].strip()
    body = text[end + 3:].strip()

    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        # Retry: wrap unquoted values in quotes
        try:
            lines = []
            for line in yaml_str.split("\n"):
                if ":" in line and not line.strip().startswith("#"):
                    key, _, val = line.partition(":")
                    val = val.strip()
                    if val and not val.startswith(("'", '"', "[", "{", "|", ">")):
                        line = f'{key}: "{val}"'
                lines.append(line)
            data = yaml.safe_load("\n".join(lines))
        except yaml.YAMLError:
            return None, body

    if not isinstance(data, dict):
        return None, body

    return data, body


class SkillFolderHandler:
    """Folder handler for Agent Skills (SKILL.md files).

    Implements discovery, catalog generation, activation, and tool
    registration following the Agent Skills specification.

    Attributes:
        name: Handler type identifier ("skills").
        marker_file: File to look for ("SKILL.md").
    """

    name: str = "skills"
    marker_file: str = "SKILL.md"

    def discover(self, path: Path) -> Optional[ExtensionEntry]:
        """Parse a SKILL.md file into an ExtensionEntry.

        Performs lenient validation:
        - Missing description → skip (log error)
        - Unparseable YAML → skip (log error)
        - Name mismatch with directory → warn, load anyway
        - Name too long → warn, load anyway

        Args:
            path: Absolute path to the SKILL.md file.

        Returns:
            ExtensionEntry if valid, None if invalid.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Failed to read %s: %s", path, e)
            return None

        frontmatter, _ = _parse_frontmatter(text)
        if frontmatter is None:
            logger.error("Unparseable YAML in %s", path)
            return None

        # Required fields
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")

        if not description:
            logger.error("Missing description in %s — skipping", path)
            return None

        # Default name to directory name if missing
        if not name:
            name = path.parent.name

        # Lenient warnings
        dir_name = path.parent.name
        if name != dir_name:
            logger.warning(
                "Skill name '%s' does not match directory '%s' in %s",
                name, dir_name, path,
            )
        if len(name) > 64:
            logger.warning("Skill name '%s' exceeds 64 characters in %s", name, path)

        # Build metadata from optional fields
        metadata = {}
        for key in ("license", "compatibility", "allowed-tools"):
            if key in frontmatter:
                metadata[key] = frontmatter[key]

        # Nested metadata dict from frontmatter
        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            metadata.update(frontmatter["metadata"])

        # Custom field: pin_in_context (default True)
        pin_value = frontmatter.get("pin_in_context")
        if pin_value is not None:
            metadata["pin_in_context"] = bool(pin_value)

        return ExtensionEntry(
            name=name,
            description=description,
            handler_type=self.name,
            path=path,
            scope="",       # filled by ExtensionManager
            priority=0,     # filled by ExtensionManager
            metadata=metadata,
        )

    def build_catalog(self, entries: List[ExtensionEntry]) -> str:
        """Build XML-formatted catalog with behavioral instructions.

        Args:
            entries: Discovered skill entries.

        Returns:
            Formatted catalog string for system prompt injection.
            Empty string if no entries.
        """
        if not entries:
            return ""

        lines = [
            "The following skills provide specialized instructions for specific tasks.",
            "When a task matches a skill's description, call the activate_skill tool",
            "with the skill's name to load its full instructions.",
            "Users can also activate skills directly with /skill-name commands.",
            "",
            "<available_skills>",
        ]
        for entry in sorted(entries, key=lambda e: e.name):
            lines.append(
                f'  <skill name="{entry.name}" location="{entry.path}">'
            )
            lines.append(f"    {entry.description}")
            lines.append("  </skill>")
        lines.append("</available_skills>")

        return "\n".join(lines)

    def activate(self, entry: ExtensionEntry) -> ActivationResult:
        """Load full SKILL.md content and enumerate resources.

        Args:
            entry: The skill entry to activate.

        Returns:
            ActivationResult with wrapped content, pinning preference, and resources.
        """
        text = entry.path.read_text(encoding="utf-8")
        _, body = _parse_frontmatter(text)

        skill_dir = entry.path.parent

        # Enumerate bundled resources
        resources = self._enumerate_resources(skill_dir)

        # Build wrapped content
        content_lines = [f'<skill_content name="{entry.name}">']
        content_lines.append(body)
        content_lines.append("")
        content_lines.append(f"Skill directory: {skill_dir}")
        content_lines.append(
            "Relative paths in this skill resolve against the skill directory."
        )

        if resources:
            content_lines.append("")
            content_lines.append("<skill_resources>")
            for res in resources:
                content_lines.append(f"  <file>{res}</file>")
            content_lines.append("</skill_resources>")

        content_lines.append("</skill_content>")

        pin = entry.metadata.get("pin_in_context", True)

        return ActivationResult(
            content="\n".join(content_lines),
            pin_in_context=pin,
            resources=resources if resources else None,
        )

    def get_tools(self, manager: "ExtensionManager") -> List[FunctionTool]:
        """Return the activate_skill tool if skills are discovered.

        Args:
            manager: The ExtensionManager (used by the tool via closure).

        Returns:
            List with one FunctionTool, or empty list if no skills.
        """
        # Only return a tool if there are skill entries
        skill_entries = [
            e for e in manager.entries.values() if e.handler_type == self.name
        ]
        if not skill_entries:
            return []

        # Build enum of valid skill names to constrain tool schema
        skill_names = sorted(e.name for e in skill_entries)
        SkillNameEnum = Enum("SkillNameEnum", {n: n for n in skill_names})

        def activate_skill(skill_name: SkillNameEnum) -> str:
            """Activate a skill by name to load its full instructions into context.

            Call this when a task matches one of the available skills.

            Args:
                skill_name: Name of the skill to activate.

            Returns:
                The skill's full instructions wrapped in structured tags.
            """
            # Extract string value from enum
            name_str = skill_name.value if isinstance(skill_name, Enum) else str(skill_name)
            result = manager.activate(name_str)
            if result is None:
                return f"Error: skill '{name_str}' not found."
            # Store for harness to pick up pinning info
            manager._pending_activations[name_str] = result
            return result.content

        tool = FunctionTool(activate_skill)
        tool.model.__name__ = "activate_skill"
        return [tool]

    @staticmethod
    def _enumerate_resources(skill_dir: Path) -> List[str]:
        """Enumerate bundled resource files in known subdirectories.

        Args:
            skill_dir: The skill's root directory.

        Returns:
            Sorted list of relative path strings (e.g., "scripts/run.py").
            Capped at _MAX_RESOURCES entries.
        """
        resources = []
        for subdir_name in _RESOURCE_DIRS:
            subdir = skill_dir / subdir_name
            if not subdir.is_dir():
                continue
            for file_path in sorted(subdir.rglob("*")):
                if file_path.is_file():
                    rel = file_path.relative_to(skill_dir)
                    resources.append(str(rel).replace("\\", "/"))

        resources.sort()
        if len(resources) > _MAX_RESOURCES:
            logger.warning(
                "Skill '%s' has %d resource files, truncating to %d",
                skill_dir.name, len(resources), _MAX_RESOURCES,
            )
            resources = resources[:_MAX_RESOURCES]

        return resources
```

- [ ] **Step 4: Run all skill handler tests**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py -k "SkillFolderHandler" -v`
Expected: All PASS

- [ ] **Step 5: Run the full test suite so far**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/extensions/skill_handler.py tests/test_extensions.py
git commit -m "feat(extensions): implement SkillFolderHandler with discovery, catalog, activation, and tool"
```

---

## Chunk 4: Harness Integration

### Task 6: Harness Integration — Sync

**Files:**
- Modify: `src/ToolAgents/agent_harness/harness.py:44-100` (constructor), `:337-355` (`_process_agent_buffer`), `:270-316` (`run`), `:420-484` (`create_harness`)
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write failing tests for harness integration**

Append to `tests/test_extensions.py`:

```python
class TestHarnessExtensionIntegration:
    """Tests for ExtensionManager integration with AgentHarness.

    These test the wiring, not full LLM interaction.
    """

    def test_create_harness_with_extension_manager_stores_it(self, tmp_path):
        """Verify the harness stores the extension_manager reference."""
        from unittest.mock import MagicMock
        from ToolAgents.agent_harness.harness import AgentHarness

        provider = MagicMock()
        mgr = ExtensionManager()
        handler = SkillFolderHandler()
        mgr.register_handler(handler)

        harness = AgentHarness(provider=provider, extension_manager=mgr)
        assert harness.extension_manager is mgr

    def test_create_harness_factory_with_extension_manager(self, tmp_path):
        """Verify create_harness accepts extension_manager and wires catalog + tools."""
        from unittest.mock import MagicMock
        from ToolAgents.agent_harness.harness import create_harness

        _create_full_skill(tmp_path, "test-skill", "Test skill", "# Content")

        provider = MagicMock()
        mgr = ExtensionManager()
        mgr.register_handler(SkillFolderHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path, scope="project", priority=10))
        mgr.discover()

        harness = create_harness(
            provider=provider,
            system_prompt="Base prompt.",
            extension_manager=mgr,
        )

        # Catalog should be appended to system prompt
        assert "test-skill" in harness.config.system_prompt
        assert "Base prompt." in harness.config.system_prompt

        # activate_skill tool should be registered
        tool_names = list(harness._tool_registry.tools.keys())
        assert "activate_skill" in tool_names

        # Manager should be accessible
        assert harness.extension_manager is mgr

    def test_create_harness_without_extension_manager(self):
        """Verify harness works fine without extensions."""
        from unittest.mock import MagicMock
        from ToolAgents.agent_harness.harness import create_harness

        provider = MagicMock()
        harness = create_harness(provider=provider, system_prompt="Hello.")

        assert harness.extension_manager is None
        assert harness.config.system_prompt == "Hello."

    def test_slash_command_interception_in_run(self, tmp_path):
        """Verify that /skill-name activates skill and doesn't call chat."""
        from unittest.mock import MagicMock, patch
        from ToolAgents.agent_harness.harness import AgentHarness

        _create_full_skill(tmp_path, "my-skill", "Test", "# Content\nInstructions.")

        provider = MagicMock()
        mgr = ExtensionManager()
        mgr.register_handler(SkillFolderHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        harness = AgentHarness(provider=provider, extension_manager=mgr)

        # Mock IO handler that returns /my-skill then None (exit)
        io_handler = MagicMock()
        io_handler.get_input = MagicMock(side_effect=["/my-skill", None])

        harness.run(io_handler=io_handler)

        # Should have shown confirmation
        io_handler.on_text.assert_called()
        confirm_text = io_handler.on_text.call_args[0][0]
        assert "my-skill" in confirm_text.lower() or "activated" in confirm_text.lower()

        # Skill content should be in messages
        assert len(harness._messages) == 1
        assert harness._messages[0].role.value == "system"

        # Turn counter should NOT have incremented
        assert harness.turn_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestHarnessExtensionIntegration::test_create_harness_with_extension_manager_stores_it -v`
Expected: FAIL (`extension_manager` parameter not accepted)

- [ ] **Step 3: Modify AgentHarness constructor to accept extension_manager**

In `src/ToolAgents/agent_harness/harness.py`, add to the constructor signature and body:

After the existing imports at the top of the file, add:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ToolAgents.extensions.manager import ExtensionManager as _ExtensionManager
```

In `AgentHarness.__init__`, add `extension_manager` parameter after `log_output`:

```python
    def __init__(
        self,
        provider: ChatAPIProvider,
        system_prompt: str = "You are a helpful assistant.",
        config: Optional[HarnessConfig] = None,
        context_manager: Optional[ContextManager] = None,
        settings: Optional[ProviderSettings] = None,
        log_output: bool = False,
        extension_manager: Optional["_ExtensionManager"] = None,
    ):
```

At the end of `__init__`, after the budget exceeded wiring, add:

```python
        # Extension manager (optional)
        self._extension_manager = extension_manager
```

Add property:

```python
    @property
    def extension_manager(self):
        """The ExtensionManager, if one was provided."""
        return self._extension_manager
```

- [ ] **Step 4: Modify `_process_agent_buffer` for extension pinning**

In `AgentHarness._process_agent_buffer`, add pinning logic after the existing loop. Add these imports at the top of the file:

```python
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole, ToolCallResultContent
```

Then modify `_process_agent_buffer`:

```python
    def _process_agent_buffer(self, buffer: List[ChatMessage]) -> None:
        """Walk the agent's last_messages_buffer and update context tracking.

        For each message in the buffer:
        - Assistant messages with token_usage: call on_response() for tracking
        - Assistant messages with tool calls: call notify_tool_call()
        - Tool messages: call notify_tool_result()
        - Tool results from activate_skill: pin if needed, register tools
        """
        for msg in buffer:
            if msg.role == ChatMessageRole.Assistant:
                if msg.token_usage is not None:
                    self._context_manager.on_response(msg)
                if msg.contains_tool_call():
                    self._context_manager.notify_tool_call(msg)
            elif msg.role == ChatMessageRole.Tool:
                self._context_manager.notify_tool_result(msg)

                # Check for extension activation results that need pinning
                if self._extension_manager is not None:
                    for content in msg.content:
                        if (isinstance(content, ToolCallResultContent)
                                and content.tool_call_name == "activate_skill"):
                            # Extract the skill name from the tool result content
                            # to match it against the specific pending activation
                            pending = self._extension_manager._pending_activations
                            # Find the matching pending activation by checking which
                            # skill's content appears in this tool result
                            for act_name in list(pending.keys()):
                                act_result = pending[act_name]
                                if act_result.content in content.tool_call_result:
                                    if act_result.pin_in_context:
                                        self._context_manager.pin_message(msg.id)
                                    if act_result.tools:
                                        self.add_tools(act_result.tools)
                                    del pending[act_name]
                                    break  # one activation per tool result message
```

- [ ] **Step 5: Add slash command interception to `run()`**

In `AgentHarness.run()`, add slash command handling after the `if not user_input.strip(): continue` check:

```python
    def run(self, io_handler: IOHandler = None) -> None:
        if io_handler is None:
            io_handler = ConsoleIOHandler()

        self._events.emit(
            HarnessEvent.HARNESS_START,
            HarnessEventData(event=HarnessEvent.HARNESS_START),
        )

        while not self._stopped:
            user_input = io_handler.get_input()
            if user_input is None:
                break

            if not user_input.strip():
                continue

            # Slash command interception for extensions
            if (user_input.strip().startswith("/")
                    and self._extension_manager is not None):
                command = user_input.strip()[1:]  # strip the /
                result = self._extension_manager.try_handle_command(command)
                if result is not None:
                    # Inject as system message
                    msg = ChatMessage.create_system_message(result.content)
                    self._messages.append(msg)
                    if result.pin_in_context:
                        self._context_manager.pin_message(msg.id)
                    if result.tools:
                        self.add_tools(result.tools)
                    io_handler.on_text(f"Skill '{command}' activated.")
                    continue  # Don't send to LLM

            try:
                if self.config.streaming:
                    for chunk in self.chat_stream(user_input):
                        io_handler.on_chunk(chunk)
                else:
                    response = self.chat(user_input)
                    io_handler.on_text(response)
            except Exception as e:
                io_handler.on_error(e)
                self._events.emit(
                    HarnessEvent.ERROR,
                    HarnessEventData(
                        event=HarnessEvent.ERROR,
                        turn_number=self._turn_count,
                        error=e,
                    ),
                )

        self._events.emit(
            HarnessEvent.HARNESS_STOP,
            HarnessEventData(event=HarnessEvent.HARNESS_STOP),
        )
```

- [ ] **Step 6: Modify `create_harness` factory**

In `create_harness()`, add `extension_manager` parameter and wiring:

```python
def create_harness(
    provider: ChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    max_context_tokens: int = 128000,
    max_turns: int = -1,
    streaming: bool = False,
    total_budget_tokens: Optional[int] = None,
    settings: Optional[ProviderSettings] = None,
    tools: Optional[List[FunctionTool]] = None,
    log_output: bool = False,
    extension_manager=None,
    **context_kwargs,
) -> AgentHarness:
    # ... existing config creation ...

    # If extension_manager provided, append catalog to system prompt
    if extension_manager is not None:
        catalog = extension_manager.build_catalog()
        if catalog:
            system_prompt = system_prompt + "\n\n" + catalog

    config = HarnessConfig(
        system_prompt=system_prompt,
        max_turns=max_turns,
        streaming=streaming,
        context_manager_config=context_config,
    )

    harness = AgentHarness(
        provider=provider,
        config=config,
        settings=settings,
        log_output=log_output,
        extension_manager=extension_manager,
    )

    if tools:
        harness.add_tools(tools)

    # Register extension tools
    if extension_manager is not None:
        ext_tools = extension_manager.get_tools()
        if ext_tools:
            harness.add_tools(ext_tools)

    return harness
```

- [ ] **Step 7: Run harness integration tests**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestHarnessExtensionIntegration -v`
Expected: All PASS

- [ ] **Step 8: Run full existing test suite to check for regressions**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/ -v`
Expected: All PASS (existing tests unaffected)

- [ ] **Step 9: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/agent_harness/harness.py tests/test_extensions.py
git commit -m "feat(extensions): integrate ExtensionManager into AgentHarness — pinning, slash commands, factory"
```

---

### Task 7: Harness Integration — Async

**Files:**
- Modify: `src/ToolAgents/agent_harness/async_harness.py:40-100` (constructor), `:330-345` (`_process_agent_buffer`), `:266-313` (`run`), `:401-450` (`create_async_harness`)
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write failing test for async harness**

Append to `tests/test_extensions.py`:

```python
class TestAsyncHarnessExtensionIntegration:
    def test_async_harness_accepts_extension_manager(self, tmp_path):
        from unittest.mock import MagicMock
        from ToolAgents.agent_harness.async_harness import AsyncAgentHarness

        provider = MagicMock()
        mgr = ExtensionManager()
        mgr.register_handler(SkillFolderHandler())

        harness = AsyncAgentHarness(provider=provider, extension_manager=mgr)
        assert harness.extension_manager is mgr

    def test_create_async_harness_factory(self, tmp_path):
        from unittest.mock import MagicMock
        from ToolAgents.agent_harness.async_harness import create_async_harness

        _create_full_skill(tmp_path, "async-skill", "Async test", "# Async Content")

        provider = MagicMock()
        mgr = ExtensionManager()
        mgr.register_handler(SkillFolderHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path))
        mgr.discover()

        harness = create_async_harness(
            provider=provider,
            system_prompt="Async base.",
            extension_manager=mgr,
        )

        assert "async-skill" in harness.config.system_prompt
        assert harness.extension_manager is mgr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestAsyncHarnessExtensionIntegration -v`
Expected: FAIL

- [ ] **Step 3: Apply the same changes to AsyncAgentHarness**

Apply identical changes to `src/ToolAgents/agent_harness/async_harness.py`:
1. Add `extension_manager` parameter to `AsyncAgentHarness.__init__`
2. Add `self._extension_manager = extension_manager` in constructor
3. Add `extension_manager` property
4. Add pinning logic in `_process_agent_buffer` (same as sync)
5. Add slash command interception in `async def run()` (same logic, after `await asyncio.to_thread(io_handler.get_input)`)
6. Add `extension_manager` parameter to `create_async_harness()` factory (same wiring as sync)

The only difference from sync: in `run()`, the `io_handler.get_input` is already `await asyncio.to_thread(...)`. The slash command check goes right after the empty-string check, same as sync.

- [ ] **Step 4: Run async harness tests**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestAsyncHarnessExtensionIntegration -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/agent_harness/async_harness.py tests/test_extensions.py
git commit -m "feat(extensions): integrate ExtensionManager into AsyncAgentHarness"
```

---

## Chunk 5: Convenience Factory and Final Wiring

### Task 8: Convenience Factory

**Files:**
- Modify: `src/ToolAgents/agent_harness/harness.py` (add `create_harness_with_extensions`)
- Modify: `src/ToolAgents/agent_harness/async_harness.py` (add `create_async_harness_with_extensions`)
- Modify: `src/ToolAgents/agent_harness/__init__.py` (add new exports)
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write failing test**

Append to `tests/test_extensions.py`:

```python
class TestConvenienceFactory:
    def test_create_harness_with_extensions(self, tmp_path):
        from unittest.mock import MagicMock
        from ToolAgents.agent_harness import create_harness_with_extensions

        _create_full_skill(tmp_path, "conv-skill", "Convenience test", "# Content")

        provider = MagicMock()
        harness = create_harness_with_extensions(
            provider=provider,
            system_prompt="Base.",
            skill_paths=[tmp_path],
            scan_defaults=False,
        )

        assert harness.extension_manager is not None
        assert "conv-skill" in harness.config.system_prompt
        assert "activate_skill" in harness._tool_registry.tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestConvenienceFactory -v`
Expected: FAIL

- [ ] **Step 3: Implement convenience factories**

Add to end of `src/ToolAgents/agent_harness/harness.py`:

```python
def create_harness_with_extensions(
    provider: ChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    skill_paths: Optional[List] = None,
    scan_defaults: bool = True,
    **kwargs,
) -> AgentHarness:
    """Create a harness with extension system pre-configured.

    Sets up ExtensionManager + SkillFolderHandler, scans for skills,
    and passes everything to create_harness().

    Args:
        provider: The LLM provider.
        system_prompt: Base system prompt.
        skill_paths: Additional directories to scan for skills.
        scan_defaults: Whether to scan default locations (.agents/skills/).
        **kwargs: Additional arguments passed to create_harness().

    Returns:
        A configured AgentHarness with extensions enabled.
    """
    from pathlib import Path
    from ToolAgents.extensions import ExtensionManager, SkillFolderHandler, ExtensionScanPath

    manager = ExtensionManager()
    manager.register_handler(SkillFolderHandler())

    # Add default scan paths
    if scan_defaults:
        cwd = Path.cwd()
        home = Path.home()

        # Project-level
        for subdir in [".agents/skills", ".claude/skills"]:
            project_path = cwd / subdir
            if project_path.is_dir():
                manager.add_scan_path(ExtensionScanPath(
                    path=project_path, scope="project", priority=10,
                ))

        # User-level
        for subdir in [".agents/skills", ".claude/skills"]:
            user_path = home / subdir
            if user_path.is_dir():
                manager.add_scan_path(ExtensionScanPath(
                    path=user_path, scope="user", priority=0,
                ))

    # Add user-provided paths
    if skill_paths:
        for sp in skill_paths:
            manager.add_scan_path(ExtensionScanPath(
                path=Path(sp), scope="project", priority=10,
            ))

    manager.discover()

    return create_harness(
        provider=provider,
        system_prompt=system_prompt,
        extension_manager=manager,
        **kwargs,
    )
```

Add equivalent `create_async_harness_with_extensions` to `src/ToolAgents/agent_harness/async_harness.py` (same logic, calls `create_async_harness` instead).

- [ ] **Step 4: Update `__init__.py` with new exports**

Add `create_harness_with_extensions` and `create_async_harness_with_extensions` to `__all__` and lazy imports in `src/ToolAgents/agent_harness/__init__.py`.

- [ ] **Step 5: Run tests**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestConvenienceFactory -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add src/ToolAgents/agent_harness/harness.py src/ToolAgents/agent_harness/async_harness.py src/ToolAgents/agent_harness/__init__.py tests/test_extensions.py
git commit -m "feat(extensions): add create_harness_with_extensions convenience factory"
```

---

### Task 9: Final Integration Test

**Files:**
- Test: `tests/test_extensions.py` (append)

- [ ] **Step 1: Write an end-to-end integration test**

Append to `tests/test_extensions.py`:

```python
class TestEndToEnd:
    """Full round-trip: create skills → discover → build catalog → activate → verify pinning."""

    def test_full_lifecycle(self, tmp_path):
        # Create two skills
        _create_full_skill(tmp_path, "code-review", "Review code for bugs and style",
                          "# Code Review\n\n1. Check for bugs\n2. Check style",
                          add_scripts=True)
        _create_full_skill(tmp_path, "testing", "Write and run tests",
                          "# Testing\n\nUse pytest.", pin_in_context=False)

        # Setup
        mgr = ExtensionManager()
        mgr.register_handler(SkillFolderHandler())
        mgr.add_scan_path(ExtensionScanPath(path=tmp_path, scope="project", priority=10))

        # Discovery
        result = mgr.discover()
        assert len(result["skills"]) == 2

        # Catalog
        catalog = mgr.build_catalog()
        assert "code-review" in catalog
        assert "testing" in catalog
        assert "<available_skills>" in catalog

        # Tools
        tools = mgr.get_tools()
        assert len(tools) == 1
        assert tools[0].model.__name__ == "activate_skill"

        # Activation via manager
        cr_result = mgr.activate("code-review")
        assert cr_result is not None
        assert cr_result.pin_in_context is True
        assert "Check for bugs" in cr_result.content
        assert '<skill_content name="code-review">' in cr_result.content
        assert cr_result.resources is not None  # has scripts/

        # Activation via slash command
        test_result = mgr.try_handle_command("testing")
        assert test_result is not None
        assert test_result.pin_in_context is False
        assert "Use pytest" in test_result.content

        # Deduplication
        cr_result2 = mgr.activate("code-review")
        assert cr_result2 is cr_result

        # Deactivate and re-activate
        mgr.deactivate("code-review")
        assert not mgr.is_active("code-review")
        cr_result3 = mgr.activate("code-review")
        assert cr_result3 is not cr_result  # fresh activation

        # Introspection
        assert len(mgr.entries) == 2
        assert len(mgr.active_entries) == 2  # both active now

        # Tool execution
        tool = tools[0]
        tool_output = tool.execute({"skill_name": "testing"})
        assert "Use pytest" in tool_output

        # Unknown activation
        assert mgr.activate("nonexistent") is None
        assert mgr.try_handle_command("nonexistent") is None
```

- [ ] **Step 2: Run the end-to-end test**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/test_extensions.py::TestEndToEnd -v`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `cd H:/Dev42/ToolAgentsDev && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd H:/Dev42/ToolAgentsDev
git add tests/test_extensions.py
git commit -m "test(extensions): add end-to-end integration test for full extension lifecycle"
```

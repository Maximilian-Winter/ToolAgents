"""Tests for the ToolAgents extension system."""
import tempfile
from pathlib import Path

from ToolAgents.extensions.models import ExtensionEntry, ActivationResult, ExtensionScanPath


class TestExtensionEntry:
    def test_create_entry(self):
        entry = ExtensionEntry(
            name="test-skill", description="A test skill", handler_type="skills",
            path=Path("/tmp/skills/test-skill/SKILL.md"), scope="project", priority=10,
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
            path=Path("/tmp/x/SKILL.md"), scope="user", priority=0, metadata={},
        )
        assert entry.metadata == {}


class TestActivationResult:
    def test_defaults(self):
        result = ActivationResult(content="# Instructions")
        assert result.pin_in_context is True
        assert result.tools is None
        assert result.resources is None

    def test_custom_values(self):
        result = ActivationResult(content="hello", pin_in_context=False, resources=["scripts/run.py"])
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


class TestFolderHandlerProtocol:
    def test_protocol_is_runtime_checkable(self):
        from ToolAgents.extensions.handler import FolderHandler
        from ToolAgents.extensions.models import ActivationResult

        class DummyHandler:
            name = "dummy"
            marker_file = "DUMMY.md"
            def discover(self, path): return None
            def build_catalog(self, entries): return ""
            def activate(self, entry): return ActivationResult(content="")
            def get_tools(self, manager): return []

        handler = DummyHandler()
        assert isinstance(handler, FolderHandler)


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


# ---------------------------------------------------------------------------
# Task 4: ExtensionManager tests
# ---------------------------------------------------------------------------

import tempfile

from ToolAgents.extensions.manager import ExtensionManager
from ToolAgents.extensions.models import ExtensionEntry, ActivationResult, ExtensionScanPath


class DummySkillHandler:
    """A minimal handler for testing the manager."""
    name = "skills"
    marker_file = "SKILL.md"

    def discover(self, path):
        text = path.read_text(encoding="utf-8")
        lines = text.strip().split("\n")
        name = path.parent.name
        desc = "No description"
        for line in lines:
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                desc = line.split(":", 1)[1].strip()
        if not desc or desc == "No description":
            return None
        return ExtensionEntry(name=name, description=desc, handler_type=self.name,
                              path=path, scope="", priority=0, metadata={})

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
    skill_dir = base_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nInstructions here.\n")
    return skill_dir


class TestExtensionManagerSetup:
    def test_register_handler(self):
        manager = ExtensionManager()
        handler = DummySkillHandler()
        manager.register_handler(handler)
        assert "skills" in manager._handlers
        assert manager._handlers["skills"] is handler

    def test_add_scan_path(self):
        manager = ExtensionManager()
        sp = ExtensionScanPath(path=Path("/tmp/skills"), scope="project", priority=10)
        manager.add_scan_path(sp)
        assert len(manager._scan_paths) == 1
        assert manager._scan_paths[0] is sp

    def test_register_multiple_handlers(self):
        manager = ExtensionManager()
        h1 = DummySkillHandler()
        h2 = DummySkillHandler()
        h2.name = "tools"
        manager.register_handler(h1)
        manager.register_handler(h2)
        assert "skills" in manager._handlers
        assert "tools" in manager._handlers

    def test_add_multiple_scan_paths(self):
        manager = ExtensionManager()
        sp1 = ExtensionScanPath(path=Path("/tmp/a"), scope="project", priority=10)
        sp2 = ExtensionScanPath(path=Path("/tmp/b"), scope="user", priority=0)
        manager.add_scan_path(sp1)
        manager.add_scan_path(sp2)
        assert len(manager._scan_paths) == 2


class TestExtensionManagerDiscovery:
    def test_discover_finds_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something useful")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            result = manager.discover()
            assert "skills" in result
            assert len(result["skills"]) == 1
            assert result["skills"][0].name == "my-skill"

    def test_discover_sets_scope_and_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "scoped-skill", "Has scope")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="user", priority=5))
            result = manager.discover()
            entry = result["skills"][0]
            assert entry.scope == "user"
            assert entry.priority == 5

    def test_discover_skips_git_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Create a .git directory with a SKILL.md — should be skipped
            git_dir = base / ".git"
            git_dir.mkdir()
            (git_dir / "SKILL.md").write_text("name: git-skill\ndescription: Should be skipped\n")
            # Create a real skill
            _create_skill_dir(base, "real-skill", "Real skill")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            result = manager.discover()
            names = [e.name for e in result.get("skills", [])]
            assert "git-skill" not in names
            assert "real-skill" in names

    def test_discover_higher_priority_wins_collision(self):
        with tempfile.TemporaryDirectory() as tmp_a:
            with tempfile.TemporaryDirectory() as tmp_b:
                base_a = Path(tmp_a)
                base_b = Path(tmp_b)
                _create_skill_dir(base_a, "shared-skill", "Version A")
                _create_skill_dir(base_b, "shared-skill", "Version B")
                manager = ExtensionManager()
                manager.register_handler(DummySkillHandler())
                # base_a has higher priority
                manager.add_scan_path(ExtensionScanPath(path=base_a, scope="project", priority=10))
                manager.add_scan_path(ExtensionScanPath(path=base_b, scope="user", priority=0))
                result = manager.discover()
                assert len(result["skills"]) == 1
                assert result["skills"][0].description == "Version A"

    def test_discover_equal_priority_first_wins(self):
        with tempfile.TemporaryDirectory() as tmp_a:
            with tempfile.TemporaryDirectory() as tmp_b:
                base_a = Path(tmp_a)
                base_b = Path(tmp_b)
                _create_skill_dir(base_a, "shared-skill", "Version A")
                _create_skill_dir(base_b, "shared-skill", "Version B")
                manager = ExtensionManager()
                manager.register_handler(DummySkillHandler())
                # Both have same priority — first added wins
                manager.add_scan_path(ExtensionScanPath(path=base_a, scope="project", priority=5))
                manager.add_scan_path(ExtensionScanPath(path=base_b, scope="user", priority=5))
                result = manager.discover()
                assert len(result["skills"]) == 1
                assert result["skills"][0].description == "Version A"

    def test_discover_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=Path(tmp), scope="project", priority=10))
            result = manager.discover()
            assert result == {}

    def test_discover_nonexistent_path(self):
        manager = ExtensionManager()
        manager.register_handler(DummySkillHandler())
        manager.add_scan_path(ExtensionScanPath(path=Path("/nonexistent/path/xyz"), scope="project", priority=10))
        result = manager.discover()
        assert result == {}

    def test_discover_clears_previous_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "skill-one", "First skill")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            assert "skill-one" in manager.entries
            # Remove skill and rediscover
            import shutil
            shutil.rmtree(base / "skill-one")
            result2 = manager.discover()
            assert "skill-one" not in manager.entries
            assert result2 == {}


class TestExtensionManagerCatalog:
    def test_build_catalog_with_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "alpha", "Does alpha things")
            _create_skill_dir(base, "beta", "Does beta things")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            catalog = manager.build_catalog()
            assert "<skills>" in catalog
            assert "alpha" in catalog
            assert "beta" in catalog

    def test_build_catalog_empty(self):
        manager = ExtensionManager()
        manager.register_handler(DummySkillHandler())
        catalog = manager.build_catalog()
        assert catalog == ""


class TestExtensionManagerActivation:
    def test_activate_known_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            result = manager.activate("my-skill")
            assert result is not None
            assert isinstance(result, ActivationResult)
            assert "my-skill" in result.content

    def test_activate_unknown_returns_none(self):
        manager = ExtensionManager()
        manager.register_handler(DummySkillHandler())
        result = manager.activate("nonexistent-skill")
        assert result is None

    def test_activate_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            result1 = manager.activate("my-skill")
            result2 = manager.activate("my-skill")
            assert result1 is result2

    def test_is_active_after_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            assert not manager.is_active("my-skill")
            manager.activate("my-skill")
            assert manager.is_active("my-skill")

    def test_deactivate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            manager.activate("my-skill")
            assert manager.is_active("my-skill")
            manager.deactivate("my-skill")
            assert not manager.is_active("my-skill")

    def test_deactivate_allows_reactivation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            manager.activate("my-skill")
            manager.deactivate("my-skill")
            result = manager.activate("my-skill")
            assert result is not None
            assert manager.is_active("my-skill")

    def test_deactivate_unknown_no_error(self):
        manager = ExtensionManager()
        manager.deactivate("nonexistent")  # Should not raise


class TestExtensionManagerSlashCommands:
    def test_try_handle_known_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "my-skill", "Does something")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            result = manager.try_handle_command("my-skill")
            assert result is not None
            assert isinstance(result, ActivationResult)

    def test_try_handle_unknown_command(self):
        manager = ExtensionManager()
        manager.register_handler(DummySkillHandler())
        result = manager.try_handle_command("unknown-command")
        assert result is None


class TestExtensionManagerIntrospection:
    def test_entries_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "skill-a", "Skill A")
            _create_skill_dir(base, "skill-b", "Skill B")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            entries = manager.entries
            assert "skill-a" in entries
            assert "skill-b" in entries
            # Verify it's a copy, not internal dict
            entries["skill-a"] = None
            assert manager.entries["skill-a"] is not None

    def test_active_entries_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _create_skill_dir(base, "skill-a", "Skill A")
            _create_skill_dir(base, "skill-b", "Skill B")
            manager = ExtensionManager()
            manager.register_handler(DummySkillHandler())
            manager.add_scan_path(ExtensionScanPath(path=base, scope="project", priority=10))
            manager.discover()
            assert manager.active_entries == {}
            manager.activate("skill-a")
            active = manager.active_entries
            assert "skill-a" in active
            assert "skill-b" not in active

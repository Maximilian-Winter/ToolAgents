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

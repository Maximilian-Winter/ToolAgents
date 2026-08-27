#!/usr/bin/env python3
"""Extended NavigableMemory feature test suite.

Standalone example tests for the Obsidian Forge navigable-memory demo.
No LLM calls, no network calls, no pytest dependency.

Run from the project root:
    python examples/agents/navigable_memory/test/test_extended_features.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
SRC_DIR = PROJECT_ROOT / "src"

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
    """Load only the modules this standalone script needs from ./src.

    This fallback lets the example run from a checkout even if importing the
    top-level ToolAgents package pulls in optional dependencies that are not
    installed in the current environment.
    """
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
    from ToolAgents.agent_harness.prompt_composer import PromptComposer
    from ToolAgents.agent_harness.smart_messages import (
        ExpiryAction,
        MessageLifecycle,
        SmartMessageManager,
    )
    from ToolAgents.agent_memory.navigable_memory import (
        FilesystemBackend,
        InMemoryBackend,
        JSONBackend,
        NavigableMemory,
        RefType,
        SQLiteBackend,
        migrate,
    )
    from ToolAgents.data_models.messages import ChatMessage
except Exception:
    _bootstrap_local_imports()
    from ToolAgents.agent_harness.prompt_composer import PromptComposer
    from ToolAgents.agent_harness.smart_messages import (
        ExpiryAction,
        MessageLifecycle,
        SmartMessageManager,
    )
    from ToolAgents.agent_memory.navigable_memory import (
        FilesystemBackend,
        InMemoryBackend,
        JSONBackend,
        NavigableMemory,
        RefType,
        SQLiteBackend,
        migrate,
    )
    from ToolAgents.data_models.messages import ChatMessage

from seed_obsidian_forge import seed as seed_knowledge_base


BINARY_PAYLOAD = b"\x89PNG\r\n\x1a\nnavmem-test-image-bytes"
TEXT_PATH = "lab/text.md"
VERSION_PATH = "lab/versioned.md"
BINARY_PATH = "lab/assets/diagram.png"
TAG_PATH = "lab/tags.md"
REF_A = "lab/refs/a.md"
REF_B = "lab/refs/b.md"
REF_C = "lab/refs/c.md"
REF_MISSING = "lab/refs/missing.md"


class TestResults:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.errors: list[tuple[str, str]] = []
        self.current_section = ""

    def section(self, name: str) -> None:
        self.current_section = name
        print(f"\n-- {name} --")

    def ok(self, name: str) -> None:
        self.passed += 1
        print(f"  OK   {name}")

    def fail(self, name: str, reason: str) -> None:
        self.failed += 1
        label = f"{self.current_section}: {name}" if self.current_section else name
        self.errors.append((label, reason))
        print(f"  FAIL {name}: {reason}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print("\n" + "=" * 72)
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print("=" * 72)
        return self.failed == 0


results = TestResults()


def assert_eq(name: str, actual: Any, expected: Any) -> None:
    if actual == expected:
        results.ok(name)
    else:
        results.fail(name, f"expected {expected!r}, got {actual!r}")


def assert_true(name: str, condition: Any, detail: str = "") -> None:
    if condition:
        results.ok(name)
    else:
        results.fail(name, detail or "condition was false")


def assert_in(name: str, needle: str, haystack: str) -> None:
    if needle in haystack:
        results.ok(name)
    else:
        results.fail(name, f"{needle!r} not found")


def assert_not_in(name: str, needle: str, haystack: str) -> None:
    if needle not in haystack:
        results.ok(name)
    else:
        results.fail(name, f"{needle!r} unexpectedly found")


def assert_paths(name: str, docs: list[Any], expected: set[str]) -> None:
    actual = {doc.path for doc in docs}
    missing = expected - actual
    if not missing:
        results.ok(name)
    else:
        results.fail(name, f"missing paths: {sorted(missing)!r}")


def close_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def backend_specs(tmpdir: str) -> list[tuple[str, Callable[[], Any], Optional[Callable[[], Any]]]]:
    root = Path(tmpdir)
    return [
        ("InMemoryBackend", lambda: InMemoryBackend(), None),
        (
            "JSONBackend",
            lambda: JSONBackend(str(root / "memory.json")),
            lambda: JSONBackend(str(root / "memory.json")),
        ),
        (
            "SQLiteBackend",
            lambda: SQLiteBackend(str(root / "memory.db")),
            lambda: SQLiteBackend(str(root / "memory.db")),
        ),
        (
            "FilesystemBackend",
            lambda: FilesystemBackend(str(root / "fs-memory")),
            lambda: FilesystemBackend(str(root / "fs-memory")),
        ),
    ]


def write_core_fixture(nav: NavigableMemory) -> None:
    nav.write(TEXT_PATH, "Text Doc", "alpha searchable body", ["alpha"], {"n": 1})
    nav.write("lab/delete-me.md", "Delete Me", "temporary")
    nav.write(VERSION_PATH, "Versioned Doc", "first line", ["history"])
    nav.write(VERSION_PATH, "Versioned Doc", "second line", ["history"])
    nav.write_binary(
        BINARY_PATH,
        "Architecture Diagram",
        "image/png",
        BINARY_PAYLOAD,
        caption="diagram caption",
        tags=["asset", "diagram"],
        metadata={"kind": "fixture"},
    )
    nav.write(TAG_PATH, "Tag Doc", "tag body", ["initial"])
    nav.write(REF_A, "Reference A", "alpha node", ["graph"])
    nav.write(REF_B, "Reference B", "beta node", ["graph"])
    nav.write(REF_C, "Reference C", "gamma node", ["graph"])


def check_backend_core(name: str, nav: NavigableMemory) -> None:
    prefix = f"{name}: "

    assert_eq(prefix + "read text", nav.read(TEXT_PATH).content, "alpha searchable body")
    assert_paths(prefix + "list prefix", nav.list_at("lab/"), {TEXT_PATH, VERSION_PATH})
    assert_paths(prefix + "search text", nav.search("searchable"), {TEXT_PATH})

    assert_true(prefix + "delete existing", nav.backend.delete("lab/delete-me.md"))
    assert_eq(prefix + "delete removed doc", nav.read("lab/delete-me.md"), None)

    binary_doc = nav.read(BINARY_PATH)
    assert_true(prefix + "binary doc exists", binary_doc is not None)
    assert_true(prefix + "binary doc is binary", binary_doc.is_binary)
    assert_eq(prefix + "binary mime", binary_doc.mime_type, "image/png")
    assert_eq(prefix + "binary bytes", nav.read_binary(BINARY_PATH), BINARY_PAYLOAD)
    if hasattr(nav.backend, "list_by_mime_type"):
        assert_paths(
            prefix + "mime listing",
            nav.backend.list_by_mime_type("image/"),
            {BINARY_PATH},
        )

    assert_eq(prefix + "raw versions", [v.version for v in nav.list_versions(VERSION_PATH)], [2, 1])
    assert_eq(prefix + "history excludes current", [v.version for v in nav.list_history(VERSION_PATH)], [1])
    assert_in(prefix + "format version", "first line", nav.format_version(VERSION_PATH, 1))
    diff = nav.compare_versions(VERSION_PATH, 1)
    assert_in(prefix + "version diff old", "-first line", diff)
    assert_in(prefix + "version diff current", "+second line", diff)
    assert_in(
        prefix + "version context",
        "Previous versions of",
        nav.build_version_context(VERSION_PATH, include_content=True),
    )
    assert_true(prefix + "rollback version", nav.rollback(VERSION_PATH, 1))
    assert_eq(prefix + "rollback content", nav.read(VERSION_PATH).content, "first line")
    removed_versions = nav.prune_versions(VERSION_PATH, 2)
    assert_true(prefix + "prune removed versions", removed_versions >= 1)
    assert_eq(prefix + "prune keeps two", len(nav.list_versions(VERSION_PATH)), 2)

    assert_true(prefix + "set tags", nav.set_tags(TAG_PATH, ["alpha", "beta"]))
    assert_true(prefix + "add tags", nav.add_tags(TAG_PATH, "gamma", "alpha"))
    assert_eq(prefix + "ordered tag merge", nav.read(TAG_PATH).tags, ["alpha", "beta", "gamma"])
    assert_true(prefix + "remove tag", nav.remove_tags(TAG_PATH, "beta"))
    assert_eq(prefix + "tag removed", nav.read(TAG_PATH).tags, ["alpha", "gamma"])
    assert_true(prefix + "list tags", "alpha" in nav.list_tags())
    assert_paths(prefix + "list by tag", nav.list_by_tag("alpha"), {TAG_PATH})
    assert_paths(prefix + "find tags any", nav.find_by_tags(["gamma"], "any"), {TAG_PATH})
    assert_paths(prefix + "find tags all", nav.find_by_tags(["alpha", "gamma"], "all"), {TAG_PATH})
    none_paths = {d.path for d in nav.find_by_tags(["alpha"], "none")}
    assert_true(prefix + "find tags none excludes", TAG_PATH not in none_paths)

    assert_true(prefix + "add reference", nav.add_reference(REF_A, REF_B, RefType.DEPENDS_ON, "depends"))
    duplicate_ref = nav.add_reference(REF_A, REF_B, RefType.DEPENDS_ON, "duplicate")
    assert_true(prefix + "duplicate reference false", not duplicate_ref)
    assert_true(prefix + "add second reference", nav.add_reference(REF_B, REF_C, RefType.SEE_ALSO, "next"))
    assert_true(
        prefix + "add missing reference",
        nav.add_reference(REF_C, REF_MISSING, RefType.LINKS_TO, "missing target"),
    )
    assert_eq(prefix + "outgoing ref count", len(nav.references_from(REF_A)), 1)
    assert_eq(prefix + "incoming ref count", len(nav.references_to(REF_B)), 1)
    assert_true(prefix + "all refs count", len(nav.all_references()) >= 3)
    walk = nav.walk_references(REF_A, direction="outgoing", max_depth=3)
    assert_true(prefix + "walk reaches c", REF_C in {n["path"] for n in walk["nodes"]})
    assert_true(prefix + "walk includes missing node", REF_MISSING in {n["path"] for n in walk["nodes"]})
    rendered = nav.render_reference_walk(walk)
    assert_in(prefix + "render reference walk", REF_A, rendered)
    filtered = nav.walk_references(REF_A, direction="outgoing", ref_types=[RefType.DEPENDS_ON])
    assert_true(prefix + "filtered walk excludes c", REF_C not in {n["path"] for n in filtered["nodes"]})
    truncated = nav.walk_references(REF_A, direction="outgoing", max_depth=3, max_nodes=2)
    assert_true(prefix + "walk truncates", truncated["truncated"])
    assert_eq(prefix + "remove reference", nav.remove_reference(REF_A, REF_B, RefType.DEPENDS_ON), 1)

    if hasattr(nav.backend, "tree"):
        tree = nav.backend.tree("lab/")
        assert_true(prefix + "tree returns mapping", isinstance(tree, dict) and bool(tree))
    if hasattr(nav.backend, "stats"):
        stats = nav.backend.stats()
        assert_true(prefix + "stats count docs", stats.get("documents", 0) >= 5)


def test_backend_matrix() -> None:
    results.section("Backend Matrix")
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, factory, _reload_factory in backend_specs(tmpdir):
            backend = factory()
            nav = NavigableMemory(backend)
            write_core_fixture(nav)
            check_backend_core(name, nav)
            close_backend(backend)


def test_backend_persistence_reload() -> None:
    results.section("Backend Persistence Reload")
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, factory, reload_factory in backend_specs(tmpdir):
            if reload_factory is None:
                continue
            backend = factory()
            nav = NavigableMemory(backend)
            write_core_fixture(nav)
            nav.add_reference(REF_A, REF_B, RefType.LINKS_TO, "reload edge")
            nav.set_tags(TAG_PATH, ["persisted"])
            close_backend(backend)

            reloaded_backend = reload_factory()
            reloaded = NavigableMemory(reloaded_backend)
            assert_eq(name + ": reload text", reloaded.read(TEXT_PATH).content, "alpha searchable body")
            assert_eq(name + ": reload binary", reloaded.read_binary(BINARY_PATH), BINARY_PAYLOAD)
            assert_eq(name + ": reload versions", [v.version for v in reloaded.list_versions(VERSION_PATH)], [2, 1])
            assert_eq(name + ": reload tags", reloaded.read(TAG_PATH).tags, ["persisted"])
            assert_eq(name + ": reload references", len(reloaded.references_from(REF_A)), 1)
            close_backend(reloaded_backend)


def test_navigation_context() -> None:
    results.section("Navigation Context")
    departures = []

    def on_depart(record: Any) -> None:
        departures.append(record)

    nav = NavigableMemory(
        InMemoryBackend(),
        on_depart=on_depart,
        context_window=3,
        include_siblings=True,
        include_parent=True,
    )
    count = seed_knowledge_base(nav)
    assert_true("seed corpus size", count >= 40, f"got {count}")

    chain = [
        "studio/overview.md",
        "studio/projects/ashenmoor/overview.md",
        "studio/projects/ashenmoor/design/overview.md",
        "studio/projects/ashenmoor/design/combat/overview.md",
        "studio/projects/ashenmoor/design/combat/boss-design/overview.md",
        "studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md",
    ]
    for path in chain:
        nav.navigate(path)

    assert_eq("current path", nav.current_path, chain[-1])
    assert_eq("history length", len(nav.history), len(chain))
    assert_eq("departure count", len(departures), len(chain) - 1)
    context = nav.build_context()
    assert_in("context current doc", "Thornqueen", context)
    assert_in("context sibling", "Ashen Guardian", context)
    assert_in("context parent", "Boss Design", context)
    assert_in("history context", "Boss Design", nav.build_history_context())

    up_result = nav.navigate_up()
    assert_in("navigate up overview", "Boss Design Philosophy", up_result)
    assert_eq("navigate up path", nav.current_path, "studio/projects/ashenmoor/design/combat/boss-design/overview.md")

    append_result = nav.append(nav.current_path, "Extended test note")
    assert_in("append result", "Appended", append_result)
    assert_in("append visible", "Extended test note", nav.build_context())

    missing = nav.navigate("studio/does-not-exist.md")
    assert_in("missing path handled", "Location not found", missing)


def test_generated_tools() -> None:
    results.section("Generated Tools")
    nav = NavigableMemory(InMemoryBackend(), include_siblings=True, include_parent=True)
    write_core_fixture(nav)
    nav.add_reference(REF_A, REF_B, RefType.LINKS_TO, "tool edge")

    tools = {tool.__name__: tool for tool in nav.create_tools()}
    expected = {
        "Navigate",
        "NavigateUp",
        "ListLocations",
        "SearchKnowledge",
        "ReadDocument",
        "WriteDocument",
        "AppendToDocument",
        "ListVersions",
        "ReadVersion",
        "CompareVersions",
        "ShowVersionContext",
        "AddReference",
        "RemoveReference",
        "ListReferences",
        "FollowReferences",
        "ListTags",
        "FindByTag",
        "FindByTags",
        "AddTags",
        "RemoveTags",
        "SetTags",
        "DescribeBinary",
    }
    assert_true("tool names complete", expected.issubset(set(tools)), str(sorted(set(tools))))

    assert_in("navigate tool", "Navigated", tools["Navigate"](path=TEXT_PATH).run())
    assert_in("list tool", TEXT_PATH, tools["ListLocations"](prefix="lab/").run())
    assert_in("search tool", "Text Doc", tools["SearchKnowledge"](query="searchable").run())
    assert_in("read tool", "alpha searchable", tools["ReadDocument"](path=TEXT_PATH).run())
    assert_in(
        "write tool",
        "Written",
        tools["WriteDocument"](
            path="lab/tool-write.md", title="Tool Write", content="body",
        ).run(),
    )
    assert_in("append tool", "Appended", tools["AppendToDocument"](path="lab/tool-write.md", content="more").run())

    assert_in("list versions tool", "v1", tools["ListVersions"](path=VERSION_PATH).run())
    assert_in("read version tool", "first line", tools["ReadVersion"](path=VERSION_PATH, version=1).run())
    assert_in(
        "compare versions tool",
        "-first line",
        tools["CompareVersions"](path=VERSION_PATH, from_version=1).run(),
    )
    assert_in(
        "version context tool",
        "Previous versions",
        tools["ShowVersionContext"](path=VERSION_PATH, include_content=True).run(),
    )

    assert_in("list tags tool", "history", tools["ListTags"]().run())
    assert_in("find tag tool", VERSION_PATH, tools["FindByTag"](tag="history").run())
    assert_in("find tags tool", VERSION_PATH, tools["FindByTags"](tags=["history"], mode="any").run())
    assert_in("add tags tool", "Current", tools["AddTags"](path=TEXT_PATH, tags=["tooltag"]).run())
    assert_in("remove tags tool", "Current", tools["RemoveTags"](path=TEXT_PATH, tags=["tooltag"]).run())
    assert_in("set tags tool", "set to", tools["SetTags"](path=TEXT_PATH, tags=["reset"]).run())

    assert_in(
        "add reference tool",
        "Linked",
        tools["AddReference"](from_path=REF_B, to_path=REF_C, ref_type="see_also", note="tool").run(),
    )
    assert_in("list references tool", REF_C, tools["ListReferences"](path=REF_B).run())
    assert_in("follow references tool", REF_B, tools["FollowReferences"](path=REF_A, max_depth=2).run())
    assert_in(
        "remove reference tool",
        "Removed",
        tools["RemoveReference"](from_path=REF_B, to_path=REF_C, ref_type="see_also").run(),
    )
    assert_in("describe binary tool", "image/png", tools["DescribeBinary"](path=BINARY_PATH).run())


def make_migration_source() -> NavigableMemory:
    nav = NavigableMemory(InMemoryBackend())
    write_core_fixture(nav)
    nav.add_reference(REF_A, REF_B, RefType.LINKS_TO, "migrate edge")
    nav.set_tags(TAG_PATH, ["migrated", "tagged"])
    return nav


def assert_migrated_state(label: str, nav: NavigableMemory) -> None:
    assert_eq(label + " text", nav.read(TEXT_PATH).content, "alpha searchable body")
    assert_eq(label + " binary", nav.read_binary(BINARY_PATH), BINARY_PAYLOAD)
    assert_eq(label + " versions", [v.version for v in nav.list_versions(VERSION_PATH)], [2, 1])
    assert_eq(label + " tags", nav.read(TAG_PATH).tags, ["migrated", "tagged"])
    assert_eq(label + " refs", len(nav.references_from(REF_A)), 1)


def test_migration() -> None:
    results.section("Migration")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = make_migration_source()

        json_backend = JSONBackend(str(root / "migrated.json"))
        report_1 = migrate(source.backend, json_backend)
        assert_true("inmemory to json ok", report_1.ok, str(report_1.errors))
        assert_true("inmemory to json docs", report_1.documents >= 4, str(report_1))
        assert_true("inmemory to json binaries", report_1.binaries >= 1, str(report_1))
        assert_true("inmemory to json versions", report_1.versions >= 2, str(report_1))
        assert_true("inmemory to json refs", report_1.references >= 1, str(report_1))
        json_nav = NavigableMemory(json_backend)
        assert_migrated_state("json migrated", json_nav)

        skip_report = migrate(source.backend, json_backend, overwrite=False)
        assert_true("skip existing docs", skip_report.skipped >= 1, str(skip_report))

        sqlite_backend = SQLiteBackend(str(root / "migrated.db"))
        report_2 = migrate(json_backend, sqlite_backend)
        assert_true("json to sqlite ok", report_2.ok, str(report_2.errors))
        sqlite_nav = NavigableMemory(sqlite_backend)
        assert_migrated_state("sqlite migrated", sqlite_nav)

        fs_backend = FilesystemBackend(str(root / "migrated-fs"))
        report_3 = migrate(sqlite_backend, fs_backend)
        assert_true("sqlite to filesystem ok", report_3.ok, str(report_3.errors))
        fs_nav = NavigableMemory(fs_backend)
        assert_migrated_state("filesystem migrated", fs_nav)

        close_backend(json_backend)
        close_backend(sqlite_backend)


class CoreMemory:
    def __init__(self, block_limit: int = 600) -> None:
        self.blocks: dict[str, str] = {}
        self.block_limit = block_limit
        self.last_modified = "never"

    def set_block(self, name: str, content: str) -> str:
        if len(content) > self.block_limit:
            return f"Error: exceeds {self.block_limit} char limit."
        self.blocks[name] = content
        self.last_modified = "updated"
        return f"Core memory '{name}' updated."

    def build_context(self) -> str:
        if not self.blocks:
            return "(no memory blocks stored)"
        return "\n".join(f"<{key}>{value}</{key}>" for key, value in self.blocks.items())

    def to_dict(self) -> dict[str, Any]:
        return {"blocks": dict(self.blocks), "last_modified": self.last_modified}

    def from_dict(self, data: dict[str, Any]) -> None:
        self.blocks = dict(data.get("blocks", {}))
        self.last_modified = data.get("last_modified", "restored")


def archive_search(manager: SmartMessageManager, query: str) -> str:
    found = []
    for msg in manager.archive:
        text = msg.get_as_text()
        if query.lower() in text.lower():
            found.append(text[:200])
    if not found:
        return f"No archived items matching '{query}'."
    return f"Found {len(found)} archived item(s):\n" + "\n---\n".join(found)


def message_from_role(role: str, text: str) -> ChatMessage:
    if role == "user":
        return ChatMessage.create_user_message(text)
    if role == "assistant":
        return ChatMessage.create_assistant_message(text)
    return ChatMessage.create_system_message(text)


def test_demo_integration() -> None:
    results.section("Demo Integration")
    core = CoreMemory()
    core.set_block("focus", "Reviewing Ashenmoor risks")
    manager = SmartMessageManager()

    def on_depart(record: Any) -> None:
        snippet = record.content[:120].replace("\n", " ")
        manager.add_message(
            ChatMessage.create_system_message(f"[Previously at] {record.title}: {snippet}"),
            MessageLifecycle(ttl=2, on_expire=ExpiryAction.ARCHIVE),
        )

    nav = NavigableMemory(
        InMemoryBackend(),
        on_depart=on_depart,
        context_window=3,
        include_siblings=True,
        include_parent=True,
    )
    seed_knowledge_base(nav)

    composer = PromptComposer()
    composer.add_module("instructions", position=0, content="You are Forge.")
    composer.add_module("core", position=5, content_fn=core.build_context)
    composer.add_module("location", position=10, content_fn=nav.build_context)
    composer.add_module(
        "session",
        position=20,
        content_fn=lambda: f"Active msgs: {manager.message_count}",
    )

    nav.navigate("studio/overview.md")
    nav.navigate("studio/projects/ashenmoor/qa/critical-bugs.md")
    prompt = composer.compile()
    assert_in("prompt has core memory", "Reviewing Ashenmoor risks", prompt)
    assert_in("prompt has current location", "Critical Bugs", prompt)
    assert_in("prompt has session", "Active msgs", prompt)
    assert_eq("departure message active", manager.message_count, 1)

    manager.tick()
    manager.tick()
    assert_true("departure archived", len(manager.archive) >= 1)
    assert_in("archive search", "Obsidian Forge", archive_search(manager, "Obsidian"))

    manager.add_message(ChatMessage.create_user_message("Summarize blockers"))
    manager.add_message(
        ChatMessage.create_assistant_message("VFX and multiplayer are key blockers."),
        MessageLifecycle(ttl=5, on_expire=ExpiryAction.ARCHIVE),
    )
    state = {
        "core_memory": core.to_dict(),
        "current_location": nav.current_path,
        "location_history": nav.history,
        "active_messages": [
            {
                "role": sm.message.role.value,
                "text": sm.message.get_as_text(),
                "ttl": sm.lifecycle.ttl,
                "turns_alive": sm.lifecycle.turns_alive,
                "pinned": sm.lifecycle.pinned,
                "on_expire": sm.lifecycle.on_expire.value,
            }
            for sm in manager.get_smart_messages()
        ],
        "archive": [msg.get_as_text() for msg in manager.archive],
        "tick_count": manager.tick_count,
    }
    assert_eq("saved current location", state["current_location"], "studio/projects/ashenmoor/qa/critical-bugs.md")
    assert_true("saved active messages", len(state["active_messages"]) >= 2)
    assert_true("saved archive", len(state["archive"]) >= 1)

    restored_core = CoreMemory()
    restored_core.from_dict(state["core_memory"])
    restored_manager = SmartMessageManager()
    for item in state["active_messages"]:
        restored_manager.add_message(
            message_from_role(item["role"], item["text"]),
            MessageLifecycle(
                ttl=item["ttl"],
                turns_alive=item["turns_alive"],
                pinned=item["pinned"],
                on_expire=ExpiryAction(item["on_expire"]),
            ),
        )
    restored_nav = NavigableMemory(InMemoryBackend())
    seed_knowledge_base(restored_nav)
    restored_nav.navigate(state["current_location"])

    assert_eq("restored core memory", restored_core.blocks["focus"], "Reviewing Ashenmoor risks")
    assert_eq("restored active count", restored_manager.message_count, len(state["active_messages"]))
    assert_eq("restored nav location", restored_nav.current_path, state["current_location"])


def main() -> int:
    print("=" * 72)
    print("Extended NavigableMemory Feature Suite")
    print("Obsidian Forge corpus + backend parity + generated tools")
    print("=" * 72)

    test_backend_matrix()
    test_backend_persistence_reload()
    test_navigation_context()
    test_generated_tools()
    test_migration()
    test_demo_integration()

    return 0 if results.summary() else 1


if __name__ == "__main__":
    sys.exit(main())

import tempfile

from ToolAgents.agent_memory.navigable_memory import (
    FilesystemBackend,
    InMemoryBackend,
    JSONBackend,
    NavigableMemory,
    SQLiteBackend,
)


def test_navigable_memory_lists_previous_versions_and_diffs_against_current():
    memory = NavigableMemory(InMemoryBackend())

    memory.write(
        "notes/profile.md",
        "Profile",
        "Favorite language: Python\nFavorite editor: Vim",
        change_note="initial",
    )
    memory.write(
        "notes/profile.md",
        "Profile",
        "Favorite language: Python\nFavorite editor: VS Code",
        change_note="updated editor",
    )

    current = memory.read("notes/profile.md")
    assert current is not None
    assert current.version == 2

    raw_versions = memory.list_versions("notes/profile.md")
    assert [v.version for v in raw_versions] == [2, 1]

    previous_versions = memory.list_history("notes/profile.md")
    assert [v.version for v in previous_versions] == [1]
    assert previous_versions[0].content.endswith("Vim")

    rendered = memory.format_version("notes/profile.md", 1)
    assert "## Profile (v1)" in rendered
    assert "Favorite editor: Vim" in rendered

    diff = memory.compare_versions("notes/profile.md", from_version=1)
    assert "--- notes/profile.md (v1)" in diff
    assert "+++ notes/profile.md (current v2)" in diff
    assert "-Favorite editor: Vim" in diff
    assert "+Favorite editor: VS Code" in diff

    context = memory.build_version_context(
        "notes/profile.md", include_content=True,
    )
    assert "Previous versions of 'notes/profile.md'" in context
    assert "Favorite editor: Vim" in context

    assert memory.rollback("notes/profile.md", 1, change_note="restore v1")
    restored = memory.read("notes/profile.md")
    assert restored is not None
    assert restored.version == 3
    assert restored.content.endswith("Vim")


def test_generated_version_tools_show_old_versions_by_default():
    memory = NavigableMemory(InMemoryBackend())
    memory.write("docs/spec.md", "Spec", "alpha")
    memory.write("docs/spec.md", "Spec", "beta")

    tools = {tool.__name__: tool for tool in memory.create_tools()}

    output = tools["ListVersions"](path="docs/spec.md").run()
    assert "v1" in output
    assert "v2" not in output

    output_with_current = tools["ListVersions"](
        path="docs/spec.md", include_current=True,
    ).run()
    assert "v1" in output_with_current
    assert "v2" in output_with_current

    version_output = tools["ReadVersion"](path="docs/spec.md", version=1).run()
    assert "## Spec (v1)" in version_output
    assert "alpha" in version_output

    diff_output = tools["CompareVersions"](
        path="docs/spec.md", from_version=1,
    ).run()
    assert "-alpha" in diff_output
    assert "+beta" in diff_output


def test_packaged_backends_store_versions_and_roll_back():
    backend_factories = [
        InMemoryBackend,
        lambda: JSONBackend(":memory:"),
        lambda: SQLiteBackend(":memory:"),
    ]

    with tempfile.TemporaryDirectory() as root:
        backend_factories.append(lambda: FilesystemBackend(root))

        for create_backend in backend_factories:
            memory = NavigableMemory(create_backend())
            memory.write("docs/spec.md", "Spec", "alpha", change_note="draft")
            memory.write("docs/spec.md", "Spec", "beta", change_note="revision")

            assert [v.version for v in memory.list_versions("docs/spec.md")] == [2, 1]
            assert memory.get_version("docs/spec.md", 1).content == "alpha"
            assert memory.rollback("docs/spec.md", 1, change_note="restore draft")

            current = memory.read("docs/spec.md")
            assert current is not None
            assert current.version == 3
            assert current.content == "alpha"


def test_persistent_backends_reload_version_history():
    with tempfile.TemporaryDirectory() as root:
        json_path = f"{root}/memory.json"
        sqlite_path = f"{root}/memory.db"
        fs_path = f"{root}/fs-memory"

        persistent_backends = [
            (lambda: JSONBackend(json_path), lambda backend: backend.close()),
            (lambda: SQLiteBackend(sqlite_path), lambda backend: backend.close()),
            (lambda: FilesystemBackend(fs_path), lambda _backend: None),
        ]

        for create_backend, close_backend in persistent_backends:
            backend = create_backend()
            memory = NavigableMemory(backend)
            memory.write("docs/spec.md", "Spec", "alpha")
            memory.write("docs/spec.md", "Spec", "beta")
            close_backend(backend)

            reloaded = NavigableMemory(create_backend())
            assert [v.version for v in reloaded.list_versions("docs/spec.md")] == [
                2,
                1,
            ]
            assert reloaded.get_version("docs/spec.md", 1).content == "alpha"

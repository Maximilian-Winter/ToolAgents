"""Smoke test for navigable_memory binary/versioning/reference support."""
from __future__ import annotations

import io
import os
import sys
import tempfile

# Force UTF-8 stdout on Windows so arrows / box-drawing render
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure src is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ToolAgents.agent_memory.navigable_memory import (  # noqa: E402
    NavigableMemory, InMemoryBackend, SQLiteBackend, JSONBackend,
    FilesystemBackend, RefType, migrate, MigrationReport,
)


# Tiny valid PNG bytes (1x1 transparent pixel)
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63fcffff3f0300050001fff3eef0660000000049454e44"
    "ae426082"
)


def run_suite(name: str, memory: NavigableMemory) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

    # ── Text writes + versioning ──────────────────────────────
    memory.write("docs/intro.md", "Intro", "First version of the intro.")
    memory.write("docs/intro.md", "Intro", "Second version — improved!",
                 change_note="rewrote opening paragraph")
    memory.write("docs/intro.md", "Intro v3", "Third pass.",
                 change_note="title tweak")

    versions = memory.list_versions("docs/intro.md")
    print(f"versions of intro.md: {[v.version for v in versions]}")
    assert [v.version for v in versions] == [3, 2, 1], versions

    v1 = memory.get_version("docs/intro.md", 1)
    assert v1 is not None and v1.content == "First version of the intro.", v1
    print(f"v1 content: {v1.content!r}")

    # Rollback creates v4
    assert memory.rollback("docs/intro.md", 1, change_note="reverting hype")
    cur = memory.read("docs/intro.md")
    assert cur is not None
    assert cur.version == 4
    assert cur.content == "First version of the intro."
    print(f"after rollback: v{cur.version}, content={cur.content!r}")

    # ── Binary support ───────────────────────────────────────
    ok = memory.write_binary(
        "assets/diagrams/arch.png",
        title="Architecture diagram",
        mime_type="image/png",
        data=PNG_BYTES,
        caption="Top-level service layout.",
        tags=["diagram", "architecture"],
    )
    assert ok
    bdoc = memory.read("assets/diagrams/arch.png")
    assert bdoc is not None and bdoc.is_image and bdoc.is_binary
    assert bdoc.size_bytes == len(PNG_BYTES)
    print(f"binary doc: {bdoc.title} ({bdoc.mime_type}, {bdoc.human_size})")

    raw = memory.read_binary("assets/diagrams/arch.png")
    assert raw == PNG_BYTES
    print(f"read_binary returned {len(raw)} bytes - match OK")

    # Update binary → new version
    memory.write_binary(
        "assets/diagrams/arch.png",
        title="Architecture diagram (revised)",
        mime_type="image/png",
        data=PNG_BYTES + b"\x00",  # different size
        caption="Now with v2 layout.",
        change_note="redrew gateway",
    )
    bversions = memory.list_versions("assets/diagrams/arch.png")
    assert [v.version for v in bversions] == [2, 1], bversions
    assert bversions[1].binary_data == PNG_BYTES
    print(f"binary versions: {[v.version for v in bversions]}")

    # ── References ───────────────────────────────────────────
    assert memory.add_reference(
        "docs/intro.md", "assets/diagrams/arch.png",
        ref_type=RefType.EMBEDS, note="see opening figure",
    )
    # Idempotent — second add returns False (no duplicate created)
    duplicate = memory.add_reference(
        "docs/intro.md", "assets/diagrams/arch.png",
        ref_type=RefType.EMBEDS,
    )
    assert duplicate is False, "expected idempotent no-op"

    memory.write("docs/architecture.md", "Architecture",
                 "Detailed architecture writeup.")
    memory.add_reference(
        "docs/architecture.md", "assets/diagrams/arch.png",
        ref_type=RefType.EMBEDS,
    )
    memory.add_reference(
        "docs/intro.md", "docs/architecture.md",
        ref_type=RefType.SEE_ALSO,
    )

    out = memory.references_from("docs/intro.md")
    assert len(out) == 2, out
    incoming = memory.references_to("assets/diagrams/arch.png")
    assert len(incoming) == 2, incoming
    print(f"intro outgoing: {[(r.ref_type, r.to_path) for r in out]}")
    print(f"diagram backlinks: {[(r.ref_type, r.from_path) for r in incoming]}")

    # Remove one specific edge
    n = memory.remove_reference(
        "docs/intro.md", "docs/architecture.md", ref_type=RefType.SEE_ALSO,
    )
    assert n == 1, n
    print(f"removed {n} reference")

    # ── Build context (renders refs + binary nicely) ─────────
    memory.navigate("docs/intro.md")
    ctx = memory.build_context()
    print("\n--- build_context for docs/intro.md ---")
    print(ctx)
    assert "References from here" in ctx, "expected outgoing refs in context"

    memory.navigate("assets/diagrams/arch.png")
    ctx2 = memory.build_context()
    print("\n--- build_context for binary ---")
    print(ctx2)
    assert "image/png" in ctx2
    assert "attachment" in ctx2

    # ── Tools surface the new capabilities ──────────────────
    tools = memory.create_tools()
    tool_names = [t.__name__ for t in tools]
    print(f"\ntools: {tool_names}")
    assert "ListVersions" in tool_names
    assert "RollbackToVersion" in tool_names
    assert "AddReference" in tool_names
    assert "ListReferences" in tool_names
    assert "DescribeBinary" in tool_names

    # Exercise a tool
    AddRef = next(t for t in tools if t.__name__ == "AddReference")
    msg = AddRef(
        from_path="docs/architecture.md",
        to_path="docs/intro.md",
        ref_type=RefType.DEPENDS_ON,
        note="needs intro for context",
    ).run()
    print(f"AddReference tool said: {msg}")

    # Prune
    pruned = memory.prune_versions("docs/intro.md", keep_last_n=2)
    remaining = [v.version for v in memory.list_versions("docs/intro.md")]
    print(f"pruned {pruned} versions, remaining: {remaining}")
    assert pruned >= 1
    assert len(remaining) == 2


def verify_json_persistence(json_path: str) -> None:
    """Reopen the JSON backend and confirm everything survived."""
    print(f"\n{'=' * 60}\n  JSONBackend reload (persistence check)\n{'=' * 60}")
    backend = JSONBackend(json_path)
    memory = NavigableMemory(backend)

    intro = memory.read("docs/intro.md")
    assert intro is not None, "intro.md should have persisted"
    print(f"reloaded intro: v{intro.version}, content={intro.content!r}")

    bdoc = memory.read("assets/diagrams/arch.png")
    assert bdoc is not None and bdoc.is_image
    raw = memory.read_binary("assets/diagrams/arch.png")
    assert raw is not None and len(raw) == bdoc.size_bytes
    print(f"reloaded binary: {bdoc.title} ({bdoc.human_size}), bytes restored OK")

    versions = memory.list_versions("docs/intro.md")
    print(f"reloaded versions of intro.md: {[v.version for v in versions]}")
    assert len(versions) >= 2

    refs_out = memory.references_from("docs/intro.md")
    print(f"reloaded outgoing refs from intro: {[(r.ref_type, r.to_path) for r in refs_out]}")
    assert len(refs_out) >= 1


def inspect_json_file(json_path: str) -> None:
    """Show a fragment of the on-disk JSON to prove it's human-readable."""
    print(f"\n{'=' * 60}\n  JSON file inspection\n{'=' * 60}")
    size = os.path.getsize(json_path)
    print(f"file size: {size} bytes")
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(800)
    print(f"first 800 chars:\n{head}")
    if size > 800:
        print("...[truncated]")


def main() -> None:
    # In-memory
    run_suite("InMemoryBackend", NavigableMemory(InMemoryBackend()))

    # SQLite (temp file)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        backend = SQLiteBackend(db_path)
        try:
            run_suite("SQLiteBackend", NavigableMemory(backend))
            stats = backend.stats()
            print(f"\nSQLite stats: {stats}")
        finally:
            backend.close()

    # JSON backend (temp file, with reload check)
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "memory.json")
        backend = JSONBackend(json_path)
        run_suite("JSONBackend", NavigableMemory(backend))
        stats = backend.stats()
        print(f"\nJSON stats: {stats}")
        backend.close()
        # Reopen the same file in a brand new backend instance:
        verify_json_persistence(json_path)
        inspect_json_file(json_path)

    # Filesystem backend
    with tempfile.TemporaryDirectory() as tmp:
        fs_root = os.path.join(tmp, "agent_memory")
        backend = FilesystemBackend(fs_root)
        run_suite("FilesystemBackend", NavigableMemory(backend))
        stats = backend.stats()
        print(f"\nFilesystem stats: {stats}")
        verify_filesystem_layout(fs_root)
        verify_filesystem_persistence(fs_root)

    # Migration matrix: prove data flows correctly between every backend
    run_migration_matrix()

    print("\n*** ALL SMOKE TESTS PASSED ***")


def verify_filesystem_layout(root: str) -> None:
    """Confirm the filesystem backend wrote pristine, openable files."""
    print(f"\n{'=' * 60}\n  Filesystem layout inspection\n{'=' * 60}")
    intro_md = os.path.join(root, "docs", "intro.md")
    arch_png = os.path.join(root, "assets", "diagrams", "arch.png")
    refs_json = os.path.join(root, ".navmem", "references.json")
    assert os.path.isfile(intro_md), f"missing {intro_md}"
    assert os.path.isfile(arch_png), f"missing {arch_png}"
    assert os.path.isfile(refs_json), f"missing {refs_json}"

    with open(intro_md, "r", encoding="utf-8") as f:
        body = f.read()
    print(f"docs/intro.md (pure markdown, no frontmatter):\n  {body!r}")
    assert body == "First version of the intro.", "expected post-rollback content"

    with open(arch_png, "rb") as f:
        head = f.read(8)
    assert head[:8] == bytes.fromhex("89504e470d0a1a0a"), f"PNG magic missing: {head!r}"
    print(f"assets/diagrams/arch.png starts with PNG magic bytes: OK")

    print("on-disk tree:")
    for dirpath, dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        indent = "  " * (rel.count(os.sep) + (0 if rel == "." else 1))
        if rel != ".":
            print(f"{indent}{os.path.basename(dirpath)}/")
        for fname in sorted(files):
            inner_indent = "  " * (rel.count(os.sep) + (1 if rel == "." else 2))
            print(f"{inner_indent}{fname}")


def verify_filesystem_persistence(root: str) -> None:
    """Reopen the filesystem backend and verify state is intact."""
    print(f"\n{'=' * 60}\n  FilesystemBackend reload\n{'=' * 60}")
    backend = FilesystemBackend(root)
    memory = NavigableMemory(backend)

    intro = memory.read("docs/intro.md")
    assert intro is not None, "intro.md should have persisted"
    print(f"reloaded intro: v{intro.version}")

    raw = memory.read_binary("assets/diagrams/arch.png")
    assert raw is not None and len(raw) > 0
    print(f"reloaded binary: {len(raw)} bytes")

    versions = memory.list_versions("docs/intro.md")
    print(f"reloaded versions: {[v.version for v in versions]}")
    assert len(versions) >= 2

    refs_out = memory.references_from("docs/intro.md")
    print(f"reloaded outgoing refs: {[(r.ref_type, r.to_path) for r in refs_out]}")
    assert len(refs_out) >= 1


def populate(memory: NavigableMemory) -> None:
    """Seed a backend with text + binary + versions + references."""
    memory.write("docs/intro.md", "Intro", "first cut")
    memory.write("docs/intro.md", "Intro", "second cut", change_note="rewrite")
    memory.write("docs/architecture.md", "Architecture", "system overview")
    memory.write_binary(
        "assets/logo.png", "Logo", "image/png", PNG_BYTES,
        caption="Project logo",
    )
    memory.add_reference(
        "docs/intro.md", "assets/logo.png", ref_type=RefType.EMBEDS,
    )
    memory.add_reference(
        "docs/intro.md", "docs/architecture.md", ref_type=RefType.SEE_ALSO,
    )


def assert_equivalent(src: NavigableMemory, dst: NavigableMemory, label: str) -> None:
    """Assert that dst received the essential state from src."""
    src_paths = sorted(d.path for d in src.list_at(""))
    dst_paths = sorted(d.path for d in dst.list_at(""))
    assert src_paths == dst_paths, f"{label}: {src_paths} != {dst_paths}"

    # Binary content survives if both sides support binary
    src_bin = src.read_binary("assets/logo.png")
    dst_bin = dst.read_binary("assets/logo.png")
    if src_bin is not None:
        assert src_bin == dst_bin, f"{label}: binary mismatch"

    src_refs = sorted(
        (r.from_path, r.to_path, r.ref_type) for r in src.all_references()
    )
    dst_refs = sorted(
        (r.from_path, r.to_path, r.ref_type) for r in dst.all_references()
    )
    if src_refs:
        assert src_refs == dst_refs, f"{label}: {src_refs} != {dst_refs}"

    # Latest content of intro
    src_intro = src.read("docs/intro.md")
    dst_intro = dst.read("docs/intro.md")
    assert src_intro is not None and dst_intro is not None
    assert src_intro.content == dst_intro.content, f"{label}: intro content mismatch"


def run_migration_matrix() -> None:
    """Round-trip data through every backend pair."""
    print(f"\n{'=' * 60}\n  Migration matrix\n{'=' * 60}")

    with tempfile.TemporaryDirectory() as tmp:
        # Seed an InMemoryBackend
        seed = NavigableMemory(InMemoryBackend())
        populate(seed)

        sqlite_path = os.path.join(tmp, "out.db")
        json_path = os.path.join(tmp, "out.json")
        fs_root = os.path.join(tmp, "fs_root")

        sqlite_backend = SQLiteBackend(sqlite_path)
        json_backend = JSONBackend(json_path)
        fs_backend = FilesystemBackend(fs_root)

        try:
            # InMemory → SQLite
            r1 = migrate(seed.backend, sqlite_backend)
            print(f"InMemory → SQLite: {r1}")
            assert r1.ok, r1.errors
            assert_equivalent(seed, NavigableMemory(sqlite_backend), "InMemory→SQLite")

            # SQLite → JSON
            r2 = migrate(sqlite_backend, json_backend)
            print(f"SQLite → JSON:    {r2}")
            assert r2.ok, r2.errors
            assert_equivalent(seed, NavigableMemory(json_backend), "SQLite→JSON")

            # JSON → Filesystem
            r3 = migrate(json_backend, fs_backend)
            print(f"JSON → Filesystem: {r3}")
            assert r3.ok, r3.errors
            assert_equivalent(seed, NavigableMemory(fs_backend), "JSON→Filesystem")

            # Filesystem → fresh InMemory (round trip back home)
            home = InMemoryBackend()
            r4 = migrate(fs_backend, home)
            print(f"Filesystem → InMemory: {r4}")
            assert r4.ok, r4.errors
            assert_equivalent(seed, NavigableMemory(home), "Filesystem→InMemory")

            # Idempotency check: running migrate again with overwrite=False
            # should produce mostly skips
            r5 = migrate(seed.backend, sqlite_backend)
            print(f"Re-run (no overwrite): {r5}")
            assert r5.skipped > 0, "expected skips on re-run"
            assert r5.documents == 0 and r5.binaries == 0
        finally:
            sqlite_backend.close()


if __name__ == "__main__":
    main()

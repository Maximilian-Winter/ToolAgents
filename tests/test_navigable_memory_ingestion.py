from pathlib import Path
from types import SimpleNamespace

from ToolAgents.agent_memory.navigable_memory import (
    FileIngestionConfig,
    InMemoryBackend,
    NavigableMemory,
    build_navigable_memory_skill_prompt,
    create_ingestion_tools,
    create_llm_ingestion_transformer,
    ingest_directory,
    ingest_file,
)


class TrackingSemanticIndex:
    def __init__(self):
        self.memory = None
        self.indexed = []
        self.removed = []
        self.rebuilt = []

    def attach_memory(self, memory):
        self.memory = memory

    def index_document(self, path, memory=None):
        self.indexed.append(path)
        return True

    def remove_document(self, path):
        self.removed.append(path)
        return 1

    def rebuild(self, memory=None, prefix=""):
        self.rebuilt.append(prefix)
        return len((memory or self.memory).list_at(prefix))

    def search(self, query, k=8, path_prefix=None, tags=None):
        return []

    def build_search_context(self, query, k=5, path_prefix=None, tags=None, max_chars=320):
        return ""


def test_ingest_file_writes_original_text(tmp_path):
    source = tmp_path / "notes" / "alpha.md"
    source.parent.mkdir()
    source.write_text("# Alpha Note\n\nImportant memory text.", encoding="utf-8")
    memory = NavigableMemory(InMemoryBackend())

    result = ingest_file(
        memory,
        source,
        config=FileIngestionConfig(path_prefix="imported", tags=["seed"]),
    )

    doc = memory.read("imported/alpha.md")
    assert result.status == "written"
    assert result.memory_path == "imported/alpha.md"
    assert doc.title == "Alpha Note"
    assert doc.content == "# Alpha Note\n\nImportant memory text."
    assert doc.tags == ["seed"]
    assert doc.metadata["source_name"] == "alpha.md"


def test_ingest_directory_filters_and_preserves_relative_paths(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "keep.txt").write_text("keep text", encoding="utf-8")
    (root / "skip.bin").write_bytes(b"\x00\x01")
    (root / ".hidden.md").write_text("hidden", encoding="utf-8")
    memory = NavigableMemory(InMemoryBackend())

    report = ingest_directory(
        memory,
        root,
        config=FileIngestionConfig(path_prefix="vault", extensions=(".txt",)),
    )

    assert report.scanned == 1
    assert report.written == 1
    assert memory.read("vault/sub/keep.txt").content == "keep text"
    assert memory.read("vault/skip.bin") is None
    assert memory.read("vault/.hidden.md") is None


def test_ingest_respects_overwrite_false(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("first", encoding="utf-8")
    memory = NavigableMemory(InMemoryBackend())

    assert ingest_file(memory, source).status == "written"
    source.write_text("second", encoding="utf-8")
    result = ingest_file(
        memory,
        source,
        config=FileIngestionConfig(overwrite=False),
    )

    assert result.status == "skipped"
    assert memory.read("a.txt").content == "first"


def test_ingestion_transformer_can_rewrite_fields(tmp_path):
    source = tmp_path / "raw.txt"
    source.write_text("raw source", encoding="utf-8")
    memory = NavigableMemory(InMemoryBackend())

    def transform(item):
        return {
            "title": "Cleaned",
            "content": item.content.upper(),
            "tags": item.tags + ["cleaned"],
            "metadata": {**item.metadata, "mode": "transform"},
        }

    result = ingest_file(
        memory,
        source,
        config=FileIngestionConfig(tags=["raw"]),
        transform=transform,
    )
    doc = memory.read(result.memory_path)

    assert doc.title == "Cleaned"
    assert doc.content == "RAW SOURCE"
    assert doc.tags == ["raw", "cleaned"]
    assert doc.metadata["mode"] == "transform"


def test_llm_ingestion_transformer_uses_agent_response(tmp_path):
    source = tmp_path / "raw.txt"
    source.write_text("raw source", encoding="utf-8")
    memory = NavigableMemory(InMemoryBackend())

    class FakeAgent:
        def __init__(self):
            self.prompts = []

        def get_response(self, messages, settings=None):
            self.prompts.append(messages[0].get_as_text())
            return SimpleNamespace(response="LLM cleaned content")

    agent = FakeAgent()
    result = ingest_file(memory, source, transform=create_llm_ingestion_transformer(agent))

    assert memory.read(result.memory_path).content == "LLM cleaned content"
    assert "raw source" in agent.prompts[0]


def test_ingestion_tools_write_directory_and_rebuild_semantic_index(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nalpha", encoding="utf-8")
    index = TrackingSemanticIndex()
    memory = NavigableMemory(InMemoryBackend(), semantic_index=index)

    tools = {tool.__name__: tool for tool in create_ingestion_tools(
        memory,
        config=FileIngestionConfig(path_prefix="base"),
        allowed_root=tmp_path,
    )}

    output = tools["IngestTextDirectory"](
        directory_path=str(root),
        memory_prefix="tool",
        extensions=[".md"],
    ).run()
    rebuild = tools["RebuildNavigableSemanticIndex"](path_prefix="tool").run()

    assert "written=1" in output
    assert memory.read("tool/a.md").title == "A"
    assert index.indexed == ["tool/a.md"]
    assert "1 document" in rebuild
    assert index.rebuilt == ["tool"]


def test_ingestion_tools_enforce_allowed_root(tmp_path):
    memory = NavigableMemory(InMemoryBackend())
    tools = {tool.__name__: tool for tool in create_ingestion_tools(
        memory,
        allowed_root=tmp_path / "allowed",
    )}

    result = tools["IngestTextFile"](file_path=str(tmp_path / "outside.txt")).run()

    assert "outside allowed root" in result


def test_skill_prompt_mentions_selected_capabilities():
    prompt = build_navigable_memory_skill_prompt(
        include_ingestion=True,
        include_semantic=True,
        include_versions=True,
        include_references=True,
    )

    assert "semantic or hybrid search" in prompt
    assert "Follow references" in prompt
    assert "version tools" in prompt
    assert "ingestion tools" in prompt

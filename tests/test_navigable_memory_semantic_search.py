from types import SimpleNamespace

from ToolAgents.agent_memory.navigable_memory import (
    Document,
    InMemoryBackend,
    NavigableMemory,
    NavigableSemanticIndex,
    RefType,
)


class FakeVectorProvider:
    def __init__(self):
        self.ids = []
        self.chunks = []
        self.metadata = []

    def add_texts_with_id(self, ids, texts, metadata):
        self.ids.extend(ids)
        self.chunks.extend(texts)
        self.metadata.extend(metadata)

    def remove_by_ids(self, ids):
        drop = set(ids)
        keep = [idx for idx, chunk_id in enumerate(self.ids) if chunk_id not in drop]
        self.ids = [self.ids[idx] for idx in keep]
        self.chunks = [self.chunks[idx] for idx in keep]
        self.metadata = [self.metadata[idx] for idx in keep]

    def query(self, query, query_filter=None, k=3, **kwargs):
        terms = {term.lower() for term in query.split()}
        scored = []
        for idx, chunk in enumerate(self.chunks):
            haystack = set(chunk.lower().split())
            score = float(len(terms & haystack))
            if query.lower() in chunk.lower():
                score += 2.0
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [item for item in scored if item[0] > 0][:k]
        return SimpleNamespace(
            ids=[self.ids[idx] for _, idx in top],
            chunks=[self.chunks[idx] for _, idx in top],
            scores=[score for score, _ in top],
            metadata=[self.metadata[idx] for _, idx in top],
        )

    def get_all_entries(self):
        return SimpleNamespace(
            ids=list(self.ids),
            chunks=list(self.chunks),
            embeddings=[],
            metadata=list(self.metadata),
        )


class BasicBackend:
    def __init__(self):
        self.docs = {}

    def read(self, path):
        return self.docs.get(path)

    def write(self, path, title, content, tags=None, metadata=None):
        self.docs[path] = Document(
            path=path,
            title=title,
            content=content,
            tags=tags or [],
            metadata=metadata or {},
        )
        return True

    def list(self, prefix=""):
        return [
            doc for path, doc in sorted(self.docs.items())
            if path.startswith(prefix)
        ]

    def search(self, query):
        query = query.lower()
        return [
            doc for doc in self.docs.values()
            if query in doc.title.lower()
            or query in doc.content.lower()
            or any(query in tag.lower() for tag in doc.tags)
        ]

    def delete(self, path):
        return self.docs.pop(path, None) is not None


def make_memory(include_binary_captions=False):
    backend = InMemoryBackend()
    provider = FakeVectorProvider()
    index = NavigableSemanticIndex(
        provider,
        include_binary_captions=include_binary_captions,
    )
    memory = NavigableMemory(backend, semantic_index=index)
    return memory, provider


def test_semantic_index_rebuild_and_result_metadata():
    memory, provider = make_memory()
    memory.write(
        "projects/ashenmoor/design.md",
        "Ashenmoor Design",
        "Thornqueen encounter needs mist phase and stagger tuning.",
        tags=["design", "boss"],
    )
    memory.write(
        "projects/forge/ops.md",
        "Forge Ops",
        "Build pipeline status and release checklist.",
        tags=["ops"],
    )

    indexed = memory.semantic_index.rebuild()
    results = memory.semantic_search("Thornqueen mist", k=3)

    assert indexed == 2
    assert provider.ids
    assert results[0].path == "projects/ashenmoor/design.md"
    assert results[0].title == "Ashenmoor Design"
    assert results[0].version == 1
    assert results[0].tags == ["design", "boss"]
    assert results[0].metadata["chunk_index"] == 0


def test_semantic_index_lifecycle_updates_delete_and_rollback():
    memory, provider = make_memory()
    memory.write("notes/a.md", "A", "alpha beta", tags=["one"])
    first_ids = list(provider.ids)

    assert memory.semantic_search("alpha", k=1)[0].path == "notes/a.md"

    memory.append("notes/a.md", "gamma delta")
    assert provider.ids != first_ids
    assert memory.semantic_search("gamma", k=1)[0].version == 2

    memory.write("notes/a.md", "A", "replacement epsilon", tags=["two"])
    assert memory.semantic_search("epsilon", k=1)[0].tags == ["two"]

    assert memory.rollback("notes/a.md", 1)
    rolled_back = memory.semantic_search("alpha", k=1)[0]
    assert rolled_back.path == "notes/a.md"
    assert rolled_back.version == 4

    assert memory.delete("notes/a.md")
    assert memory.semantic_search("alpha", k=1) == []
    assert provider.ids == []


def test_binary_caption_indexing_is_opt_in():
    memory, provider = make_memory(include_binary_captions=False)
    memory.write_binary(
        "assets/map.bin",
        "Map",
        "application/octet-stream",
        b"fixed-bytes",
        caption="orbital refinery layout",
    )
    assert provider.ids == []
    assert memory.semantic_search("orbital", k=1) == []

    memory, provider = make_memory(include_binary_captions=True)
    memory.write_binary(
        "assets/map.bin",
        "Map",
        "application/octet-stream",
        b"fixed-bytes",
        caption="orbital refinery layout",
    )
    result = memory.semantic_search("orbital", k=1)[0]
    assert result.path == "assets/map.bin"
    assert result.mime_type == "application/octet-stream"


def test_hybrid_search_combines_semantic_lexical_tags_and_references():
    memory, _ = make_memory()
    memory.write("current.md", "Current", "Current planning note.", tags=["hub"])
    memory.write("a.md", "A", "Thornqueen smoke and mist mechanics.", tags=["boss"])
    memory.write("b.md", "B", "Release train schedule.", tags=["release"])
    memory.add_reference("current.md", "b.md", RefType.SEE_ALSO)
    memory.navigate("current.md")

    results = memory.hybrid_search(
        "Thornqueen release", k=3, tags=None, include_references=True,
    )
    by_path = {result.path: result for result in results}

    assert "a.md" in by_path
    assert "b.md" in by_path
    assert "semantic" in by_path["a.md"].metadata["sources"]
    assert "reference" in by_path["b.md"].metadata["sources"]


def test_generated_semantic_tools_are_conditional():
    memory = NavigableMemory(InMemoryBackend())
    tool_names = {tool.__name__ for tool in memory.create_tools()}
    assert "SemanticSearchKnowledge" not in tool_names
    assert "HybridSearchKnowledge" not in tool_names

    memory, _ = make_memory()
    memory.write("docs/concept.md", "Concept", "Semantic navigation target.")
    tool_map = {tool.__name__: tool for tool in memory.create_tools()}
    assert "SemanticSearchKnowledge" in tool_map
    assert "HybridSearchKnowledge" in tool_map

    output = tool_map["SemanticSearchKnowledge"](query="navigation").run()
    assert "docs/concept.md" in output
    assert "Navigate to: docs/concept.md" in output


def test_create_tools_names_match_backend_capabilities():
    base_tools = {
        "Navigate",
        "NavigateUp",
        "ListLocations",
        "SearchKnowledge",
        "ReadDocument",
        "WriteDocument",
        "AppendToDocument",
        "ListTags",
        "FindByTag",
        "FindByTags",
        "AddTags",
        "RemoveTags",
        "SetTags",
    }
    storage_tools = {
        "ListVersions",
        "ReadVersion",
        "CompareVersions",
        "ShowVersionContext",
        "RollbackToVersion",
        "AddReference",
        "RemoveReference",
        "ListReferences",
        "FollowReferences",
        "DescribeBinary",
    }
    semantic_tools = {
        "SemanticSearchKnowledge",
        "HybridSearchKnowledge",
    }

    basic_names = {
        tool.__name__
        for tool in NavigableMemory(BasicBackend()).create_tools()
    }
    assert basic_names == base_tools

    capable_names = {
        tool.__name__
        for tool in NavigableMemory(InMemoryBackend()).create_tools()
    }
    assert capable_names == base_tools | storage_tools

    memory, _ = make_memory()
    semantic_names = {tool.__name__ for tool in memory.create_tools()}
    assert semantic_names == base_tools | storage_tools | semantic_tools

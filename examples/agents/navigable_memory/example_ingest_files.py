"""Ingest local text files into NavigableMemory.

This example uses a temporary mini-vault so it leaves no persistent files behind.
For a real project, pass your own directory to ``ingest_directory`` and choose
the backend you want to keep.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ToolAgents.agent_memory.navigable_memory import (
    FileIngestionConfig,
    InMemoryBackend,
    NavigableMemory,
    build_navigable_memory_skill_prompt,
    create_ingestion_tools,
    ingest_directory,
)


def uppercase_summary(source):
    """Tiny stand-in for an LLM transformer."""
    return {
        "title": f"Imported: {source.title}",
        "content": source.content.upper(),
        "metadata": {**source.metadata, "transform": "uppercase_summary"},
    }


if __name__ == "__main__":
    memory = NavigableMemory(InMemoryBackend())

    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"
        vault.mkdir()
        (vault / "overview.md").write_text(
            "# Forge Notes\n\nThe build has QA and art follow-up.",
            encoding="utf-8",
        )
        (vault / "qa.txt").write_text(
            "Thornqueen visibility blocker needs VFX review.",
            encoding="utf-8",
        )

        report = ingest_directory(
            memory,
            vault,
            config=FileIngestionConfig(path_prefix="imported", tags=["imported"]),
        )
        print(report.summary())

        transformed = ingest_directory(
            memory,
            vault,
            config=FileIngestionConfig(path_prefix="summaries", extensions=(".txt",)),
            transform=uppercase_summary,
        )
        print(transformed.summary())

        tools = create_ingestion_tools(
            memory,
            config=FileIngestionConfig(path_prefix="tool-imports"),
            allowed_root=vault,
        )
        print([tool.__name__ for tool in tools])

    print(memory.navigate("imported/overview.md"))
    print(memory.build_context())
    print(build_navigable_memory_skill_prompt(include_ingestion=True))

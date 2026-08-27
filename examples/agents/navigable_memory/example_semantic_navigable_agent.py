"""Composable NavigableMemory example with optional semantic search.

This is the preferred pattern for new memory-enabled agents:

    ChatToolAgent + NavigableMemory + optional NavigableSemanticIndex

The semantic index uses the existing vector database abstractions. Install the
optional memory dependencies before running the semantic path:

    pip install "ToolAgents[memory]"

Set OPENROUTER_API_KEY to run an LLM turn. Without an API key, the script still
demonstrates indexing and semantic/hybrid search.
"""

from __future__ import annotations

import os

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.agent_memory.navigable_memory import (
    InMemoryBackend,
    NavigableMemory,
    NavigableSemanticIndex,
    RefType,
)
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI


def build_memory() -> NavigableMemory:
    memory = NavigableMemory(InMemoryBackend())
    memory.write(
        "studio/projects/ashenmoor/design/bosses.md",
        "Ashenmoor Bosses",
        "The Thornqueen boss fight depends on readable mist phase cues.",
        tags=["ashenmoor", "design", "boss"],
    )
    memory.write(
        "studio/projects/ashenmoor/qa/blockers.md",
        "Ashenmoor QA Blockers",
        "QA is tracking Thornqueen visibility bugs and controller remapping.",
        tags=["ashenmoor", "qa", "blocker"],
    )
    memory.write(
        "studio/projects/ashenmoor/art/vfx.md",
        "Ashenmoor VFX",
        "VFX backlog includes mist silhouettes, root burst timing, and hit sparks.",
        tags=["ashenmoor", "art", "vfx"],
    )
    memory.add_reference(
        "studio/projects/ashenmoor/qa/blockers.md",
        "studio/projects/ashenmoor/art/vfx.md",
        RefType.DEPENDS_ON,
        "Visibility bugs need VFX follow-up.",
    )
    return memory


def attach_semantic_index(memory: NavigableMemory) -> bool:
    try:
        from ToolAgents.knowledge.vector_database.implementations.chroma_db_vector_database import (
            ChromaDbVectorDatabaseProvider,
        )
        from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import (
            SentenceTransformerEmbeddingProvider,
        )
    except ImportError as exc:
        print(f"Semantic index disabled: {exc}")
        return False

    provider = ChromaDbVectorDatabaseProvider(
        SentenceTransformerEmbeddingProvider(),
        persistent=False,
    )
    index = NavigableSemanticIndex(provider, scores_are_distances=True)
    memory.set_semantic_index(index)
    indexed = index.rebuild()
    print(f"Semantic index ready: {indexed} documents indexed.")
    return True


def run_agent(memory: NavigableMemory) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return

    api = OpenAIChatAPI(
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", "xiaomi/mimo-v2.5-pro"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    agent = ChatToolAgent(chat_api=api)
    registry = ToolRegistry()
    registry.add_tools([FunctionTool(tool) for tool in memory.create_tools()])

    messages = [
        ChatMessage.create_system_message(
            "You are a studio assistant. Use navigable memory tools for evidence."
        ),
        ChatMessage.create_user_message(
            "Find Thornqueen blockers and follow related references."
        ),
    ]
    response = agent.get_response(messages, tool_registry=registry)
    print(response.response)


if __name__ == "__main__":
    nav_memory = build_memory()
    semantic_enabled = attach_semantic_index(nav_memory)

    print(nav_memory.search("Thornqueen"))
    if semantic_enabled:
        print(nav_memory.build_semantic_search_context("mist visibility blocker"))
        print(nav_memory.hybrid_search("Thornqueen visibility follow-up", k=3))

    run_agent(nav_memory)

---
title: Memory
---

# Memory

ToolAgents has lightweight retrieval examples under `examples/agents/memory/`
and the preferred memory-agent examples under `examples/agents/navigable_memory/`.

Representative examples:

- `examples/agents/memory/rag.py`
- `examples/agents/memory/bm_25.py`
- `examples/agents/memory/ensemble.py`
- `examples/agents/navigable_memory/example_ingest_files.py`
- `examples/agents/navigable_memory/example_semantic_navigable_agent.py`

Optional dependencies:

- install `ToolAgents[memory]` for vector-store and embedding-provider support

Notes:

- the memory surface is still optional and heavier than the core framework
- legacy `SemanticMemory` and `ContextAppState` examples were removed from the maintained examples tree
- prefer `NavigableMemory` with an optional semantic index for document retrieval
- `NavigableMemory` includes helpers for ingesting local text files/directories and opt-in ingestion tools for agents
- generated local vector-store artifacts are intentionally excluded from the maintained repo state

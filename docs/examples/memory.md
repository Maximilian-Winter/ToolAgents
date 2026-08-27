---
title: Memory
---

# Memory

ToolAgents has optional semantic-memory and retrieval examples under `examples/agents/memory/`.

Representative examples:

- `examples/agents/memory/rag.py`
- `examples/agents/memory/ensemble.py`
- `examples/agents/navigable_memory/example_semantic_navigable_agent.py`
- `examples/agents/memory/context_app_state_test.py`

Optional dependencies:

- install `ToolAgents[memory]` for vector-store and embedding-provider support
- install `ToolAgents[advanced]` if you also want YAML-backed app-state helpers

Notes:

- the memory surface is still optional and heavier than the core framework
- `SemanticMemory` is legacy; prefer `NavigableMemory` with an optional semantic index for document retrieval
- generated local vector-store artifacts are intentionally excluded from the maintained repo state

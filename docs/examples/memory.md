---
title: Memory
---

# Memory

ToolAgents has optional semantic-memory and retrieval examples under `examples/agents/memory/`.

Representative examples:

- `examples/agents/memory/rag.py`
- `examples/agents/memory/ensemble.py`
- `examples/agents/memory/semantic_memory_test.py`
- `examples/agents/memory/context_app_state_test.py`

Optional dependencies:

- install `ToolAgents[memory]` for semantic memory and vector-store support
- install `ToolAgents[advanced]` if you also want YAML-backed app-state helpers

Notes:

- the memory surface is still optional and heavier than the core framework
- generated local vector-store artifacts are intentionally excluded from the maintained repo state

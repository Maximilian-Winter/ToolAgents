---
title: Advanced Agents
---

# Advanced Agents

The curated advanced examples currently focus on higher-level workflows built on top of `ChatToolAgent` and `AdvancedAgent`.

Representative examples in the repo:

- `examples/agents/advanced/example_user_loop.py`
- `examples/agents/advanced/example_user_loop_streaming.py`
- `examples/agents/advanced/structured_output_agent.py`
- `examples/agents/advanced/output_knowledge_graph.py`

What these examples cover:

- stateful user loops on top of the cleaned message and provider APIs
- structured output flows
- higher-level orchestration patterns that sit above the base chat/tool agent

Notes:

- OCR-specific examples were intentionally retired from the maintained example surface.
- Treat the examples in `examples/advanced_agent/` as exploratory unless they are explicitly referenced by the current docs.

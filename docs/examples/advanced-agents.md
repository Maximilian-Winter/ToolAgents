---
title: Advanced Agent Examples
---

# Advanced Agent Examples

The curated advanced examples focus on workflows built on top of
`ChatToolAgent`. For longer-running interactive assistants, prefer the
`AgentHarness` examples in `examples/agents/context_and_harness/`.

Representative examples in the repo:

- `examples/agents/advanced/example_user_loop.py`
- `examples/agents/advanced/example_user_loop_streaming.py`
- `examples/agents/advanced/structured_output_agent.py`
- `examples/agents/advanced/output_knowledge_graph.py`

What these examples cover:

- stateful user loops on top of the cleaned message and provider APIs
- structured output flows
- small orchestration patterns that sit above the base chat/tool agent

Notes:

- OCR-specific examples were intentionally retired from the maintained example surface.
- Legacy `AdvancedAgent`/`AgentConfig` examples were removed from the
  maintained examples tree. Use older release tags or source archives for
  historical compatibility references. See [Removed Legacy APIs](../removed-legacy.md)
  for replacement guidance.

---
title: Removed Legacy APIs
---

# Removed Legacy APIs

ToolAgents keeps the maintained API surface focused on explicit agent
composition, current provider adapters, and `NavigableMemory`.

The following legacy surfaces were removed from the maintained package source
or public exports:

- `ToolAgents.agents.advanced_agent.AdvancedAgent`
- `ToolAgents.agents.advanced_agent.AgentConfig`

!!! note "Not to be confused with `ToolAgents.pipelines.AgentConfig`"

    A new, unrelated `AgentConfig` exists in `ToolAgents.pipelines`. It is a
    declarative description of a provider and model for a
    [pipeline](guides/pipelines.md#declaring-agents-and-endpoints) — nothing to
    do with the removed `advanced_agent.AgentConfig`.
- `ToolAgents.agent_memory.semantic_memory`
- public `ToolAgents.agent_memory.ContextAppState` re-export
- legacy `SemanticMemory` examples
- legacy `AdvancedAgent` examples
- legacy Gradio `AdvancedAgent` demo

`ContextAppState` remains only as historical support code for archived examples
that have not yet been rewritten. It is not part of the maintained public memory
surface.

Use these current APIs instead:

- `ChatToolAgent` or `AsyncChatToolAgent` for direct tool-calling agents
- `AgentHarness` for higher-level agent loops, prompt composition, events, and
  interactive workflows
- `NavigableMemory` for maintained memory workflows
- `NavigableSemanticIndex` plus the `memory` extra for optional semantic search
- `Pipeline` with flow-control processes for declarative multi-step workflows —
  the closest modern equivalent to what `AdvancedAgent` was reached for

Historical source archives and older Git tags still contain the removed code for
projects that need a compatibility reference.

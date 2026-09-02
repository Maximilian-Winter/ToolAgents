# ToolAgents Examples

The maintained examples are grouped by the current APIs they demonstrate.

## Start Here

- `cli/`: three runnable `.tool-agents` workspaces for the `tool-agents` CLI,
  from a single-step workflow to one using tools, prompts, providers, custom
  adapters, sources and sinks together.

- `agents/basics/`: small `ChatToolAgent`, tool-calling, streaming, async, and chat-history examples.
- `agents/context_and_harness/`: preferred higher-level agent loops using `AgentHarness`.
- `agents/navigable_memory/`: preferred memory examples using `NavigableMemory`, including optional semantic search.
- `agents/model_context_protocol/`: MCP tool integration examples.
- `agents/prompt_composer/`: prompt composition examples.

## Optional Or Experimental

- `agents/gradio/`: UI demos that require Gradio and provider credentials.
- `agents/mem_gpt_like/`: exploratory memory-agent pattern.
- `agents/personal_agent_ada/`: larger personal-agent demo.
- `agents/pipeline/`: pipeline examples, including JSON-defined pipelines
  with flow control (map/parallel/loop/conditional), declarative provider
  endpoints, and file/folder sources with file, stream and HTTP sinks.

## Legacy Surface

Old `AdvancedAgent`, `AgentConfig`, `SemanticMemory`, and `ContextAppState`
examples have been removed from the maintained examples tree. They remain
available from older release tags and source archives for users who still need
the historical compatibility examples. See the docs page
`docs/removed-legacy.md` for the current replacement guidance.

`agents/virtual_game_master/` is an archived large demo that still references
legacy state APIs. It is kept as historical application code, not as a
recommended example to copy.

New examples should use explicit `ChatToolAgent` or `AgentHarness` composition
plus `NavigableMemory` where memory is needed.

Generated local outputs such as chat histories, vector-store databases, graph
renderings, and `__pycache__` directories should stay untracked.

# ToolAgents Examples

The maintained examples are grouped by the current APIs they demonstrate.

## Start Here

- `agents/basics/`: small `ChatToolAgent`, tool-calling, streaming, async, and chat-history examples.
- `agents/context_and_harness/`: preferred higher-level agent loops using `AgentHarness`.
- `agents/navigable_memory/`: preferred memory examples using `NavigableMemory`, including optional semantic search.
- `agents/model_context_protocol/`: MCP tool integration examples.
- `agents/prompt_composer/`: prompt composition examples.

## Optional Or Experimental

- `agents/gradio/`: UI demos that require Gradio and provider credentials.
- `agents/mem_gpt_like/`: exploratory memory-agent pattern.
- `agents/personal_agent_ada/`: larger personal-agent demo.
- `agents/pipeline/`: provider-specific pipeline sketches.
- `agents/virtual_game_master/`: large application demo with its own local service and data model.

## Legacy Surface

Old `AdvancedAgent`, `AgentConfig`, `SemanticMemory`, and `ContextAppState`
examples are kept only as compatibility references when they are still present
in the tree. They are not part of the maintained example path. New examples
should use explicit
`ChatToolAgent` or `AgentHarness` composition plus `NavigableMemory` where
memory is needed.

Generated local outputs such as chat histories, vector-store databases, graph
renderings, and `__pycache__` directories should stay untracked.

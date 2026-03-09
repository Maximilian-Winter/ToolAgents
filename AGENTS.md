# ToolAgents Project Guide

## Overview

ToolAgents is a Python framework for building tool-calling LLM agents on top of multiple provider backends.

Current package metadata:
- Name: `ToolAgents`
- Version: `0.3.0`
- Python: `>=3.10`
- Source layout: `src/ToolAgents`

The framework is centered on a provider-neutral message model, tool registry, and agent abstractions.

## Current Architecture

Primary package areas:
- `src/ToolAgents/agents`: agent implementations such as `ChatToolAgent`, `AsyncChatToolAgent`, `StructuredOutputAgent`, and the higher-level `AdvancedAgent`
- `src/ToolAgents/data_models`: core message/history/response models
- `src/ToolAgents/provider`: chat API providers, provider settings, completion-provider support
- `src/ToolAgents/knowledge`: retrieval and text-processing utilities
- `src/ToolAgents/utilities`: prompt/message helpers, logging, docs generation, and database-backed chat storage
- `src/ToolAgents/function_tool.py`: `FunctionTool` and `ToolRegistry`

## Supported Public Surface

Main supported imports right now:

```python
from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.agents import ChatToolAgent, AsyncChatToolAgent, StructuredOutputAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.data_models.chat_history import ChatHistory, Chats
from ToolAgents.provider import (
    OpenAIChatAPI,
    AnthropicChatAPI,
    MistralChatAPI,
    GroqChatAPI,
    ProviderSettings,
    create_openai_settings,
    create_anthropic_settings,
    create_standard_settings,
    LlamaCppServer,
)
```

Advanced surface:
- `AdvancedAgent` and `AgentConfig` are still available via `ToolAgents.agents`, but they are lazy-imported because they pull in heavier optional dependencies.
- `ChatManager` exists at `ToolAgents.utilities.chat_database` for SQLite-backed chat storage.

## Current State

The branch has already gone through a cleanup and stabilization pass.

What is true now:
- The framework has been moved to a clean-break API direction.
- The old backward-compatibility layer has been removed.
- `ToolAgents.messages` no longer exists.
- Legacy provider aliases like `OpenAISettings` and `AnthropicSettings` have been removed.
- Tests have been reduced to the supported surface rather than carrying ignored legacy integration tests.
- Generated example artifacts and stale tracked outputs have been removed and are now ignored.
- Optional dependencies are now grouped by feature area in `pyproject.toml`.

Current test baseline:
- `python -m pytest -q`
- Result at last validation: `17 passed, 1 warning`
- Remaining warning: an existing Pydantic deprecation warning about class-based config

## Intentionally Retired Surface

The following should be treated as removed, not deprecated:
- `ToolAgents.messages.*`
- compatibility-only `ChatHistory` helpers such as old `load_history`, `save_history`, `to_list`, and `add_list_of_dicts`
- legacy provider settings aliases such as `OpenAISettings` and `AnthropicSettings`
- the previously ignored legacy integration tests under `tests/`

If you see docs/examples referring to those names, they are stale and should be updated rather than restored.

## Project Conventions

Codebase assumptions after the cleanup:
- Prefer `ChatMessage` / `ChatHistory` from `ToolAgents.data_models`
- Prefer `ToolRegistry` for tool wiring
- Prefer `api.get_default_settings()` or the exported settings builders over ad hoc legacy settings classes
- Treat generated example outputs as disposable; they should not be committed
- Keep the public API intentional rather than reintroducing compatibility shims

## Development Notes

Useful commands:

```bash
pip install -e .
python -m pytest -q
```

Repo-local test setup is already configured via `pytest.ini` and `tests/conftest.py`.

## Known Sharp Edges

These areas still deserve careful handling:
- optional-dependency surfaces such as semantic memory and some advanced-agent paths
- MCP and retrieval examples, which may be valid but are broader than the minimal supported baseline
- docs/examples may still need occasional alignment as the cleaned API continues to settle
- Google GenAI support is currently not part of the maintained dependency surface. If it is restored later, it should come back with explicit code ownership, tests, and documentation.

## Recommended Next Steps

High-value follow-up work:
1. Continue trimming or modernizing docs/examples so they only describe the supported surface.
2. Decide the fate of deferred provider support such as Google GenAI.
3. Add targeted tests for optional subsystems that are still meant to be supported.
4. Review `pyproject.toml` dependencies and remove anything no longer aligned with the maintained API.

## Source Of Truth

When in doubt, trust these in order:
1. current code under `src/ToolAgents`
2. current supported tests under `tests`
3. updated docs under `docs`
4. older examples only if they match the code above


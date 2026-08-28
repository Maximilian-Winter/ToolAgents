# ToolAgents Repo Notes

## Project Overview

ToolAgents is a Python package for building function-calling agents across multiple LLM providers. It includes chat agents, tool registration, provider adapters, context management, extension/skill handling, and knowledge/memory utilities.

The package uses a `src` layout:

- `src/ToolAgents/agents`: agent implementations, including `ChatToolAgent` and legacy `AdvancedAgent`.
- `src/ToolAgents/agent_harness`: higher-level harness loop, prompt composition, smart messages, and extension integration.
- `src/ToolAgents/agent_memory`: memory systems. `navigable_memory` is the preferred current memory implementation; `semantic_memory` is legacy.
- `src/ToolAgents/agent_tools`: built-in tools for files, coding, git, GitHub, web search, and related workflows.
- `src/ToolAgents/knowledge`: RAG, document, text splitting, vector database, crawler, and search abstractions.
- `src/ToolAgents/provider`: LLM provider wrappers for OpenAI-compatible APIs, Anthropic, Groq, Mistral, and completion backends.
- `tests`: pytest test suite.
- `examples`: runnable examples and demos. Some older memory/advanced-agent examples are legacy.
- `docs`: MkDocs documentation source.

## Environment

- Python requirement: `>=3.10`
- Package metadata: `pyproject.toml`
- Test config: `pytest.ini`
- Docs config: `mkdocs.yml`

Basic install:

```bash
python -m pip install -e .
python -m pip install pytest
```

For the full test suite that touches extension/skill handling, install the `advanced` extra because it provides `PyYAML`:

```bash
python -m pip install -e ".[advanced]"
python -m pip install pytest
```

Optional extras:

- `advanced`: YAML-backed app state and skill frontmatter support.
- `storage`: SQLAlchemy-backed chat storage.
- `memory`: legacy semantic memory and vector search dependencies.
- `local-inference`: local Hugging Face/transformer inference helpers.
- `ocr`: OCR/PDF ingestion dependencies.
- `search`: web search and crawler dependencies.
- `mcp`: Model Context Protocol tooling.
- `all`: every optional feature.

## Common Commands

Run all tests:

```bash
python -m pytest
```

Run focused navigable-memory tests:

```bash
python -m pytest tests/test_navigable_memory_ingestion.py tests/test_navigable_memory_semantic_search.py tests/test_navigable_memory_versioning.py
```

Build the package locally:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Build docs:

```bash
python -m pip install -r docs/requirements.txt
mkdocs build --strict
```

## CI/CD

GitHub Actions workflows live in `.github/workflows`.

- `ci.yml`: runs pytest across CPython versions and PyPy, then builds package artifacts.
- `docs.yml`: builds MkDocs docs and deploys GitHub Pages on non-PR runs.
- `publish.yml`: builds and publishes distributions to PyPI on `v*` tags or manual dispatch.

Publishing expects PyPI Trusted Publishing with:

- Repository: `Maximilian-Winter/ToolAgents`
- Workflow filename: `publish.yml`
- Environment: `pypi`

Release flow:

```bash
# update version in pyproject.toml first
git add pyproject.toml
git commit -m "Release X.Y.Z"
git tag vX.Y.Z
git push
git push origin vX.Y.Z
```

Docs are published at:

https://maximilian-winter.github.io/ToolAgents/

## Development Notes

- Prefer `NavigableMemory` for new memory work.
- `SemanticMemory`, `SemanticMemoryConfig`, `AdvancedAgent`, and `AgentConfig` are legacy compatibility surfaces. Keep imports working unless explicitly asked to remove them.
- Navigable-memory source-of-truth backends should stay focused on documents, versions, tags, binaries, and references. Semantic search belongs in the optional semantic index layer.
- Avoid importing heavy optional dependencies such as `chromadb`, `sentence_transformers`, `numpy`, rerankers, or OCR dependencies from core modules unless the optional feature is actually constructed.
- Keep examples runnable, but do not present legacy memory/advanced-agent examples as the preferred pattern.
- Tests are stdlib/pytest oriented. Use fake or in-memory providers for semantic/vector tests when possible.
- Preserve existing public APIs and lazy exports unless a task explicitly calls for a breaking change.

## Editing Guidelines

- Use `rg`/`rg --files` for searching.
- Keep changes scoped to the requested subsystem.
- Do not remove or rewrite unrelated examples, generated files, or user changes.
- Use temporary directories in tests for persistent backends and generated state.
- Do not require API keys, network, `.env`, or live LLM calls in tests.
- Run at least the relevant focused tests before handing work back; run the full suite when touching shared harness, provider, package, or workflow behavior.

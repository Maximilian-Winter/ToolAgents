# Memory Examples

Current examples in this directory:

- `rag.py`: dense vector retrieval through the current vector database package.
- `bm_25.py`: dependency-light keyword retrieval.
- `ensemble.py`: hybrid dense and sparse retrieval with optional memory extras.

For agent memory, prefer `../navigable_memory/`. `SemanticMemory` and
`ContextAppState` are legacy compatibility APIs; old scripts that use them are
not part of the maintained example path.

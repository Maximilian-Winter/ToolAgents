# Memory Examples

Current examples in this directory:

- `rag.py`: dense vector retrieval through the current vector database package.
- `bm_25.py`: dependency-light keyword retrieval.
- `ensemble.py`: hybrid dense and sparse retrieval with optional memory extras.

For agent memory, prefer `../navigable_memory/`. Legacy `SemanticMemory` and
`ContextAppState` scripts were removed from this directory; older release tags
and source archives still contain them if compatibility references are needed.
See `../../../docs/removed-legacy.md` for current replacement guidance.

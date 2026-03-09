"""
Optional memory and app-state helpers for ToolAgents.

This package intentionally uses lazy imports so optional dependencies such as
PyYAML, chromadb, sentence-transformers, hdbscan, numpy, and torch are only
required when the related functionality is actually accessed.
"""

__all__ = [
    "ContextAppState",
    "HDBSCANClusterEmbeddingsStrategy",
    "SemanticMemory",
    "SemanticMemoryConfig",
    "SummarizationExtractPatternStrategy",
    "nomic_text_embeddings_gpu_config",
    "semantic_memory_nomic_text_gpu_config",
]


def __getattr__(name: str):
    if name == "ContextAppState":
        try:
            from .context_app_state import ContextAppState
        except ImportError as exc:
            raise ImportError(
                "ContextAppState requires the optional 'advanced' dependencies. "
                "Install ToolAgents[advanced]."
            ) from exc
        return ContextAppState

    if name in {
        "SemanticMemory",
        "SemanticMemoryConfig",
        "SummarizationExtractPatternStrategy",
        "nomic_text_embeddings_gpu_config",
        "semantic_memory_nomic_text_gpu_config",
    }:
        try:
            from .semantic_memory.memory import (
                SemanticMemory,
                SemanticMemoryConfig,
                SummarizationExtractPatternStrategy,
                nomic_text_embeddings_gpu_config,
                semantic_memory_nomic_text_gpu_config,
            )
        except ImportError as exc:
            raise ImportError(
                "Semantic memory features require the optional 'memory' dependencies. "
                "Install ToolAgents[memory]."
            ) from exc

        return {
            "SemanticMemory": SemanticMemory,
            "SemanticMemoryConfig": SemanticMemoryConfig,
            "SummarizationExtractPatternStrategy": SummarizationExtractPatternStrategy,
            "nomic_text_embeddings_gpu_config": nomic_text_embeddings_gpu_config,
            "semantic_memory_nomic_text_gpu_config": semantic_memory_nomic_text_gpu_config,
        }[name]

    if name == "HDBSCANClusterEmbeddingsStrategy":
        try:
            from .semantic_memory.hdbscan_cluster_embeddings_strategy import (
                HDBSCANClusterEmbeddingsStrategy,
            )
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN clustering requires the optional 'memory' dependencies. "
                "Install ToolAgents[memory]."
            ) from exc
        return HDBSCANClusterEmbeddingsStrategy

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

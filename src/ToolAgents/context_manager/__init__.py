"""
Context management for ToolAgents.

Provides token tracking, context window trimming, summarization,
and event hooks for building LLM harnesses on top of ToolAgents.

Uses lazy imports so the module is lightweight until accessed.
"""

__all__ = [
    "ContextManager",
    "create_context_manager",
    "ContextManagerConfig",
    "ContextState",
    "ContextEvent",
    "EventBus",
    "EventData",
    "TokenTracker",
    "ContextStrategy",
    "SlidingWindowStrategy",
    "KeepLastNTurnsStrategy",
    "SummarizeAndTrimStrategy",
    "SummarizationProvider",
    "LLMSummarizationProvider",
    "RuleBasedSummarizationProvider",
]


def __getattr__(name: str):
    if name in {"ContextManager", "create_context_manager"}:
        from .context_manager import ContextManager, create_context_manager

        return {"ContextManager": ContextManager, "create_context_manager": create_context_manager}[name]

    if name in {"ContextManagerConfig", "ContextState"}:
        from .models import ContextManagerConfig, ContextState

        return {"ContextManagerConfig": ContextManagerConfig, "ContextState": ContextState}[name]

    if name in {"ContextEvent", "EventBus", "EventData"}:
        from .events import ContextEvent, EventBus, EventData

        return {"ContextEvent": ContextEvent, "EventBus": EventBus, "EventData": EventData}[name]

    if name == "TokenTracker":
        from .token_tracker import TokenTracker

        return TokenTracker

    if name in {"ContextStrategy", "SlidingWindowStrategy", "KeepLastNTurnsStrategy", "SummarizeAndTrimStrategy"}:
        from .strategies import (
            ContextStrategy,
            SlidingWindowStrategy,
            KeepLastNTurnsStrategy,
            SummarizeAndTrimStrategy,
        )

        return {
            "ContextStrategy": ContextStrategy,
            "SlidingWindowStrategy": SlidingWindowStrategy,
            "KeepLastNTurnsStrategy": KeepLastNTurnsStrategy,
            "SummarizeAndTrimStrategy": SummarizeAndTrimStrategy,
        }[name]

    if name in {"SummarizationProvider", "LLMSummarizationProvider", "RuleBasedSummarizationProvider"}:
        from .summarization import (
            SummarizationProvider,
            LLMSummarizationProvider,
            RuleBasedSummarizationProvider,
        )

        return {
            "SummarizationProvider": SummarizationProvider,
            "LLMSummarizationProvider": LLMSummarizationProvider,
            "RuleBasedSummarizationProvider": RuleBasedSummarizationProvider,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

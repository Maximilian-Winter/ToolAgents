"""
Agent harness for ToolAgents.

Wraps ChatToolAgent + ContextManager into an interactive runtime
with conversation lifecycle, I/O abstraction, and event hooks.

Uses lazy imports so the module is lightweight until accessed.
"""

__all__ = [
    "AgentHarness",
    "create_harness",
    "HarnessConfig",
    "HarnessEvent",
    "HarnessEventData",
    "HarnessEventBus",
    "IOHandler",
    "ConsoleIOHandler",
]


def __getattr__(name: str):
    if name in {"AgentHarness", "create_harness"}:
        from .harness import AgentHarness, create_harness

        return {
            "AgentHarness": AgentHarness,
            "create_harness": create_harness,
        }[name]

    if name == "HarnessConfig":
        from .config import HarnessConfig

        return HarnessConfig

    if name in {"HarnessEvent", "HarnessEventData", "HarnessEventBus"}:
        from .events import HarnessEvent, HarnessEventData, HarnessEventBus

        return {
            "HarnessEvent": HarnessEvent,
            "HarnessEventData": HarnessEventData,
            "HarnessEventBus": HarnessEventBus,
        }[name]

    if name in {"IOHandler", "ConsoleIOHandler"}:
        from .io_handlers import IOHandler, ConsoleIOHandler

        return {
            "IOHandler": IOHandler,
            "ConsoleIOHandler": ConsoleIOHandler,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

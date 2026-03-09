# events.py — Event system for the AgentHarness.
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

from ToolAgents.data_models.responses import ChatResponse


class HarnessEvent(str, Enum):
    """Events emitted by the AgentHarness during its lifecycle."""

    TURN_START = "turn_start"
    TURN_END = "turn_end"
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    HARNESS_START = "harness_start"
    HARNESS_STOP = "harness_stop"
    ERROR = "error"


@dataclass
class HarnessEventData:
    """Payload for harness-level events.

    Attributes:
        event: The event that was fired.
        turn_number: Current turn number.
        user_input: The user's input string (for TURN_START/USER_INPUT).
        response: The agent's ChatResponse (for AGENT_RESPONSE/TURN_END).
        error: Exception if an error occurred.
        metadata: Arbitrary extra data.
    """

    event: HarnessEvent
    turn_number: int = 0
    user_input: Optional[str] = None
    response: Optional[ChatResponse] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


HarnessEventHandler = Callable[[HarnessEventData], None]


class HarnessEventBus:
    """Simple synchronous event bus for harness lifecycle events.

    Same pattern as context_manager.EventBus but keyed on HarnessEvent.

    Usage:
        bus = HarnessEventBus()
        bus.on(HarnessEvent.TURN_END, my_handler)
        bus.emit(HarnessEvent.TURN_END, event_data)
        bus.off(HarnessEvent.TURN_END, my_handler)
    """

    def __init__(self):
        self._handlers: Dict[HarnessEvent, List[HarnessEventHandler]] = {}

    def on(self, event: HarnessEvent, handler: HarnessEventHandler) -> None:
        """Register a handler for an event."""
        if event not in self._handlers:
            self._handlers[event] = []
        if handler not in self._handlers[event]:
            self._handlers[event].append(handler)

    def off(self, event: HarnessEvent, handler: HarnessEventHandler) -> None:
        """Unregister a handler for an event."""
        if event in self._handlers:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass

    def emit(self, event: HarnessEvent, data: HarnessEventData) -> None:
        """Fire all handlers registered for the given event."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(data)

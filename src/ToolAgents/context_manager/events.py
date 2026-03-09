# events.py — Event system for the ContextManager.
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

from ToolAgents.data_models.messages import ChatMessage
from .models import ContextState


class ContextEvent(str, Enum):
    """Events emitted by the ContextManager during its lifecycle."""

    PRE_REQUEST = "pre_request"
    POST_RESPONSE = "post_response"
    MESSAGES_TRIMMED = "messages_trimmed"
    SUMMARY_GENERATED = "summary_generated"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"
    TURN_COMPLETE = "turn_complete"


@dataclass
class EventData:
    """Data payload passed to event handlers.

    Attributes:
        event: The event that was fired.
        state: A snapshot of the ContextState at the time of the event.
        messages: The current message list (if relevant).
        trimmed_messages: Messages that were removed during trimming.
        response: The LLM response message (for POST_RESPONSE events).
        metadata: Arbitrary extra data for custom event handlers.
    """

    event: ContextEvent
    state: ContextState
    messages: Optional[List[ChatMessage]] = None
    trimmed_messages: Optional[List[ChatMessage]] = None
    response: Optional[ChatMessage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[EventData], None]


class EventBus:
    """Simple synchronous event bus for ContextManager lifecycle events.

    Usage:
        bus = EventBus()
        bus.on(ContextEvent.POST_RESPONSE, my_handler)
        bus.emit(ContextEvent.POST_RESPONSE, event_data)
        bus.off(ContextEvent.POST_RESPONSE, my_handler)
    """

    def __init__(self):
        self._handlers: Dict[ContextEvent, List[EventHandler]] = {}

    def on(self, event: ContextEvent, handler: EventHandler) -> None:
        """Register a handler for an event."""
        if event not in self._handlers:
            self._handlers[event] = []
        if handler not in self._handlers[event]:
            self._handlers[event].append(handler)

    def off(self, event: ContextEvent, handler: EventHandler) -> None:
        """Unregister a handler for an event."""
        if event in self._handlers:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass

    def emit(self, event: ContextEvent, data: EventData) -> None:
        """Fire all handlers registered for the given event."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(data)

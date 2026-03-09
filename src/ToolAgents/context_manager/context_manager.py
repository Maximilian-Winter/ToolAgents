# context_manager.py — Core ContextManager orchestrating trimming, tracking, and events.
from typing import List, Optional

from ToolAgents.data_models.messages import ChatMessage
from .models import ContextManagerConfig, ContextState
from .events import EventBus, ContextEvent, EventData
from .token_tracker import TokenTracker
from .strategies import (
    ContextStrategy,
    SlidingWindowStrategy,
    KeepLastNTurnsStrategy,
    SummarizeAndTrimStrategy,
)
from .summarization import SummarizationProvider


class ContextManager:
    """Manages context window size, token budgets, and lifecycle events.

    Sits outside the agent — called by a harness before and after LLM interactions.
    The agent itself is never modified.

    Usage:
        cm = create_context_manager(max_context_tokens=128000)

        # Register event handlers
        cm.events.on(ContextEvent.BUDGET_WARNING, my_warning_handler)

        # Before each agent call
        trimmed = cm.prepare_messages(messages)
        response = agent.get_response(trimmed, tools, settings)

        # After each agent call
        cm.on_response(response)
    """

    def __init__(self, config: ContextManagerConfig):
        self.config = config
        self.tracker = TokenTracker(config)
        self.events = EventBus()
        self._strategy = self._create_strategy(config.strategy)
        self._summarizer: Optional[SummarizationProvider] = None

    # --- Core API ---

    def prepare_messages(
        self,
        messages: List[ChatMessage],
        tools: Optional[List] = None,
    ) -> List[ChatMessage]:
        """Prepare messages for sending to the LLM.

        Fires PRE_REQUEST event, then trims if the context window is too large.
        Returns the message list to actually send.

        Args:
            messages: The full conversation message list.
            tools: Optional list of tools (for event metadata).

        Returns:
            The (possibly trimmed) message list.
        """
        self.events.emit(
            ContextEvent.PRE_REQUEST,
            EventData(
                event=ContextEvent.PRE_REQUEST,
                state=self.tracker.state.model_copy(),
                messages=messages,
                metadata={"tools_count": len(tools) if tools else 0},
            ),
        )

        if not self.tracker.needs_trimming():
            return messages

        kept, trimmed = self._strategy.trim(
            messages, self.tracker.state, self.config
        )

        if trimmed:
            self.tracker.state.messages_trimmed += len(trimmed)
            self.events.emit(
                ContextEvent.MESSAGES_TRIMMED,
                EventData(
                    event=ContextEvent.MESSAGES_TRIMMED,
                    state=self.tracker.state.model_copy(),
                    messages=kept,
                    trimmed_messages=trimmed,
                ),
            )

        return kept

    def on_response(self, response: ChatMessage) -> None:
        """Process an LLM response: update token tracking and fire events.

        Call this after every LLM response (including intermediate tool-call
        responses within an agent loop).

        Args:
            response: The ChatMessage returned by the LLM.
        """
        self.tracker.update_from_response(response)
        self.tracker.state.turn_count += 1

        self.events.emit(
            ContextEvent.POST_RESPONSE,
            EventData(
                event=ContextEvent.POST_RESPONSE,
                state=self.tracker.state.model_copy(),
                response=response,
            ),
        )

        # Check budget thresholds
        if self.tracker.budget_exceeded():
            self.events.emit(
                ContextEvent.BUDGET_EXCEEDED,
                EventData(
                    event=ContextEvent.BUDGET_EXCEEDED,
                    state=self.tracker.state.model_copy(),
                    response=response,
                    metadata={
                        "total_used": self.tracker.state.total_tokens_used,
                        "budget": self.config.total_budget_tokens,
                    },
                ),
            )
        elif self.tracker.budget_warning():
            self.events.emit(
                ContextEvent.BUDGET_WARNING,
                EventData(
                    event=ContextEvent.BUDGET_WARNING,
                    state=self.tracker.state.model_copy(),
                    response=response,
                    metadata={
                        "total_used": self.tracker.state.total_tokens_used,
                        "budget": self.config.total_budget_tokens,
                        "percentage": (
                            self.tracker.state.total_tokens_used
                            / self.config.total_budget_tokens
                            * 100
                        )
                        if self.config.total_budget_tokens
                        else 0,
                    },
                ),
            )

    # --- Event helpers for harness use ---

    def notify_tool_call(self, message: ChatMessage) -> None:
        """Fire TOOL_CALL event. Call when the agent makes a tool call."""
        self.events.emit(
            ContextEvent.TOOL_CALL,
            EventData(
                event=ContextEvent.TOOL_CALL,
                state=self.tracker.state.model_copy(),
                response=message,
            ),
        )

    def notify_tool_result(self, message: ChatMessage) -> None:
        """Fire TOOL_RESULT event. Call when a tool returns its result."""
        self.events.emit(
            ContextEvent.TOOL_RESULT,
            EventData(
                event=ContextEvent.TOOL_RESULT,
                state=self.tracker.state.model_copy(),
                response=message,
            ),
        )

    def notify_user_message(self, message: ChatMessage) -> None:
        """Fire USER_MESSAGE event. Call when the user sends a message."""
        self.events.emit(
            ContextEvent.USER_MESSAGE,
            EventData(
                event=ContextEvent.USER_MESSAGE,
                state=self.tracker.state.model_copy(),
                response=message,
            ),
        )

    def notify_turn_complete(self) -> None:
        """Fire TURN_COMPLETE event. Call when a full agent turn finishes."""
        self.events.emit(
            ContextEvent.TURN_COMPLETE,
            EventData(
                event=ContextEvent.TURN_COMPLETE,
                state=self.tracker.state.model_copy(),
            ),
        )

    # --- Pinning ---

    def pin_message(self, message_id: str) -> None:
        """Pin a message so it is never trimmed."""
        self.tracker.state.pinned_message_ids.add(message_id)

    def unpin_message(self, message_id: str) -> None:
        """Unpin a message, allowing it to be trimmed."""
        self.tracker.state.pinned_message_ids.discard(message_id)

    def is_pinned(self, message_id: str) -> bool:
        """Check if a message is pinned."""
        return message_id in self.tracker.state.pinned_message_ids

    # --- Configuration ---

    def set_strategy(self, strategy: ContextStrategy) -> None:
        """Replace the current trimming strategy."""
        self._strategy = strategy

    def set_summarizer(self, summarizer: SummarizationProvider) -> None:
        """Set or replace the summarization provider.

        If the current strategy is SummarizeAndTrimStrategy, the summarizer
        is also set on the strategy.
        """
        self._summarizer = summarizer
        if isinstance(self._strategy, SummarizeAndTrimStrategy):
            self._strategy.summarizer = summarizer

    @property
    def state(self) -> ContextState:
        """Current context manager state (read-only snapshot)."""
        return self.tracker.state.model_copy()

    # --- Internal ---

    def _create_strategy(self, strategy_name: str) -> ContextStrategy:
        """Create a strategy instance from its name."""
        if strategy_name == "sliding_window":
            return SlidingWindowStrategy()
        elif strategy_name == "keep_last_n_turns":
            return KeepLastNTurnsStrategy()
        elif strategy_name == "summarize_and_trim":
            return SummarizeAndTrimStrategy(summarizer=self._summarizer)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Valid options: 'sliding_window', 'keep_last_n_turns', 'summarize_and_trim'"
            )


def create_context_manager(
    max_context_tokens: int = 128000,
    strategy: str = "sliding_window",
    reserve_tokens: int = 4096,
    total_budget_tokens: Optional[int] = None,
    **kwargs,
) -> ContextManager:
    """Convenience factory for creating a ContextManager.

    Args:
        max_context_tokens: Maximum tokens allowed in the context window.
        strategy: Trimming strategy name ('sliding_window', 'keep_last_n_turns', 'summarize_and_trim').
        reserve_tokens: Tokens reserved for the LLM response.
        total_budget_tokens: Optional hard cap on total conversation tokens.
        **kwargs: Additional fields for ContextManagerConfig.

    Returns:
        A configured ContextManager instance.

    Example:
        cm = create_context_manager(
            max_context_tokens=32000,
            strategy="keep_last_n_turns",
            reserve_tokens=2048,
            keep_last_n=5,
        )
    """
    config = ContextManagerConfig(
        max_context_tokens=max_context_tokens,
        strategy=strategy,
        reserve_tokens=reserve_tokens,
        total_budget_tokens=total_budget_tokens,
        **kwargs,
    )
    return ContextManager(config)

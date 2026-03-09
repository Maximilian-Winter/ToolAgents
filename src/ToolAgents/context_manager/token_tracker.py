# token_tracker.py — Tracks token usage from LLM responses.
from ToolAgents.data_models.messages import ChatMessage
from .models import ContextManagerConfig, ContextState


class TokenTracker:
    """Accumulates token usage from ChatMessage.token_usage fields.

    No external tokenizer needed — reads directly from provider-reported counts.
    The first request in a conversation is "blind" (no prior usage data), which
    is acceptable since we don't yet know the context size.

    Usage:
        tracker = TokenTracker(config)
        tracker.update_from_response(response_message)
        if tracker.needs_trimming():
            # trim messages
    """

    def __init__(self, config: ContextManagerConfig):
        self.config = config
        self.state = ContextState()

    def update_from_response(self, message: ChatMessage) -> None:
        """Update state from a ChatMessage's token_usage.

        Called after each LLM response. The input_tokens field from the response
        tells us how large the current context window is.
        """
        if message.token_usage is not None:
            self.state.total_input_tokens += message.token_usage.input_tokens
            self.state.total_output_tokens += message.token_usage.output_tokens
            self.state.total_tokens_used += message.token_usage.total_tokens
            # The input_tokens from the latest response = current context size
            self.state.current_context_tokens = message.token_usage.input_tokens

    def needs_trimming(self) -> bool:
        """True if the current context exceeds (max_context_tokens - reserve_tokens)."""
        effective_limit = self.config.max_context_tokens - self.config.reserve_tokens
        return self.state.current_context_tokens > effective_limit

    def budget_exceeded(self) -> bool:
        """True if the total conversation budget has been exceeded."""
        if self.config.total_budget_tokens is None:
            return False
        return self.state.total_tokens_used >= self.config.total_budget_tokens

    def budget_warning(self) -> bool:
        """True when 80% of the total conversation budget has been used."""
        if self.config.total_budget_tokens is None:
            return False
        return self.state.total_tokens_used >= self.config.total_budget_tokens * 0.8

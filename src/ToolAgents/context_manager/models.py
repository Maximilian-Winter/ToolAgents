# models.py — Configuration and state models for the ContextManager.
from typing import Optional, Set, Dict, Any

from pydantic import BaseModel, Field


class ContextManagerConfig(BaseModel):
    """Configuration for the ContextManager.

    Attributes:
        max_context_tokens: Maximum tokens allowed in the context window.
        reserve_tokens: Tokens reserved for the LLM response (headroom).
        strategy: Name of the trimming strategy to use.
        summarization_model: Model name for LLM-based summarization (if applicable).
        summarization_max_tokens: Max tokens for generated summaries.
        keep_last_n: Number of turns to keep (for keep_last_n_turns strategy).
        always_keep_system: If True, system messages are never trimmed.
        total_budget_tokens: Optional hard cap on total tokens for the entire conversation.
    """

    max_context_tokens: int = Field(
        default=128000, description="Maximum tokens allowed in the context window."
    )
    reserve_tokens: int = Field(
        default=4096, description="Tokens reserved for the LLM response."
    )
    strategy: str = Field(
        default="sliding_window",
        description="Trimming strategy: 'sliding_window', 'summarize_and_trim', or 'keep_last_n_turns'.",
    )

    # Summarization settings
    summarization_model: Optional[str] = Field(
        default=None, description="Model name for LLM-based summarization."
    )
    summarization_max_tokens: int = Field(
        default=500, description="Max tokens for generated summaries."
    )

    # Strategy-specific
    keep_last_n: int = Field(
        default=10, description="Number of turns to keep (for keep_last_n_turns)."
    )
    always_keep_system: bool = Field(
        default=True, description="If True, system messages are never trimmed."
    )

    # Budget tracking
    total_budget_tokens: Optional[int] = Field(
        default=None,
        description="Optional hard cap on total tokens for the entire conversation.",
    )


class ContextState(BaseModel):
    """Tracks the current state of the context manager.

    Updated after each LLM response. Readable at any time via ContextManager.state.

    Attributes:
        total_input_tokens: Cumulative input tokens across all calls.
        total_output_tokens: Cumulative output tokens across all calls.
        total_tokens_used: Cumulative total tokens across all calls.
        current_context_tokens: Estimated token count of the current context window.
        messages_trimmed: Total number of messages removed by trimming.
        summaries_generated: Number of summaries created by summarization strategies.
        turn_count: Number of complete LLM call cycles.
        pinned_message_ids: IDs of messages that should never be trimmed.
    """

    total_input_tokens: int = Field(
        default=0, description="Cumulative input tokens across all calls."
    )
    total_output_tokens: int = Field(
        default=0, description="Cumulative output tokens across all calls."
    )
    total_tokens_used: int = Field(
        default=0, description="Cumulative total tokens across all calls."
    )
    current_context_tokens: int = Field(
        default=0, description="Estimated current context window size in tokens."
    )
    messages_trimmed: int = Field(
        default=0, description="Total number of messages removed by trimming."
    )
    summaries_generated: int = Field(
        default=0, description="Number of summaries created."
    )
    turn_count: int = Field(
        default=0, description="Number of complete LLM call cycles."
    )
    pinned_message_ids: Set[str] = Field(
        default_factory=set, description="IDs of messages that should never be trimmed."
    )

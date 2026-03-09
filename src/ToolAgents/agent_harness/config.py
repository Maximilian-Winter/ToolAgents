# config.py — Configuration model for the AgentHarness.
from typing import Optional

from pydantic import BaseModel, Field


class HarnessConfig(BaseModel):
    """Configuration for the AgentHarness.

    Attributes:
        system_prompt: System prompt prepended to every conversation.
        max_turns: Maximum user turns before auto-stop (-1 for unlimited).
        stop_on_budget_exceeded: Whether to stop when context budget is exceeded.
        streaming: Whether to use streaming mode by default in run().
        context_manager_config: Kwargs passed to create_context_manager().
    """

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt prepended to every conversation.",
    )
    max_turns: int = Field(
        default=-1,
        description="Maximum user turns before auto-stop (-1 for unlimited).",
    )
    stop_on_budget_exceeded: bool = Field(
        default=True,
        description="Whether to stop the harness when the context budget is exceeded.",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to use streaming mode by default in run().",
    )
    context_manager_config: dict = Field(
        default_factory=dict,
        description="Kwargs passed to create_context_manager().",
    )

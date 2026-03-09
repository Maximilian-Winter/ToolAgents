"""
Context Manager Strategies Example
====================================

Shows the different context trimming strategies available:

1. sliding_window (default) — drops oldest messages first
2. keep_last_n_turns — keeps only the last N user-assistant turn pairs
3. summarize_and_trim — summarizes old messages before dropping them

Each strategy respects:
- System messages are never trimmed
- Pinned messages are never trimmed
- Tool call + tool result pairs are kept/removed together (atomic)
"""

import os

from dotenv import load_dotenv

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_harness import create_harness, HarnessEvent
from ToolAgents.context_manager import (
    ContextEvent,
    SlidingWindowStrategy,
    KeepLastNTurnsStrategy,
    SummarizeAndTrimStrategy,
    RuleBasedSummarizationProvider,
)

load_dotenv()

api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# Strategy 1: Sliding Window (default)
# ============================================================

print("=== Strategy: Sliding Window ===")
print("Drops oldest messages first when context is full.\n")

harness1 = create_harness(
    provider=api,
    system_prompt="You are helpful. Be very concise.",
    max_context_tokens=128000,
    strategy="sliding_window",  # This is the default
)

harness1.context_manager.events.on(
    ContextEvent.MESSAGES_TRIMMED,
    lambda e: print(f"  [Trimmed {len(e.trimmed_messages)} messages]"),
)

print("(type 'exit' to switch to next strategy)\n")
harness1.run()


# ============================================================
# Strategy 2: Keep Last N Turns
# ============================================================

print("\n=== Strategy: Keep Last N Turns ===")
print("Keeps only the last 5 user-assistant turn pairs.\n")

harness2 = create_harness(
    provider=api,
    system_prompt="You are helpful. Be very concise.",
    max_context_tokens=128000,
    strategy="keep_last_n_turns",
    keep_last_n=5,
)

harness2.context_manager.events.on(
    ContextEvent.MESSAGES_TRIMMED,
    lambda e: print(f"  [Trimmed {len(e.trimmed_messages)} messages, keeping last 5 turns]"),
)

print("(type 'exit' to switch to next strategy)\n")
harness2.run()


# ============================================================
# Strategy 3: Summarize and Trim
# ============================================================

print("\n=== Strategy: Summarize and Trim ===")
print("Summarizes old messages before dropping them.")
print("Using rule-based summarization (no extra LLM call).\n")

harness3 = create_harness(
    provider=api,
    system_prompt="You are helpful. Be very concise.",
    max_context_tokens=128000,
    strategy="summarize_and_trim",
)

# Set up a rule-based summarizer (no extra LLM call needed)
summarizer = RuleBasedSummarizationProvider(max_chars_per_message=200)
harness3.context_manager.set_summarizer(summarizer)

harness3.context_manager.events.on(
    ContextEvent.MESSAGES_TRIMMED,
    lambda e: print(f"  [Summarized and trimmed {len(e.trimmed_messages)} messages]"),
)

print("(type 'exit' to quit)\n")
harness3.run()


# ============================================================
# Bonus: Switching strategies at runtime
# ============================================================

print("\n=== Switching Strategies at Runtime ===\n")

harness4 = create_harness(
    provider=api,
    system_prompt="You are helpful. Be concise.",
    max_context_tokens=128000,
)

# Start with sliding window (default)
print("Current strategy: SlidingWindow")
print(harness4.chat("Hello! Remember the word 'elephant'."))

# Switch to keep last N turns
harness4.context_manager.set_strategy(KeepLastNTurnsStrategy())
print("\nSwitched to: KeepLastNTurns")
print(harness4.chat("What word did I ask you to remember?"))

print(f"\nTotal tokens: {harness4.context_state.total_tokens_used}")

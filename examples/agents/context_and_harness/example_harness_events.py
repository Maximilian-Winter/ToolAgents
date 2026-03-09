"""
Agent Harness Events Example
==============================

Shows how to use event hooks on both the AgentHarness and the
underlying ContextManager to observe and react to the conversation
lifecycle.

Events let you:
- Log token usage after each turn
- React when messages get trimmed from context
- Get warnings when approaching budget limits
- Track conversation flow for analytics
"""

import os
from dotenv import load_dotenv

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_harness import create_harness, HarnessEvent
from ToolAgents.context_manager import ContextEvent

load_dotenv()

# --- Set up provider ---

api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# --- Create harness with a token budget ---

harness = create_harness(
    provider=api,
    system_prompt="You are a helpful assistant. Be concise.",
    max_context_tokens=128000,
    total_budget_tokens=50000,  # Hard cap: stop after 50k total tokens
)


# --- Register harness-level event handlers ---

def on_turn_start(event_data):
    print(f"\n--- Turn {event_data.turn_number} starting ---")
    print(f"User: {event_data.user_input}")


def on_turn_end(event_data):
    state = harness.context_state
    print(f"--- Turn {event_data.turn_number} complete ---")
    print(f"  Input tokens this context:  {state.current_context_tokens}")
    print(f"  Total tokens used overall:  {state.total_tokens_used}")
    if harness.context_manager.config.total_budget_tokens:
        budget = harness.context_manager.config.total_budget_tokens
        pct = (state.total_tokens_used / budget) * 100
        print(f"  Budget usage: {pct:.1f}% ({state.total_tokens_used}/{budget})")


def on_error(event_data):
    print(f"ERROR: {event_data.error}")


harness.events.on(HarnessEvent.TURN_START, on_turn_start)
harness.events.on(HarnessEvent.TURN_END, on_turn_end)
harness.events.on(HarnessEvent.ERROR, on_error)


# --- Register context-manager-level event handlers ---

def on_messages_trimmed(event_data):
    count = len(event_data.trimmed_messages) if event_data.trimmed_messages else 0
    print(f"  [Context] Trimmed {count} messages to fit context window")


def on_budget_warning(event_data):
    pct = event_data.metadata.get("percentage", 0)
    print(f"  [Context] WARNING: Budget at {pct:.0f}%!")


def on_budget_exceeded(event_data):
    print(f"  [Context] BUDGET EXCEEDED — harness will stop")


harness.context_manager.events.on(ContextEvent.MESSAGES_TRIMMED, on_messages_trimmed)
harness.context_manager.events.on(ContextEvent.BUDGET_WARNING, on_budget_warning)
harness.context_manager.events.on(ContextEvent.BUDGET_EXCEEDED, on_budget_exceeded)


# --- Run ---

print("Chat with event tracking (type 'exit' to quit)")
print("Watch the token usage after each turn!")
print("=" * 50)
harness.run()

"""
Message Pinning Example
========================

Shows how to pin important messages so they are never trimmed
from the context window, even when the context gets full.

Useful for:
- Keeping critical instructions in context
- Preserving important facts the user mentioned
- Ensuring key tool results stay available
"""

import os
from dotenv import load_dotenv

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.agent_harness import create_harness, HarnessEvent
from ToolAgents.context_manager import ContextEvent

load_dotenv()

api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

harness = create_harness(
    provider=api,
    system_prompt="You are a helpful assistant. Be concise.",
    max_context_tokens=128000,
)

# Track trimming
harness.context_manager.events.on(
    ContextEvent.MESSAGES_TRIMMED,
    lambda e: print(f"\n  [Trimmed {len(e.trimmed_messages)} messages from context]"),
)

# --- Programmatic example: pin important messages ---

print("=== Message Pinning Demo ===\n")

# First turn — user gives important info
response = harness.chat("My API key is ABC-123-XYZ. Remember this, I'll need it later.")
print(f"Turn 1: {response}")

# Pin the user message so it's never trimmed
# The user message we just added is the first message in the conversation
user_msg = harness.messages[0]  # First message is the user msg
harness.context_manager.pin_message(user_msg.id)
print(f"  [Pinned message: '{user_msg.content[0].content[:50]}...']")

# Continue the conversation
response = harness.chat("Tell me a joke.")
print(f"\nTurn 2: {response}")

response = harness.chat("Tell me another joke.")
print(f"\nTurn 3: {response}")

# Even after many turns, the pinned message stays in context
response = harness.chat("What was my API key?")
print(f"\nTurn 4: {response}")

# Show pinned message IDs
state = harness.context_state
print(f"\nPinned messages: {len(state.pinned_message_ids)}")
print(f"Total messages: {len(harness.messages)}")

# Unpin when no longer needed
harness.context_manager.unpin_message(user_msg.id)
print(f"Unpinned the API key message.")

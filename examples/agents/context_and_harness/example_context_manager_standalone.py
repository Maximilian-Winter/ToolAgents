"""
Standalone Context Manager Example
====================================

Shows how to use the ContextManager directly with a ChatToolAgent,
WITHOUT the AgentHarness. This is useful when you want full control
over the conversation loop but still want automatic context management.

The ContextManager sits between you and the agent:
1. You build messages
2. ContextManager trims them if needed
3. You send trimmed messages to the agent
4. You tell ContextManager about the response (for tracking)
"""

import os
from copy import copy

from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.context_manager import (
    create_context_manager,
    ContextEvent,
)

load_dotenv()

# --- Set up provider and agent ---

api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.3

# --- Create a context manager ---

context_manager = create_context_manager(
    max_context_tokens=128000,
    reserve_tokens=4096,
    strategy="sliding_window",       # drops oldest messages first
    # strategy="keep_last_n_turns",  # keeps only last N turns
    # keep_last_n=5,                 # for keep_last_n_turns strategy
)


# --- Register event handlers for observability ---

def on_trimmed(event_data):
    count = len(event_data.trimmed_messages) if event_data.trimmed_messages else 0
    print(f"[Context] Trimmed {count} old messages to fit context window")


context_manager.events.on(ContextEvent.MESSAGES_TRIMMED, on_trimmed)

# --- Manual conversation loop ---

system_prompt = "You are a helpful assistant. Keep answers short."
messages = []

print("Chat with standalone ContextManager (type 'exit' to quit)")
print("=" * 50)

while True:
    user_input = input("> ")
    if user_input.strip().lower() in ("exit", "quit"):
        break
    if not user_input.strip():
        continue

    # Add user message
    user_msg = ChatMessage.create_user_message(user_input)
    messages.append(user_msg)
    context_manager.notify_user_message(user_msg)

    # Build full message list with system prompt
    full_messages = [ChatMessage.create_system_message(system_prompt)] + messages

    # Let the context manager trim if needed
    trimmed_messages = context_manager.prepare_messages(full_messages)

    # Send to agent (copy because agent mutates the list)
    response = agent.get_response(
        messages=list(trimmed_messages),
        settings=settings,
    )

    # Tell context manager about the response (for token tracking)
    for msg in agent.last_messages_buffer:
        if msg.token_usage is not None:
            context_manager.on_response(msg)

    # Add the response to our conversation history
    for msg in agent.last_messages_buffer:
        messages.append(msg)

    context_manager.notify_turn_complete()

    # Print the response
    print(response.response)

    # Show token stats
    state = context_manager.state
    print(f"  [tokens: context={state.current_context_tokens}, total={state.total_tokens_used}]")

print(f"\nFinal stats:")
print(f"  Turns: {context_manager.state.turn_count}")
print(f"  Total tokens: {context_manager.state.total_tokens_used}")
print(f"  Messages trimmed: {context_manager.state.messages_trimmed}")

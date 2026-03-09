"""
Programmatic Agent Harness Example
====================================

Shows how to use the AgentHarness programmatically (no REPL loop),
which is useful for:
- Scripts and automation
- Web APIs
- Testing
- Multi-turn conversations in code

Demonstrates chat(), chat_response(), conversation persistence,
system prompt changes, and reset().
"""

import os
from dotenv import load_dotenv

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_harness import create_harness

load_dotenv()

# --- Set up provider ---

api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# --- Simple chat() usage ---

harness = create_harness(
    provider=api,
    system_prompt="You are a helpful assistant. Answer in one sentence.",
)

# chat() returns just the response string
print("--- Simple chat() ---")
response = harness.chat("What is the capital of France?")
print(f"Q: What is the capital of France?")
print(f"A: {response}\n")

# Conversation persists — the agent remembers context
response = harness.chat("And what about Germany?")
print(f"Q: And what about Germany?")
print(f"A: {response}\n")

print(f"Turns: {harness.turn_count}")
print(f"Messages in history: {len(harness.messages)}")
print(f"Tokens used: {harness.context_state.total_tokens_used}\n")

# --- chat_response() for more detail ---

print("--- chat_response() ---")
full_response = harness.chat_response("And Japan?")
print(f"Response text: {full_response.response}")
print(f"Messages in response: {len(full_response.messages)}\n")

# --- Reset and start a new conversation ---

print("--- After reset() ---")
harness.reset()
print(f"Turns after reset: {harness.turn_count}")
print(f"Messages after reset: {len(harness.messages)}")

# Change the system prompt for the new conversation
harness.set_system_prompt("You are a pirate. Answer everything like a pirate.")

response = harness.chat("What is the capital of France?")
print(f"\nQ: What is the capital of France? (pirate mode)")
print(f"A: {response}\n")

# --- Max turns example ---

print("--- Max turns ---")
limited_harness = create_harness(
    provider=api,
    system_prompt="You are helpful. Answer in one word.",
    max_turns=3,
)

for i, question in enumerate(["Red or blue?", "Cat or dog?", "Hot or cold?", "Left or right?"], 1):
    try:
        answer = limited_harness.chat(question)
        print(f"Turn {i}: {question} -> {answer}")
    except RuntimeError as e:
        print(f"Turn {i}: {question} -> STOPPED: {e}")

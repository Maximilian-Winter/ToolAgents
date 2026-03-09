"""
Streaming Agent Harness Example
================================

Shows how to use the AgentHarness in streaming mode, where responses
are printed token-by-token as they arrive from the LLM.

Two approaches:
1. streaming=True in create_harness() — run() streams automatically
2. chat_stream() for programmatic streaming control
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

# --- Approach 1: Streaming REPL ---

print("=== Streaming REPL ===")
print("Responses stream token-by-token. Type 'exit' to quit.\n")

harness = create_harness(
    provider=api,
    system_prompt="You are a helpful assistant. Be concise.",
    streaming=True,  # This makes run() use streaming mode
)

harness.run()

# --- Approach 2: Programmatic streaming ---

print("\n=== Programmatic Streaming ===\n")

harness2 = create_harness(
    provider=api,
    system_prompt="You are a creative storyteller. Tell very short stories.",
)

# Use chat_stream() to get individual chunks
print("Prompt: Tell me a 3-sentence story about a robot.\n")
print("Response: ", end="")
for chunk in harness2.chat_stream("Tell me a 3-sentence story about a robot."):
    if chunk.chunk:
        print(chunk.chunk, end="", flush=True)
    if chunk.finished:
        print("\n")

# The conversation persists — ask a follow-up
print("Prompt: Now make it a sad ending.\n")
print("Response: ", end="")
for chunk in harness2.chat_stream("Now make it a sad ending."):
    if chunk.chunk:
        print(chunk.chunk, end="", flush=True)
    if chunk.finished:
        print("\n")

print(f"Total tokens used: {harness2.context_state.total_tokens_used}")

"""
Basic Agent Harness Example
============================

Shows the simplest way to create an interactive agent with automatic
context management using the AgentHarness.

The harness wraps ChatToolAgent + ContextManager so you get:
- Automatic conversation persistence across turns
- Prompt modules via harness.prompt_composer
- Ephemeral or pinned messages via harness.add_ephemeral_message()
- Context window management (trimming when approaching limits)
- Token usage tracking
- Interactive REPL loop

Just pick a provider, set a system prompt, add tools, and run().
"""

import os
from dotenv import load_dotenv

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_harness import create_harness
from ToolAgents.data_models.messages import ChatMessage

load_dotenv()

# --- Pick your provider (uncomment one) ---

# OpenAI
# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# OpenRouter
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# Local server (vllm, llama-cpp-server, etc.)
# api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="my-model")

# --- Create the harness ---

harness = create_harness(
    provider=api,
    system_prompt="You are a friendly and helpful assistant. Be concise in your responses.",
    max_context_tokens=128000,
)

# Optional: add dynamic prompt context that is compiled before each turn.
harness.prompt_composer.add_module(
    "session_style",
    position=10,
    content_fn=lambda: "Current session style: concise, practical answers.",
)

# Optional: inject short-lived context without manually managing message history.
harness.add_ephemeral_message(
    ChatMessage.create_system_message("For the next few turns, prefer examples."),
    ttl=3,
)

# --- Run the interactive REPL ---

print("Chat with the assistant (type 'exit' to quit)")
print("=" * 50)
harness.run()

# After the loop ends, you can inspect the conversation:
print(f"\nConversation had {harness.turn_count} turns")
print(f"Total tokens used: {harness.context_state.total_tokens_used}")

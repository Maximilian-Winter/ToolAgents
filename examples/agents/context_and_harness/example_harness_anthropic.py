"""
Agent Harness with Anthropic Example
======================================

Shows how to use the AgentHarness with Anthropic's Claude API.
Works with any provider — just swap the API object.

Also demonstrates using the native AnthropicChatAPI provider
(which uses Anthropic's own API format, not the OpenAI-compatible one).
"""

import os
import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool
from ToolAgents.provider import AnthropicChatAPI
from ToolAgents.agent_harness import create_harness, HarnessEvent

load_dotenv()

# --- Define a tool ---


def get_current_datetime(output_format: str):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time
    """
    return datetime.datetime.now().strftime(output_format)


class search_knowledge_base(BaseModel):
    """Search a knowledge base for information on a topic."""

    query: str = Field(..., description="The search query to look up")

    def run(self):
        # Simulated search results
        results = {
            "python": "Python is a high-level programming language created by Guido van Rossum.",
            "rust": "Rust is a systems programming language focused on safety and performance.",
            "javascript": "JavaScript is a dynamic language primarily used for web development.",
        }
        for key, value in results.items():
            if key in self.query.lower():
                return value
        return f"No results found for: {self.query}"


# --- Set up Anthropic provider ---

api = AnthropicChatAPI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-haiku-4-5-20251001",  # Fast and cheap for demos
)

# --- Create the harness ---

harness = create_harness(
    provider=api,
    system_prompt=(
        "You are a knowledgeable research assistant. Use your tools "
        "to look up information when asked. Be concise and factual."
    ),
    max_context_tokens=200000,  # Claude's context window
    streaming=True,
)

harness.add_tool(FunctionTool(get_current_datetime))
harness.add_tool(FunctionTool(search_knowledge_base))

# Track token usage with cache info (Anthropic supports prompt caching)
def on_turn_end(event_data):
    state = harness.context_state
    print(f"\n  [tokens: in={state.total_input_tokens}, out={state.total_output_tokens}]")

harness.events.on(HarnessEvent.TURN_END, on_turn_end)

# --- Run ---

print("Chat with Claude (type 'exit' to quit)")
print("Try: 'Search for information about Python'")
print("Try: 'What time is it?'")
print("=" * 50)
harness.run()

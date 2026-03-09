"""
Custom I/O Handler Example
============================

Shows how to create a custom IOHandler for the AgentHarness.
This lets you plug the harness into any UI: web apps, Discord bots,
Slack integrations, game engines, etc.

This example creates a simple "formatted" I/O handler that adds
timestamps and turn numbers to the output.
"""

import os
import datetime
from typing import Optional

from dotenv import load_dotenv

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_harness import create_harness
from ToolAgents.data_models.responses import ChatResponseChunk

load_dotenv()


# --- Custom I/O Handler ---

class FormattedIOHandler:
    """An I/O handler that adds timestamps and formatting to output."""

    def __init__(self, agent_name: str = "Assistant"):
        self.agent_name = agent_name
        self.turn = 0

    def get_input(self, prompt: str = "> ") -> Optional[str]:
        try:
            self.turn += 1
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            user_input = input(f"[{timestamp}] You: ")
            if user_input.strip().lower() in ("exit", "quit", "/exit"):
                return None
            return user_input
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def on_text(self, text: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.agent_name}: {text}")

    def on_chunk(self, chunk: ChatResponseChunk) -> None:
        if chunk.chunk:
            print(chunk.chunk, end="", flush=True)
        if chunk.finished:
            print()

    def on_error(self, error: Exception) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ERROR: {error}")


# --- Another example: a handler that logs to a list (useful for testing/web) ---

class BufferedIOHandler:
    """Collects all outputs into a list instead of printing.

    Useful for web APIs, testing, or any non-interactive context.
    Requires inputs to be provided upfront.
    """

    def __init__(self, inputs: list):
        self.inputs = list(inputs)
        self.outputs = []
        self._input_index = 0

    def get_input(self, prompt: str = "> ") -> Optional[str]:
        if self._input_index >= len(self.inputs):
            return None
        result = self.inputs[self._input_index]
        self._input_index += 1
        return result

    def on_text(self, text: str) -> None:
        self.outputs.append(text)

    def on_chunk(self, chunk: ChatResponseChunk) -> None:
        if chunk.finished and chunk.finished_response:
            self.outputs.append(chunk.finished_response.response)

    def on_error(self, error: Exception) -> None:
        self.outputs.append(f"ERROR: {error}")


# --- Set up and run ---

api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# Example 1: Formatted console handler
print("=== Formatted I/O Handler ===")
print("(type 'exit' to quit)\n")

harness = create_harness(
    provider=api,
    system_prompt="You are a helpful assistant named Aria. Be concise.",
)

harness.run(io_handler=FormattedIOHandler(agent_name="Aria"))

# Example 2: Buffered handler (non-interactive)
print("\n=== Buffered I/O Handler (scripted) ===\n")

harness2 = create_harness(
    provider=api,
    system_prompt="You are helpful. Answer in one sentence.",
)

buffered = BufferedIOHandler(inputs=[
    "What is the meaning of life?",
    "Summarize that in 3 words.",
])

harness2.run(io_handler=buffered)

for i, output in enumerate(buffered.outputs, 1):
    print(f"Response {i}: {output}")

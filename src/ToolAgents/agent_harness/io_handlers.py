# io_handlers.py — Pluggable I/O handlers for the AgentHarness.
from typing import Protocol, Optional, Tuple

from ToolAgents.data_models.responses import ChatResponseChunk


class IOHandler(Protocol):
    """Protocol for pluggable I/O. Implement for console, web UI, API, etc.

    The harness calls these methods during run():
    - get_input(): get user input (return None to signal exit)
    - on_text(): display final response text (non-streaming mode)
    - on_chunk(): display a streaming chunk
    - on_error(): display an error
    """

    def get_input(self, prompt: str = "> ") -> Optional[str]:
        """Get user input. Return None to signal exit."""
        ...

    def on_text(self, text: str) -> None:
        """Called with the final response text (non-streaming mode)."""
        ...

    def on_chunk(self, chunk: ChatResponseChunk) -> None:
        """Called with each streaming chunk."""
        ...

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        ...


class ConsoleIOHandler:
    """Default console I/O handler for interactive REPL use.

    Reads from stdin, prints to stdout. Exits on "exit", "quit",
    "/exit", "/quit", Ctrl+C, or EOF.

    Args:
        prompt: The input prompt string.
        exit_commands: Tuple of strings that signal exit.
    """

    def __init__(
        self,
        prompt: str = "> ",
        exit_commands: Tuple[str, ...] = ("exit", "quit", "/exit", "/quit"),
    ):
        self.prompt = prompt
        self.exit_commands = exit_commands

    def get_input(self, prompt: str = None) -> Optional[str]:
        """Read a line from stdin. Returns None on exit command or interrupt."""
        try:
            user_input = input(prompt or self.prompt)
            if user_input.strip().lower() in self.exit_commands:
                return None
            return user_input
        except (EOFError, KeyboardInterrupt):
            print()  # newline after ^C
            return None

    def on_text(self, text: str) -> None:
        """Print the response text."""
        print(text)

    def on_chunk(self, chunk: ChatResponseChunk) -> None:
        """Print streaming chunks, with a newline when finished."""
        if chunk.chunk:
            print(chunk.chunk, end="", flush=True)
        if chunk.finished:
            print()  # newline after stream completes

    def on_error(self, error: Exception) -> None:
        """Print the error."""
        print(f"Error: {error}")

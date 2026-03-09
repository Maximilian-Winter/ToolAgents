# harness.py — Core AgentHarness wrapping ChatToolAgent + ContextManager.
from typing import List, Optional, Generator

from ToolAgents.agents.chat_tool_agent import ChatToolAgent
from ToolAgents.context_manager.context_manager import ContextManager, create_context_manager
from ToolAgents.context_manager.events import ContextEvent
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole
from ToolAgents.data_models.responses import ChatResponse, ChatResponseChunk
from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.provider.llm_provider import ChatAPIProvider, ProviderSettings

from .config import HarnessConfig
from .events import HarnessEvent, HarnessEventData, HarnessEventBus
from .io_handlers import IOHandler, ConsoleIOHandler


class AgentHarness:
    """Wraps ChatToolAgent + ContextManager into an interactive runtime.

    The harness is the OUTER loop around the agent. It manages:
    - The growing message list across user turns
    - System prompt (always prepended, never trimmed)
    - Context window management (trim before, track after)
    - Tool registration
    - Turn lifecycle and event hooks
    - Interactive REPL or programmatic use

    The agent's internal tool-call loop is used as-is. After each agent call,
    the harness retroactively processes the agent's last_messages_buffer to
    update context tracking.

    Usage:
        harness = create_harness(provider=my_provider, system_prompt="You are helpful.")
        harness.add_tool(FunctionTool(my_function))

        # Programmatic
        print(harness.chat("Hello!"))
        print(harness.chat("Do something"))

        # Or interactive REPL
        harness.run()
    """

    def __init__(
        self,
        provider: ChatAPIProvider,
        system_prompt: str = "You are a helpful assistant.",
        config: Optional[HarnessConfig] = None,
        context_manager: Optional[ContextManager] = None,
        settings: Optional[ProviderSettings] = None,
        log_output: bool = False,
    ):
        """Initialize the harness.

        Args:
            provider: The LLM provider (OpenAI, Anthropic, Groq, Mistral, etc.).
            system_prompt: System prompt for the agent. Ignored if config is provided.
            config: Full HarnessConfig. If None, one is created from system_prompt.
            context_manager: Optional pre-configured ContextManager. If None, one is
                created from config.context_manager_config.
            settings: Provider settings (temperature, max_tokens, etc.).
            log_output: Whether to enable agent-level logging.
        """
        # Config
        if config is None:
            config = HarnessConfig(system_prompt=system_prompt)
        self.config = config

        # Agent (composition — we create it, never modify its internals)
        self._agent = ChatToolAgent(chat_api=provider, log_output=log_output)

        # Context manager
        if context_manager is not None:
            self._context_manager = context_manager
        elif config.context_manager_config:
            self._context_manager = create_context_manager(**config.context_manager_config)
        else:
            self._context_manager = create_context_manager()

        # Tool registry
        self._tool_registry = ToolRegistry()

        # Provider settings
        self._settings = settings

        # Conversation state
        self._messages: List[ChatMessage] = []
        self._turn_count: int = 0
        self._stopped: bool = False
        self._budget_exceeded: bool = False

        # Harness event bus
        self._events = HarnessEventBus()

        # Wire up budget exceeded from context manager
        self._context_manager.events.on(
            ContextEvent.BUDGET_EXCEEDED,
            self._on_budget_exceeded,
        )

    # --- Tool Management ---

    def add_tool(self, tool: FunctionTool) -> "AgentHarness":
        """Register a tool. Returns self for chaining."""
        self._tool_registry.add_tool(tool)
        return self

    def add_tools(self, tools: List[FunctionTool]) -> "AgentHarness":
        """Register multiple tools. Returns self for chaining."""
        self._tool_registry.add_tools(tools)
        return self

    def remove_tool(self, name: str) -> "AgentHarness":
        """Remove a tool by name. Returns self for chaining."""
        self._tool_registry.remove(name)
        return self

    # --- Core API ---

    def chat(self, user_input: str) -> str:
        """Send a message, get a response string. Simplest API.

        Args:
            user_input: The user's message text.

        Returns:
            The agent's response as a string.
        """
        response = self.chat_response(user_input)
        return response.response

    def chat_response(self, user_input: str) -> ChatResponse:
        """Send a message, get a full ChatResponse with message history.

        Args:
            user_input: The user's message text.

        Returns:
            ChatResponse containing the full message list and response text.
        """
        self._check_stopped()
        self._turn_count += 1

        # Emit TURN_START
        self._events.emit(
            HarnessEvent.TURN_START,
            HarnessEventData(
                event=HarnessEvent.TURN_START,
                turn_number=self._turn_count,
                user_input=user_input,
            ),
        )

        # Build user message and append to conversation
        user_msg = ChatMessage.create_user_message(user_input)
        self._messages.append(user_msg)
        self._context_manager.notify_user_message(user_msg)

        # Prepare messages: system prompt + trimmed conversation (as a COPY)
        send_messages = self._prepare_messages()

        # Call agent — it handles tool-call loop internally
        response = self._agent.get_response(
            messages=send_messages,
            tool_registry=self._tool_registry,
            settings=self._settings,
        )

        # Post-process: walk last_messages_buffer for context tracking
        self._process_agent_buffer(self._agent.last_messages_buffer)

        # Append buffer messages to our conversation history
        for msg in self._agent.last_messages_buffer:
            self._messages.append(msg)

        # Notify turn complete
        self._context_manager.notify_turn_complete()

        # Emit events
        self._events.emit(
            HarnessEvent.AGENT_RESPONSE,
            HarnessEventData(
                event=HarnessEvent.AGENT_RESPONSE,
                turn_number=self._turn_count,
                response=response,
            ),
        )
        self._events.emit(
            HarnessEvent.TURN_END,
            HarnessEventData(
                event=HarnessEvent.TURN_END,
                turn_number=self._turn_count,
                response=response,
            ),
        )

        # Check max turns
        if 0 < self.config.max_turns <= self._turn_count:
            self._stopped = True

        return response

    def chat_stream(self, user_input: str) -> Generator[ChatResponseChunk, None, None]:
        """Send a message, yield streaming chunks.

        Args:
            user_input: The user's message text.

        Yields:
            ChatResponseChunk objects. The final chunk has finished=True and
            contains the finished_response.
        """
        self._check_stopped()
        self._turn_count += 1

        self._events.emit(
            HarnessEvent.TURN_START,
            HarnessEventData(
                event=HarnessEvent.TURN_START,
                turn_number=self._turn_count,
                user_input=user_input,
            ),
        )

        user_msg = ChatMessage.create_user_message(user_input)
        self._messages.append(user_msg)
        self._context_manager.notify_user_message(user_msg)

        send_messages = self._prepare_messages()

        # Yield all chunks from the agent's streaming response
        finished_response = None
        for chunk in self._agent.get_streaming_response(
            messages=send_messages,
            tool_registry=self._tool_registry,
            settings=self._settings,
        ):
            yield chunk
            if chunk.finished and chunk.finished_response is not None:
                finished_response = chunk.finished_response

        # Post-process buffer (only safe after generator is fully exhausted)
        self._process_agent_buffer(self._agent.last_messages_buffer)
        for msg in self._agent.last_messages_buffer:
            self._messages.append(msg)

        self._context_manager.notify_turn_complete()

        if finished_response:
            self._events.emit(
                HarnessEvent.AGENT_RESPONSE,
                HarnessEventData(
                    event=HarnessEvent.AGENT_RESPONSE,
                    turn_number=self._turn_count,
                    response=finished_response,
                ),
            )
        self._events.emit(
            HarnessEvent.TURN_END,
            HarnessEventData(
                event=HarnessEvent.TURN_END,
                turn_number=self._turn_count,
                response=finished_response,
            ),
        )

        if 0 < self.config.max_turns <= self._turn_count:
            self._stopped = True

    def run(self, io_handler: IOHandler = None) -> None:
        """Start the interactive REPL loop.

        Reads user input, sends to agent, displays response. Loops until
        the user exits, max turns is reached, or budget is exceeded.

        Args:
            io_handler: I/O handler for input/output. Defaults to ConsoleIOHandler.
        """
        if io_handler is None:
            io_handler = ConsoleIOHandler()

        self._events.emit(
            HarnessEvent.HARNESS_START,
            HarnessEventData(event=HarnessEvent.HARNESS_START),
        )

        while not self._stopped:
            user_input = io_handler.get_input()
            if user_input is None:
                break

            if not user_input.strip():
                continue

            try:
                if self.config.streaming:
                    for chunk in self.chat_stream(user_input):
                        io_handler.on_chunk(chunk)
                else:
                    response = self.chat(user_input)
                    io_handler.on_text(response)
            except Exception as e:
                io_handler.on_error(e)
                self._events.emit(
                    HarnessEvent.ERROR,
                    HarnessEventData(
                        event=HarnessEvent.ERROR,
                        turn_number=self._turn_count,
                        error=e,
                    ),
                )

        self._events.emit(
            HarnessEvent.HARNESS_STOP,
            HarnessEventData(event=HarnessEvent.HARNESS_STOP),
        )

    # --- Internal Methods ---

    def _prepare_messages(self) -> List[ChatMessage]:
        """Build the message list for the agent: system prompt + context-managed conversation.

        Always returns a COPY so the agent's in-place mutations (appending tool call
        messages) do not affect self._messages.
        """
        system_msg = ChatMessage.create_system_message(self.config.system_prompt)
        full_messages = [system_msg] + self._messages

        tools_list = list(self._tool_registry.tools.values())
        trimmed = self._context_manager.prepare_messages(
            full_messages, tools=tools_list
        )

        # Always return a copy — the agent mutates the list in-place
        return list(trimmed)

    def _process_agent_buffer(self, buffer: List[ChatMessage]) -> None:
        """Walk the agent's last_messages_buffer and update context tracking.

        For each message in the buffer:
        - Assistant messages with token_usage: call on_response() for tracking
        - Assistant messages with tool calls: call notify_tool_call()
        - Tool messages: call notify_tool_result()
        """
        for msg in buffer:
            if msg.role == ChatMessageRole.Assistant:
                # Track token usage from every assistant response
                if msg.token_usage is not None:
                    self._context_manager.on_response(msg)
                # Notify tool calls
                if msg.contains_tool_call():
                    self._context_manager.notify_tool_call(msg)
            elif msg.role == ChatMessageRole.Tool:
                self._context_manager.notify_tool_result(msg)

    def _on_budget_exceeded(self, event_data) -> None:
        """Handler for context manager budget exceeded event."""
        self._budget_exceeded = True
        if self.config.stop_on_budget_exceeded:
            self._stopped = True

    def _check_stopped(self) -> None:
        """Raise if the harness is stopped."""
        if self._stopped:
            reason = "budget exceeded" if self._budget_exceeded else "max turns reached"
            raise RuntimeError(f"Harness is stopped ({reason}).")

    # --- State Access ---

    @property
    def messages(self) -> List[ChatMessage]:
        """Current conversation messages (copy)."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Number of completed user turns."""
        return self._turn_count

    @property
    def context_state(self):
        """Current context manager state snapshot."""
        return self._context_manager.state

    @property
    def context_manager(self) -> ContextManager:
        """The underlying ContextManager instance."""
        return self._context_manager

    @property
    def events(self) -> HarnessEventBus:
        """The harness event bus for registering handlers."""
        return self._events

    @property
    def is_stopped(self) -> bool:
        """Whether the harness has been stopped."""
        return self._stopped

    def reset(self) -> None:
        """Reset conversation state for a new conversation.

        Clears messages, turn count, and stopped flags.
        Does NOT reset the context manager's cumulative token tracking.
        """
        self._messages = []
        self._turn_count = 0
        self._stopped = False
        self._budget_exceeded = False

    def set_system_prompt(self, prompt: str) -> None:
        """Change the system prompt. Takes effect on the next chat() call."""
        self.config.system_prompt = prompt

    def set_settings(self, settings: ProviderSettings) -> None:
        """Change the provider settings (temperature, max_tokens, etc.)."""
        self._settings = settings


def create_harness(
    provider: ChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    max_context_tokens: int = 128000,
    max_turns: int = -1,
    streaming: bool = False,
    total_budget_tokens: Optional[int] = None,
    settings: Optional[ProviderSettings] = None,
    tools: Optional[List[FunctionTool]] = None,
    log_output: bool = False,
    **context_kwargs,
) -> AgentHarness:
    """Convenience factory: create a fully configured AgentHarness in one call.

    Args:
        provider: The LLM provider (OpenAI, Anthropic, Groq, Mistral, etc.).
        system_prompt: System prompt for the agent.
        max_context_tokens: Max context window size in tokens.
        max_turns: Maximum user turns (-1 for unlimited).
        streaming: Whether to stream responses by default in run().
        total_budget_tokens: Optional hard cap on total tokens for the conversation.
        settings: Provider settings (temperature, max_tokens, etc.).
        tools: Optional list of FunctionTools to register.
        log_output: Whether to enable agent-level logging.
        **context_kwargs: Additional kwargs for create_context_manager
            (e.g., strategy, reserve_tokens, keep_last_n).

    Returns:
        A configured AgentHarness ready for use.

    Example:
        harness = create_harness(
            provider=OpenAIChatAPI(api_key="sk-...", model="gpt-4o"),
            system_prompt="You are a coding assistant.",
            max_context_tokens=128000,
            streaming=True,
        )
        harness.add_tool(FunctionTool(my_function))
        harness.run()
    """
    context_config = {
        "max_context_tokens": max_context_tokens,
        **context_kwargs,
    }
    if total_budget_tokens is not None:
        context_config["total_budget_tokens"] = total_budget_tokens

    config = HarnessConfig(
        system_prompt=system_prompt,
        max_turns=max_turns,
        streaming=streaming,
        context_manager_config=context_config,
    )

    harness = AgentHarness(
        provider=provider,
        config=config,
        settings=settings,
        log_output=log_output,
    )

    if tools:
        harness.add_tools(tools)

    return harness

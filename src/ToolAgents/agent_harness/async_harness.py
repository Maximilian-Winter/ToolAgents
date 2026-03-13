# async_harness.py — Async version of AgentHarness wrapping AsyncChatToolAgent + ContextManager.
import asyncio
from typing import List, Optional, AsyncGenerator, TYPE_CHECKING

from ToolAgents.agents.chat_tool_agent import AsyncChatToolAgent
from ToolAgents.context_manager.context_manager import ContextManager, create_context_manager
from ToolAgents.context_manager.events import ContextEvent
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole, ToolCallResultContent
from ToolAgents.data_models.responses import ChatResponse, ChatResponseChunk
from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.provider.llm_provider import AsyncChatAPIProvider, ProviderSettings

from .config import HarnessConfig
from .events import HarnessEvent, HarnessEventData, HarnessEventBus
from .io_handlers import IOHandler, ConsoleIOHandler

if TYPE_CHECKING:
    from ToolAgents.extensions.manager import ExtensionManager as _ExtensionManager


class AsyncAgentHarness:
    """Async version of AgentHarness — wraps AsyncChatToolAgent + ContextManager.

    The async harness is the OUTER loop around the async agent. It manages:
    - The growing message list across user turns
    - System prompt (always prepended, never trimmed)
    - Context window management (trim before, track after)
    - Tool registration
    - Turn lifecycle and event hooks
    - Interactive async REPL or programmatic use

    Usage:
        harness = create_async_harness(provider=my_async_provider, system_prompt="You are helpful.")
        harness.add_tool(AsyncFunctionTool(my_async_function))

        # Programmatic
        print(await harness.chat("Hello!"))
        print(await harness.chat("Do something"))

        # Or interactive REPL
        await harness.run()
    """

    def __init__(
        self,
        provider: AsyncChatAPIProvider,
        system_prompt: str = "You are a helpful assistant.",
        config: Optional[HarnessConfig] = None,
        context_manager: Optional[ContextManager] = None,
        settings: Optional[ProviderSettings] = None,
        log_output: bool = False,
        extension_manager: Optional["_ExtensionManager"] = None,
    ):
        """Initialize the async harness.

        Args:
            provider: The async LLM provider.
            system_prompt: System prompt for the agent. Ignored if config is provided.
            config: Full HarnessConfig. If None, one is created from system_prompt.
            context_manager: Optional pre-configured ContextManager.
            settings: Provider settings (temperature, max_tokens, etc.).
            log_output: Whether to enable agent-level logging.
            extension_manager: Optional ExtensionManager for skill/extension support.
        """
        # Config
        if config is None:
            config = HarnessConfig(system_prompt=system_prompt)
        self.config = config

        # Agent (composition)
        self._agent = AsyncChatToolAgent(chat_api=provider, debug_output=log_output)

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

        # Extension manager (optional)
        self._extension_manager = extension_manager

    # --- Tool Management ---

    def add_tool(self, tool: FunctionTool) -> "AsyncAgentHarness":
        """Register a tool. Returns self for chaining."""
        self._tool_registry.add_tool(tool)
        return self

    def add_tools(self, tools: List[FunctionTool]) -> "AsyncAgentHarness":
        """Register multiple tools. Returns self for chaining."""
        self._tool_registry.add_tools(tools)
        return self

    def remove_tool(self, name: str) -> "AsyncAgentHarness":
        """Remove a tool by name. Returns self for chaining."""
        self._tool_registry.remove(name)
        return self

    # --- Core Async API ---

    async def chat(self, user_input: str) -> str:
        """Send a message, get a response string. Simplest async API.

        Args:
            user_input: The user's message text.

        Returns:
            The agent's response as a string.
        """
        response = await self.chat_response(user_input)
        return response.response

    async def chat_response(self, user_input: str) -> ChatResponse:
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

        # Call async agent — it handles tool-call loop internally
        response = await self._agent.get_response(
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

    async def chat_stream(self, user_input: str) -> AsyncGenerator[ChatResponseChunk, None]:
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

        # Yield all chunks from the agent's async streaming response
        finished_response = None
        async for chunk in self._agent.get_streaming_response(
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

    async def run(self, io_handler: IOHandler = None) -> None:
        """Start the interactive async REPL loop.

        Uses asyncio.to_thread for blocking input() calls so the event
        loop remains responsive.

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
            # Use asyncio.to_thread for blocking input
            user_input = await asyncio.to_thread(io_handler.get_input)
            if user_input is None:
                break

            if not user_input.strip():
                continue

            # Slash command interception for extensions
            if (user_input.strip().startswith("/")
                    and self._extension_manager is not None):
                command = user_input.strip()[1:]
                result = self._extension_manager.try_handle_command(command)
                if result is not None:
                    msg = ChatMessage.create_system_message(result.content)
                    self._messages.append(msg)
                    if result.pin_in_context:
                        self._context_manager.pin_message(msg.id)
                    if result.tools:
                        self.add_tools(result.tools)
                    io_handler.on_text(f"Skill '{command}' activated.")
                    continue

            try:
                if self.config.streaming:
                    async for chunk in self.chat_stream(user_input):
                        io_handler.on_chunk(chunk)
                else:
                    response = await self.chat(user_input)
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

    # --- Internal Methods (sync — no I/O) ---

    def _prepare_messages(self) -> List[ChatMessage]:
        """Build the message list for the agent: system prompt + context-managed conversation."""
        system_msg = ChatMessage.create_system_message(self.config.system_prompt)
        full_messages = [system_msg] + self._messages

        tools_list = list(self._tool_registry.tools.values())
        trimmed = self._context_manager.prepare_messages(
            full_messages, tools=tools_list
        )

        # Always return a copy — the agent mutates the list in-place
        return list(trimmed)

    def _process_agent_buffer(self, buffer: List[ChatMessage]) -> None:
        """Walk the agent's last_messages_buffer and update context tracking."""
        for msg in buffer:
            if msg.role == ChatMessageRole.Assistant:
                if msg.token_usage is not None:
                    self._context_manager.on_response(msg)
                if msg.contains_tool_call():
                    self._context_manager.notify_tool_call(msg)
            elif msg.role == ChatMessageRole.Tool:
                self._context_manager.notify_tool_result(msg)

                # Check for extension activation results that need pinning
                if self._extension_manager is not None:
                    for content in msg.content:
                        if (isinstance(content, ToolCallResultContent)
                                and content.tool_call_name == "activate_skill"):
                            pending = self._extension_manager._pending_activations
                            for act_name in list(pending.keys()):
                                act_result = pending[act_name]
                                if act_result.content in content.tool_call_result:
                                    if act_result.pin_in_context:
                                        self._context_manager.pin_message(msg.id)
                                    if act_result.tools:
                                        self.add_tools(act_result.tools)
                                    del pending[act_name]
                                    break

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
    def extension_manager(self):
        """The ExtensionManager, if one was provided."""
        return self._extension_manager

    @property
    def events(self) -> HarnessEventBus:
        """The harness event bus for registering handlers."""
        return self._events

    @property
    def is_stopped(self) -> bool:
        """Whether the harness has been stopped."""
        return self._stopped

    def reset(self) -> None:
        """Reset conversation state for a new conversation."""
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


def create_async_harness(
    provider: AsyncChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    max_context_tokens: int = 128000,
    max_turns: int = -1,
    streaming: bool = False,
    total_budget_tokens: Optional[int] = None,
    settings: Optional[ProviderSettings] = None,
    tools: Optional[List[FunctionTool]] = None,
    log_output: bool = False,
    extension_manager=None,
    **context_kwargs,
) -> AsyncAgentHarness:
    """Convenience factory: create a fully configured AsyncAgentHarness in one call.

    Args:
        provider: The async LLM provider.
        system_prompt: System prompt for the agent.
        max_context_tokens: Max context window size in tokens.
        max_turns: Maximum user turns (-1 for unlimited).
        streaming: Whether to stream responses by default in run().
        total_budget_tokens: Optional hard cap on total tokens for the conversation.
        settings: Provider settings (temperature, max_tokens, etc.).
        tools: Optional list of FunctionTools to register.
        log_output: Whether to enable agent-level logging.
        extension_manager: Optional ExtensionManager for skill/extension support.
        **context_kwargs: Additional kwargs for create_context_manager.

    Returns:
        A configured AsyncAgentHarness ready for use.
    """
    context_config = {
        "max_context_tokens": max_context_tokens,
        **context_kwargs,
    }
    if total_budget_tokens is not None:
        context_config["total_budget_tokens"] = total_budget_tokens

    # Append extension catalog to system prompt if extension_manager provided
    if extension_manager is not None:
        catalog = extension_manager.build_catalog()
        if catalog:
            system_prompt = system_prompt + "\n\n" + catalog

    config = HarnessConfig(
        system_prompt=system_prompt,
        max_turns=max_turns,
        streaming=streaming,
        context_manager_config=context_config,
    )

    harness = AsyncAgentHarness(
        provider=provider,
        config=config,
        settings=settings,
        log_output=log_output,
        extension_manager=extension_manager,
    )

    if tools:
        harness.add_tools(tools)

    # Register extension tools
    if extension_manager is not None:
        ext_tools = extension_manager.get_tools()
        if ext_tools:
            harness.add_tools(ext_tools)

    return harness

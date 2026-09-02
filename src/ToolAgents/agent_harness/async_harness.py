# async_harness.py - Async AgentHarness wrapper around AsyncChatToolAgent.
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Optional, Union, TYPE_CHECKING

from ToolAgents.agents.chat_tool_agent import AsyncChatToolAgent
from ToolAgents.context_manager.context_manager import ContextManager
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.data_models.responses import ChatResponse, ChatResponseChunk
from ToolAgents.function_tool import FunctionTool
from ToolAgents.provider.llm_provider import AsyncChatAPIProvider, ProviderSettings

from .config import HarnessConfig
from .events import HarnessEvent, HarnessEventBus, HarnessEventData
from .extensions import handle_slash_command
from .io_handlers import ConsoleIOHandler, IOHandler
from .prompt_composer import PromptComposer
from .runtime import HarnessRuntime
from .smart_messages import ExpiryAction, MessageLifecycle, SmartMessageManager

if TYPE_CHECKING:
    from ToolAgents.extensions.manager import ExtensionManager as _ExtensionManager


class AsyncAgentHarness:
    """Async wrapper around AsyncChatToolAgent + shared HarnessRuntime."""

    def __init__(
        self,
        provider: AsyncChatAPIProvider,
        system_prompt: str = "You are a helpful assistant.",
        config: Optional[HarnessConfig] = None,
        context_manager: Optional[ContextManager] = None,
        settings: Optional[ProviderSettings] = None,
        log_output: bool = False,
        extension_manager: Optional["_ExtensionManager"] = None,
        prompt_composer: Optional[PromptComposer] = None,
        smart_message_manager: Optional[SmartMessageManager] = None,
    ):
        if config is None:
            config = HarnessConfig(system_prompt=system_prompt)

        self._agent = AsyncChatToolAgent(chat_api=provider, debug_output=log_output)
        self._settings = settings
        self._runtime = HarnessRuntime(
            config=config,
            context_manager=context_manager,
            extension_manager=extension_manager,
            prompt_composer=prompt_composer,
            smart_message_manager=smart_message_manager,
        )

        self.config = self._runtime.config
        self._context_manager = self._runtime.context_manager
        self._prompt_composer = self._runtime.prompt_composer
        self._smart_message_manager = self._runtime.smart_messages
        self._tool_registry = self._runtime.tool_registry
        self._events = self._runtime.events
        self._extension_manager = self._runtime.extension_manager

    # --- Tool Management ---

    def add_tool(self, tool: FunctionTool) -> "AsyncAgentHarness":
        """Register a tool. Returns self for chaining."""
        self._runtime.add_tool(tool)
        return self

    def add_tools(self, tools: List[FunctionTool]) -> "AsyncAgentHarness":
        """Register multiple tools. Returns self for chaining."""
        self._runtime.add_tools(tools)
        return self

    def remove_tool(self, name: str) -> "AsyncAgentHarness":
        """Remove a tool by name. Returns self for chaining."""
        self._runtime.remove_tool(name)
        return self

    # --- Smart Message Convenience API ---

    def add_smart_message(
        self,
        message: ChatMessage,
        lifecycle: Optional[MessageLifecycle] = None,
    ) -> None:
        """Add a message with optional lifecycle to the conversation."""
        self._runtime.add_smart_message(message, lifecycle)

    def add_ephemeral_message(
        self,
        message: ChatMessage,
        ttl: int = 3,
        on_expire: ExpiryAction = ExpiryAction.REMOVE,
    ) -> None:
        """Add a message that expires after a number of turns."""
        self._runtime.add_ephemeral_message(message, ttl=ttl, on_expire=on_expire)

    def add_pinned_message(self, message: ChatMessage) -> None:
        """Add a permanent, pinned message."""
        self._runtime.add_pinned_message(message)

    # --- Core Async API ---

    async def chat(self, user_input: Union[str, ChatMessage]) -> str:
        """Send a message, get a response string."""
        response = await self.chat_response(user_input)
        return response.response

    async def chat_response(self, user_input: Union[str, ChatMessage]) -> ChatResponse:
        """Send a message, get a full ChatResponse with message history.

        `user_input` may be plain text or a pre-built ChatMessage (e.g. one carrying
        image attachments); see AgentRuntime.begin_turn.
        """
        self._runtime.begin_turn(user_input)
        send_messages = self._runtime.prepare_messages()

        response = await self._agent.get_response(
            messages=send_messages,
            tool_registry=self._tool_registry,
            settings=self._settings,
        )

        self._runtime.process_agent_buffer(self._agent.last_messages_buffer)
        self._runtime.complete_turn(response)
        return response

    async def chat_stream(self, user_input: Union[str, ChatMessage]) -> AsyncGenerator[ChatResponseChunk, None]:
        """Send a message, yield streaming chunks.

        `user_input` may be plain text or a pre-built ChatMessage (e.g. one carrying
        image attachments); see AgentRuntime.begin_turn.
        """
        self._runtime.begin_turn(user_input)
        send_messages = self._runtime.prepare_messages()

        finished_response = None
        async for chunk in self._agent.get_streaming_response(
            messages=send_messages,
            tool_registry=self._tool_registry,
            settings=self._settings,
        ):
            yield chunk
            if chunk.finished and chunk.finished_response is not None:
                finished_response = chunk.finished_response

        self._runtime.process_agent_buffer(self._agent.last_messages_buffer)
        self._runtime.complete_turn(finished_response)

    async def run(self, io_handler: IOHandler = None) -> None:
        """Start the interactive async REPL loop."""
        if io_handler is None:
            io_handler = ConsoleIOHandler()

        self._events.emit(
            HarnessEvent.HARNESS_START,
            HarnessEventData(event=HarnessEvent.HARNESS_START),
        )

        while not self._runtime.stopped:
            user_input = await asyncio.to_thread(io_handler.get_input)
            if user_input is None:
                break
            if not user_input.strip():
                continue

            confirmation = handle_slash_command(
                user_input,
                self._extension_manager,
                self._smart_message_manager,
                self.add_tools,
            )
            if confirmation is not None:
                io_handler.on_text(confirmation)
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
                        turn_number=self._runtime.turn_count,
                        error=e,
                    ),
                )

        self._events.emit(
            HarnessEvent.HARNESS_STOP,
            HarnessEventData(event=HarnessEvent.HARNESS_STOP),
        )

    # --- Compatibility helpers ---

    def _prepare_messages(self) -> List[ChatMessage]:
        """Build the message list for the agent."""
        return self._runtime.prepare_messages()

    def _process_agent_buffer(self, buffer: List[ChatMessage]) -> None:
        """Walk the agent's last_messages_buffer and update context tracking."""
        self._runtime.process_agent_buffer(buffer)

    def _on_budget_exceeded(self, event_data) -> None:
        self._runtime._on_budget_exceeded(event_data)

    def _check_stopped(self) -> None:
        self._runtime.check_stopped()

    # --- State Access ---

    @property
    def messages(self) -> List[ChatMessage]:
        """Current active conversation messages."""
        return self._runtime.messages

    @property
    def _messages(self) -> List[ChatMessage]:
        """Legacy alias for the active message list."""
        return self.messages

    @property
    def prompt_composer(self) -> PromptComposer:
        """The PromptComposer for modular system prompt management."""
        return self._prompt_composer

    @property
    def smart_messages(self) -> SmartMessageManager:
        """The SmartMessageManager for lifecycle-aware messages."""
        return self._smart_message_manager

    @property
    def turn_count(self) -> int:
        """Number of user turns started."""
        return self._runtime.turn_count

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
        return self._runtime.stopped

    def reset(self) -> None:
        """Reset conversation state for a new conversation."""
        self._runtime.reset()

    def set_system_prompt(self, prompt: str) -> None:
        """Change the base instructions prompt module."""
        self._runtime.set_system_prompt(prompt)

    def set_settings(self, settings: ProviderSettings) -> None:
        """Change the provider settings."""
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
    prompt_composer: Optional[PromptComposer] = None,
    smart_message_manager: Optional[SmartMessageManager] = None,
    **context_kwargs,
) -> AsyncAgentHarness:
    """Convenience factory: create a fully configured AsyncAgentHarness."""
    from .factories import create_async_harness as _create_async_harness

    return _create_async_harness(
        provider=provider,
        system_prompt=system_prompt,
        max_context_tokens=max_context_tokens,
        max_turns=max_turns,
        streaming=streaming,
        total_budget_tokens=total_budget_tokens,
        settings=settings,
        tools=tools,
        log_output=log_output,
        extension_manager=extension_manager,
        prompt_composer=prompt_composer,
        smart_message_manager=smart_message_manager,
        **context_kwargs,
    )


def create_async_harness_with_extensions(
    provider: AsyncChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    skill_paths: Optional[List] = None,
    scan_defaults: bool = True,
    **kwargs,
) -> AsyncAgentHarness:
    """Create an async harness with extension system pre-configured."""
    from .factories import create_async_harness_with_extensions as _create

    return _create(
        provider=provider,
        system_prompt=system_prompt,
        skill_paths=skill_paths,
        scan_defaults=scan_defaults,
        **kwargs,
    )

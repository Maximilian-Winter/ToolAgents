# harness.py — Core AgentHarness wrapping ChatToolAgent + ContextManager.
# Turn cycle order:
#   1. PromptComposer.compile() → system message
#   2. SmartMessageManager.tick() → process lifecycles, expire messages
#   3. SmartMessageManager.get_active_messages() → filtered conversation
#   4. ContextManager.prepare_messages() → trim if over budget
#   5. Send to LLM
#   6. Post-process: track tokens, handle tool calls, add response messages

from typing import List, Optional, Generator, TYPE_CHECKING

from ToolAgents.agents.chat_tool_agent import ChatToolAgent
from ToolAgents.context_manager.context_manager import ContextManager, create_context_manager
from ToolAgents.context_manager.events import ContextEvent
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole, ToolCallResultContent
from ToolAgents.data_models.responses import ChatResponse, ChatResponseChunk
from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.provider.llm_provider import ChatAPIProvider, ProviderSettings

from .config import HarnessConfig
from .events import HarnessEvent, HarnessEventData, HarnessEventBus
from .io_handlers import IOHandler, ConsoleIOHandler

# New imports for the modular systems
from .prompt_composer import PromptComposer, PromptModule, create_prompt_composer
from .smart_messages import (
    SmartMessageManager,
    MessageLifecycle,
    ExpiryAction,
    ExpiryResult,
)

if TYPE_CHECKING:
    from ToolAgents.extensions.manager import ExtensionManager as _ExtensionManager


class AgentHarness:
    """Wraps ChatToolAgent + ContextManager into an interactive runtime.

    The harness is the OUTER loop around the agent. It manages:
    - Modular system prompt composition (PromptComposer)
    - Lifecycle-aware conversation messages (SmartMessageManager)
    - Context window management (ContextManager)
    - Tool registration
    - Turn lifecycle and event hooks
    - Interactive REPL or programmatic use

    Turn cycle:
        1. compile system prompt from modules
        2. tick smart messages (expire, summarize, archive)
        3. get active messages
        4. context trim if needed
        5. call LLM
        6. post-process response

    Usage:
        harness = create_harness(provider=my_provider, system_prompt="You are helpful.")
        harness.add_tool(FunctionTool(my_function))

        # Add a dynamic prompt module
        harness.prompt_composer.add_module(
            "memory", position=10,
            content_fn=lambda: core_memory.build_context(),
            prefix="<core_memory>", suffix="</core_memory>"
        )

        # Add an ephemeral message
        harness.add_smart_message(
            ChatMessage.create_system_message("Temporary context"),
            lifecycle=MessageLifecycle(ttl=3, on_expire=ExpiryAction.REMOVE)
        )

        print(harness.chat("Hello!"))
    """

    def __init__(
        self,
        provider: ChatAPIProvider,
        system_prompt: str = "You are a helpful assistant.",
        config: Optional[HarnessConfig] = None,
        context_manager: Optional[ContextManager] = None,
        settings: Optional[ProviderSettings] = None,
        log_output: bool = False,
        extension_manager: Optional["_ExtensionManager"] = None,
        prompt_composer: Optional[PromptComposer] = None,
        smart_message_manager: Optional[SmartMessageManager] = None,
    ):
        """Initialize the harness.

        Args:
            provider: The LLM provider.
            system_prompt: System prompt for the agent. Used to create a default
                PromptComposer if prompt_composer is not provided. Ignored if
                config is provided (config.system_prompt is used instead).
            config: Full HarnessConfig. If None, one is created from system_prompt.
            context_manager: Optional pre-configured ContextManager.
            settings: Provider settings (temperature, max_tokens, etc.).
            log_output: Whether to enable agent-level logging.
            extension_manager: Optional ExtensionManager for skill/extension support.
            prompt_composer: Optional pre-configured PromptComposer. If None, one
                is created from the system_prompt with a single "instructions" module.
            smart_message_manager: Optional pre-configured SmartMessageManager.
                If None, one is created with default settings.
        """
        # Config
        if config is None:
            config = HarnessConfig(system_prompt=system_prompt)
        self.config = config

        # Agent (composition)
        self._agent = ChatToolAgent(chat_api=provider, log_output=log_output)

        # Context manager
        if context_manager is not None:
            self._context_manager = context_manager
        elif config.context_manager_config:
            self._context_manager = create_context_manager(**config.context_manager_config)
        else:
            self._context_manager = create_context_manager()

        # Prompt composer
        if prompt_composer is not None:
            self._prompt_composer = prompt_composer
        else:
            self._prompt_composer = create_prompt_composer(config.system_prompt)

        # Smart message manager
        if smart_message_manager is not None:
            self._smart_message_manager = smart_message_manager
        else:
            self._smart_message_manager = SmartMessageManager()

        # Tool registry
        self._tool_registry = ToolRegistry()

        # Provider settings
        self._settings = settings

        # Conversation state (legacy _messages kept for backward compat)
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

    # --- Smart Message Convenience API ---

    def add_smart_message(
        self,
        message: ChatMessage,
        lifecycle: Optional[MessageLifecycle] = None,
    ) -> None:
        """Add a message with optional lifecycle to the conversation.

        This is the primary way to add messages to the conversation when
        using smart message features.

        Args:
            message: The ChatMessage to add.
            lifecycle: Optional lifecycle configuration. None = permanent.
        """
        self._smart_message_manager.add_message(message, lifecycle)

    def add_ephemeral_message(
        self,
        message: ChatMessage,
        ttl: int = 3,
        on_expire: ExpiryAction = ExpiryAction.REMOVE,
    ) -> None:
        """Convenience: add a message that expires after a number of turns.

        Args:
            message: The ChatMessage to add.
            ttl: Number of turns before expiry.
            on_expire: What to do on expiry. Defaults to REMOVE.
        """
        self._smart_message_manager.add_message(
            message,
            MessageLifecycle(ttl=ttl, on_expire=on_expire),
        )

    def add_pinned_message(self, message: ChatMessage) -> None:
        """Convenience: add a permanent, pinned message.

        Pinned messages are exempt from both lifecycle expiry and context
        trimming.

        Args:
            message: The ChatMessage to add.
        """
        self._smart_message_manager.add_message(
            message,
            MessageLifecycle(pinned=True),
        )

    # --- Core API ---

    def chat(self, user_input: str) -> str:
        """Send a message, get a response string. Simplest API."""
        response = self.chat_response(user_input)
        return response.response

    def chat_response(self, user_input: str) -> ChatResponse:
        """Send a message, get a full ChatResponse with message history."""
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

        # Add user message to smart message manager
        user_msg = ChatMessage.create_user_message(user_input)
        self._smart_message_manager.add_message(user_msg)
        self._context_manager.notify_user_message(user_msg)

        # Prepare messages: compile prompt → tick lifecycles → trim context
        send_messages = self._prepare_messages()

        # Call agent
        response = self._agent.get_response(
            messages=send_messages,
            tool_registry=self._tool_registry,
            settings=self._settings,
        )

        # Post-process
        self._process_agent_buffer(self._agent.last_messages_buffer)

        # Add agent response messages to smart message manager
        for msg in self._agent.last_messages_buffer:
            self._smart_message_manager.add_message(msg)

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

        if 0 < self.config.max_turns <= self._turn_count:
            self._stopped = True

        return response

    def chat_stream(self, user_input: str) -> Generator[ChatResponseChunk, None, None]:
        """Send a message, yield streaming chunks."""
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
        self._smart_message_manager.add_message(user_msg)
        self._context_manager.notify_user_message(user_msg)

        send_messages = self._prepare_messages()

        finished_response = None
        for chunk in self._agent.get_streaming_response(
            messages=send_messages,
            tool_registry=self._tool_registry,
            settings=self._settings,
        ):
            yield chunk
            if chunk.finished and chunk.finished_response is not None:
                finished_response = chunk.finished_response

        self._process_agent_buffer(self._agent.last_messages_buffer)
        for msg in self._agent.last_messages_buffer:
            self._smart_message_manager.add_message(msg)

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
        """Start the interactive REPL loop."""
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

            # Slash command interception for extensions
            if (user_input.strip().startswith("/")
                    and self._extension_manager is not None):
                command = user_input.strip()[1:]
                result = self._extension_manager.try_handle_command(command)
                if result is not None:
                    msg = ChatMessage.create_system_message(result.content)
                    # Add as smart message — skills can be ephemeral or permanent
                    lifecycle = MessageLifecycle(pinned=result.pin_in_context)
                    self._smart_message_manager.add_message(msg, lifecycle)
                    if result.tools:
                        self.add_tools(result.tools)
                    io_handler.on_text(f"Skill '{command}' activated.")
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
        """Build the message list for the agent.

        Order of operations:
        1. Compile system prompt from PromptComposer modules
        2. Tick smart message lifecycles (process expiry)
        3. Get active (non-expired) messages
        4. Sync pinned message IDs to the context manager
        5. Run context trimming if needed
        6. Return a copy for the agent

        Returns:
            The assembled and trimmed message list.
        """
        # 1. Compile system prompt from modules
        system_prompt = self._prompt_composer.compile()
        system_msg = ChatMessage.create_system_message(system_prompt)

        # 2. Tick smart message lifecycles
        expiry_result = self._smart_message_manager.tick()

        # Emit expiry event if anything changed
        if expiry_result.has_changes:
            self._events.emit(
                HarnessEvent.TURN_START,  # Reuse TURN_START or add a new event type
                HarnessEventData(
                    event=HarnessEvent.TURN_START,
                    turn_number=self._turn_count,
                    metadata={"expiry_result": expiry_result},
                ),
            )

        # 3. Get active messages
        active_messages = self._smart_message_manager.get_active_messages()

        # 4. Sync pinned IDs to context manager
        pinned_ids = self._smart_message_manager.get_pinned_message_ids()
        self._context_manager.state.pinned_message_ids = pinned_ids

        # 5. Build full message list and run context trimming
        full_messages = [system_msg] + active_messages

        tools_list = list(self._tool_registry.tools.values())
        trimmed = self._context_manager.prepare_messages(
            full_messages, tools=tools_list
        )

        # 6. Return a copy
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
                                        self._smart_message_manager.pin_message(msg.id)
                                    if act_result.tools:
                                        self.add_tools(act_result.tools)
                                    del pending[act_name]
                                    break

    def _on_budget_exceeded(self, event_data) -> None:
        self._budget_exceeded = True
        if self.config.stop_on_budget_exceeded:
            self._stopped = True

    def _check_stopped(self) -> None:
        if self._stopped:
            reason = "budget exceeded" if self._budget_exceeded else "max turns reached"
            raise RuntimeError(f"Harness is stopped ({reason}).")

    # --- State Access ---

    @property
    def messages(self) -> List[ChatMessage]:
        """Current active conversation messages (from smart message manager)."""
        return self._smart_message_manager.get_active_messages()

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
        return self._turn_count

    @property
    def context_state(self):
        return self._context_manager.state

    @property
    def context_manager(self) -> ContextManager:
        return self._context_manager

    @property
    def extension_manager(self):
        return self._extension_manager

    @property
    def events(self) -> HarnessEventBus:
        return self._events

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    def reset(self) -> None:
        """Reset conversation state for a new conversation."""
        self._smart_message_manager.clear()
        self._turn_count = 0
        self._stopped = False
        self._budget_exceeded = False

    def set_system_prompt(self, prompt: str) -> None:
        """Change the base instructions prompt module.

        If using the PromptComposer, this updates the "instructions" module.
        For more control, use prompt_composer directly.
        """
        if self._prompt_composer.has_module("instructions"):
            self._prompt_composer.update_module("instructions", content=prompt)
        else:
            self._prompt_composer.add_module("instructions", position=0, content=prompt)
        self.config.system_prompt = prompt

    def set_settings(self, settings: ProviderSettings) -> None:
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
    extension_manager=None,
    prompt_composer: Optional[PromptComposer] = None,
    smart_message_manager: Optional[SmartMessageManager] = None,
    **context_kwargs,
) -> AgentHarness:
    """Convenience factory: create a fully configured AgentHarness.

    Args:
        provider: The LLM provider.
        system_prompt: System prompt for the agent.
        max_context_tokens: Max context window size in tokens.
        max_turns: Maximum user turns (-1 for unlimited).
        streaming: Whether to stream responses by default in run().
        total_budget_tokens: Optional hard cap on total tokens.
        settings: Provider settings.
        tools: Optional list of FunctionTools to register.
        log_output: Whether to enable agent-level logging.
        extension_manager: Optional ExtensionManager.
        prompt_composer: Optional pre-configured PromptComposer.
        smart_message_manager: Optional pre-configured SmartMessageManager.
        **context_kwargs: Additional kwargs for create_context_manager.

    Returns:
        A configured AgentHarness.

    Example:
        harness = create_harness(
            provider=OpenAIChatAPI(api_key="sk-...", model="gpt-4o"),
            system_prompt="You are a coding assistant.",
            max_context_tokens=128000,
            streaming=True,
        )

        # Add a dynamic core memory module
        harness.prompt_composer.add_module(
            "core_memory", position=10,
            content_fn=lambda: my_memory.build_context(),
            prefix="<core_memory>", suffix="</core_memory>"
        )

        # Add an ephemeral context injection
        harness.add_ephemeral_message(
            ChatMessage.create_system_message("The user prefers dark mode."),
            ttl=5
        )

        harness.run()
    """
    context_config = {
        "max_context_tokens": max_context_tokens,
        **context_kwargs,
    }
    if total_budget_tokens is not None:
        context_config["total_budget_tokens"] = total_budget_tokens

    # Build prompt composer if not provided
    if prompt_composer is None:
        prompt_composer = create_prompt_composer(system_prompt)

    # If extension_manager provided, add catalog as a prompt module
    if extension_manager is not None:
        catalog = extension_manager.build_catalog()
        if catalog:
            prompt_composer.add_module(
                name="extension_catalog",
                position=100,  # late in the prompt
                content=catalog,
            )

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
        extension_manager=extension_manager,
        prompt_composer=prompt_composer,
        smart_message_manager=smart_message_manager,
    )

    if tools:
        harness.add_tools(tools)

    # Register extension tools
    if extension_manager is not None:
        ext_tools = extension_manager.get_tools()
        if ext_tools:
            harness.add_tools(ext_tools)

    return harness


def create_harness_with_extensions(
    provider: ChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    skill_paths: Optional[List] = None,
    scan_defaults: bool = True,
    **kwargs,
) -> AgentHarness:
    """Create a harness with extension system pre-configured.

    Args:
        provider: The LLM provider.
        system_prompt: Base system prompt.
        skill_paths: Additional directories to scan for skills.
        scan_defaults: Whether to scan default locations.
        **kwargs: Additional arguments passed to create_harness().

    Returns:
        A configured AgentHarness with extensions enabled.
    """
    from pathlib import Path
    from ToolAgents.extensions import ExtensionManager, SkillFolderHandler, ExtensionScanPath

    manager = ExtensionManager()
    manager.register_handler(SkillFolderHandler())

    if scan_defaults:
        cwd = Path.cwd()
        home = Path.home()
        for subdir in [".agents/skills", ".claude/skills"]:
            project_path = cwd / subdir
            if project_path.is_dir():
                manager.add_scan_path(ExtensionScanPath(
                    path=project_path, scope="project", priority=10,
                ))
        for subdir in [".agents/skills", ".claude/skills"]:
            user_path = home / subdir
            if user_path.is_dir():
                manager.add_scan_path(ExtensionScanPath(
                    path=user_path, scope="user", priority=0,
                ))

    if skill_paths:
        for sp in skill_paths:
            manager.add_scan_path(ExtensionScanPath(
                path=Path(sp), scope="project", priority=10,
            ))

    manager.discover()

    return create_harness(
        provider=provider,
        system_prompt=system_prompt,
        extension_manager=manager,
        **kwargs,
    )

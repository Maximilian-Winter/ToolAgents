from __future__ import annotations

from typing import List, Optional, Union, TYPE_CHECKING

from ToolAgents.context_manager.context_manager import ContextManager, create_context_manager
from ToolAgents.context_manager.events import ContextEvent
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole
from ToolAgents.data_models.responses import ChatResponse
from ToolAgents.function_tool import FunctionTool, ToolRegistry

from .config import HarnessConfig
from .events import HarnessEvent, HarnessEventBus, HarnessEventData
from .extensions import handle_extension_tool_result
from .prompt_composer import PromptComposer, create_prompt_composer
from .smart_messages import ExpiryAction, MessageLifecycle, SmartMessageManager

if TYPE_CHECKING:
    from ToolAgents.extensions.manager import ExtensionManager as _ExtensionManager


class HarnessRuntime:
    """Shared turn lifecycle for sync and async harness wrappers."""

    def __init__(
        self,
        config: HarnessConfig,
        context_manager: Optional[ContextManager] = None,
        extension_manager: Optional["_ExtensionManager"] = None,
        prompt_composer: Optional[PromptComposer] = None,
        smart_message_manager: Optional[SmartMessageManager] = None,
    ) -> None:
        self.config = config

        if context_manager is not None:
            self.context_manager = context_manager
        elif config.context_manager_config:
            self.context_manager = create_context_manager(**config.context_manager_config)
        else:
            self.context_manager = create_context_manager()

        self.prompt_composer = (
            prompt_composer
            if prompt_composer is not None
            else create_prompt_composer(config.system_prompt)
        )
        self.smart_messages = (
            smart_message_manager
            if smart_message_manager is not None
            else SmartMessageManager()
        )
        self.tool_registry = ToolRegistry()
        self.extension_manager = extension_manager
        self.events = HarnessEventBus()

        self.turn_count = 0
        self.stopped = False
        self.budget_exceeded = False

        self.context_manager.events.on(
            ContextEvent.BUDGET_EXCEEDED,
            self._on_budget_exceeded,
        )

    def add_tool(self, tool: FunctionTool) -> None:
        self.tool_registry.add_tool(tool)

    def add_tools(self, tools: List[FunctionTool]) -> None:
        self.tool_registry.add_tools(tools)

    def remove_tool(self, name: str) -> None:
        self.tool_registry.remove(name)

    def add_smart_message(
        self,
        message: ChatMessage,
        lifecycle: Optional[MessageLifecycle] = None,
    ) -> None:
        self.smart_messages.add_message(message, lifecycle)

    def add_ephemeral_message(
        self,
        message: ChatMessage,
        ttl: int = 3,
        on_expire: ExpiryAction = ExpiryAction.REMOVE,
    ) -> None:
        self.smart_messages.add_message(
            message,
            MessageLifecycle(ttl=ttl, on_expire=on_expire),
        )

    def add_pinned_message(self, message: ChatMessage) -> None:
        self.smart_messages.add_message(message, MessageLifecycle(pinned=True))

    def begin_turn(self, user_input: Union[str, ChatMessage]) -> None:
        """Start a turn from the user's input.

        `user_input` is usually the plain text the user typed. It may also be a
        pre-built ChatMessage, which lets a turn carry more than text -- images or
        other BinaryContent attachments alongside the prompt -- without this layer
        needing to know how the message was assembled. A string is wrapped into a
        text-only user message exactly as before.
        """
        self.check_stopped()
        self.turn_count += 1
        message_text = (
            user_input if isinstance(user_input, str) else user_input.get_text_content()
        )
        self.events.emit(
            HarnessEvent.TURN_START,
            HarnessEventData(
                event=HarnessEvent.TURN_START,
                turn_number=self.turn_count,
                user_input=message_text,
            ),
        )

        user_message = (
            user_input
            if isinstance(user_input, ChatMessage)
            else ChatMessage.create_user_message(user_input)
        )
        self.smart_messages.add_message(user_message)
        self.context_manager.notify_user_message(user_message)

    def prepare_messages(self) -> List[ChatMessage]:
        system_prompt = self.prompt_composer.compile()
        system_message = ChatMessage.create_system_message(system_prompt)

        expiry_result = self.smart_messages.tick()
        if expiry_result.has_changes:
            self.events.emit(
                HarnessEvent.TURN_START,
                HarnessEventData(
                    event=HarnessEvent.TURN_START,
                    turn_number=self.turn_count,
                    metadata={"expiry_result": expiry_result},
                ),
            )

        active_messages = self.smart_messages.get_active_messages()
        self.context_manager.set_pinned_message_ids(
            self.smart_messages.get_pinned_message_ids()
        )

        full_messages = [system_message] + active_messages
        trimmed = self.context_manager.prepare_messages(
            full_messages,
            tools=list(self.tool_registry.tools.values()),
        )
        return list(trimmed)

    def process_agent_buffer(self, buffer: List[ChatMessage]) -> None:
        for message in buffer:
            self.smart_messages.add_message(message)

        for message in buffer:
            if message.role == ChatMessageRole.Assistant:
                if message.token_usage is not None:
                    self.context_manager.on_response(message)
                if message.contains_tool_call():
                    self.context_manager.notify_tool_call(message)
            elif message.role == ChatMessageRole.Tool:
                self.context_manager.notify_tool_result(message)
                handle_extension_tool_result(
                    message,
                    self.extension_manager,
                    self.smart_messages,
                    self.add_tools,
                )

    def complete_turn(self, response: Optional[ChatResponse] = None) -> None:
        self.context_manager.notify_turn_complete()

        if response is not None:
            self.events.emit(
                HarnessEvent.AGENT_RESPONSE,
                HarnessEventData(
                    event=HarnessEvent.AGENT_RESPONSE,
                    turn_number=self.turn_count,
                    response=response,
                ),
            )

        self.events.emit(
            HarnessEvent.TURN_END,
            HarnessEventData(
                event=HarnessEvent.TURN_END,
                turn_number=self.turn_count,
                response=response,
            ),
        )

        if 0 < self.config.max_turns <= self.turn_count:
            self.stopped = True

    def reset(self) -> None:
        self.smart_messages.clear()
        self.context_manager.reset()
        self.turn_count = 0
        self.stopped = False
        self.budget_exceeded = False

    def set_system_prompt(self, prompt: str) -> None:
        if self.prompt_composer.has_module("instructions"):
            self.prompt_composer.update_module("instructions", content=prompt)
        else:
            self.prompt_composer.add_module("instructions", position=0, content=prompt)
        self.config.system_prompt = prompt

    @property
    def messages(self) -> List[ChatMessage]:
        return self.smart_messages.get_active_messages()

    def check_stopped(self) -> None:
        if self.stopped:
            reason = "budget exceeded" if self.budget_exceeded else "max turns reached"
            raise RuntimeError(f"Harness is stopped ({reason}).")

    def _on_budget_exceeded(self, event_data) -> None:
        self.budget_exceeded = True
        if self.config.stop_on_budget_exceeded:
            self.stopped = True

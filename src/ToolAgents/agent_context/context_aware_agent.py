from typing import Optional, List, Generator, AsyncGenerator
from datetime import datetime

from ToolAgents.agents.chat_tool_agent import ChatToolAgent, AsyncChatToolAgent
from ToolAgents.data_models.responses import ChatResponse, ChatResponseChunk
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.function_tool import ToolRegistry
from ToolAgents.provider.llm_provider import (
    ChatAPIProvider,
    AsyncChatAPIProvider,
    ProviderSettings,
    StreamingChatMessage
)
from ToolAgents.utilities.logging import EasyLogger
from .agent_context import AgentContext


class ContextAwareChatToolAgent(ChatToolAgent):
    """
    Enhanced ChatToolAgent with integrated context management.
    Inherits all functionality from ChatToolAgent while adding context awareness.
    """

    def __init__(
            self,
            chat_api: ChatAPIProvider,
            context: Optional[AgentContext] = None,
            log_output: bool = False,
            log_to_file: bool = False,
    ):
        super().__init__(chat_api, log_output, log_to_file)
        self.context = context or AgentContext()
        self.auto_update_context = True

    def set_system_prompt(self, prompt: str) -> None:
        """Configure agent behavior via system prompt."""
        self.context.set_system_prompt(prompt)

    def prepare_messages_with_context(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Combine context with new messages for API calls.

        Returns context messages followed by new messages,
        avoiding duplicates.
        """
        context_messages = self.context.get_messages()

        if not context_messages:
            return messages

        # Avoid duplicate system messages
        if messages and messages[0].role == context_messages[-1].role:
            if messages[0].id == context_messages[-1].id:
                return context_messages + messages[1:]

        return context_messages + messages

    def step(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[ProviderSettings] = None,
            reset_last_messages_buffer: bool = True,
            use_context: bool = True,
    ) -> ChatMessage:
        """
        Enhanced step that optionally includes context.

        Args:
            messages: Input messages
            tool_registry: Available tools
            settings: Provider settings
            reset_last_messages_buffer: Clear internal buffer
            use_context: Whether to prepend context messages

        Returns:
            Agent's response message
        """
        prepared_messages = (
            self.prepare_messages_with_context(messages)
            if use_context
            else messages
        )

        result = super().step(
            prepared_messages,
            tool_registry,
            settings,
            reset_last_messages_buffer
        )

        if self.auto_update_context and use_context:
            # Add the original messages (not prepared) to avoid duplicating context
            self.context.extend(messages)
            self.context.append(result)

        return result

    def stream_step(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[ProviderSettings] = None,
            reset_last_messages_buffer: bool = True,
            use_context: bool = True,
    ) -> Generator[StreamingChatMessage, None, None]:
        """
        Enhanced streaming step with context support.
        """
        prepared_messages = (
            self.prepare_messages_with_context(messages)
            if use_context
            else messages
        )

        final_message = None
        for chunk in super().stream_step(
                prepared_messages,
                tool_registry,
                settings,
                reset_last_messages_buffer
        ):
            if chunk.get_finished():
                final_message = chunk.get_finished_chat_message()
            yield chunk

        if self.auto_update_context and use_context and final_message:
            self.context.extend(messages)
            self.context.append(final_message)

    def get_response(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[ProviderSettings] = None,
            reset_last_messages_buffer: bool = True,
            use_context: bool = True,
    ) -> ChatResponse:
        """
        Get complete response with automatic context management.
        """
        prepared_messages = (
            self.prepare_messages_with_context(messages)
            if use_context
            else messages
        )

        response = super().get_response(
            prepared_messages,
            tool_registry,
            settings,
            reset_last_messages_buffer
        )

        if self.auto_update_context and use_context:
            self.context.extend(messages)
            # Add all messages from the response that aren't already in context
            for msg in response.messages:
                if msg not in self.context.message_buffer:
                    self.context.append(msg)

        return response

    def get_streaming_response(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[ProviderSettings] = None,
            reset_last_messages_buffer: bool = True,
            use_context: bool = True,
    ) -> Generator[ChatResponseChunk, None, None]:
        """
        Get streaming response with context management.
        """
        prepared_messages = (
            self.prepare_messages_with_context(messages)
            if use_context
            else messages
        )

        final_response = None
        for chunk in super().get_streaming_response(
                prepared_messages,
                tool_registry,
                settings,
                reset_last_messages_buffer
        ):
            if chunk.finished and chunk.finished_response:
                final_response = chunk.finished_response
            yield chunk

        if self.auto_update_context and use_context and final_response:
            self.context.extend(messages)
            for msg in final_response.messages:
                if msg not in self.context.message_buffer:
                    self.context.append(msg)

    def chat(
            self,
            user_input: str,
            tool_registry: Optional[ToolRegistry] = None,
            settings: Optional[ProviderSettings] = None
    ) -> str:
        """
        Simple chat interface that automatically manages context.
        """
        user_message = ChatMessage.create_user_message(user_input)
        response = self.get_response(
            [user_message],
            tool_registry,
            settings,
            use_context=True
        )
        return response.response

    def reset_context(self, preserve_system: bool = True) -> None:
        """
        Clear conversation history.

        Args:
            preserve_system: Keep system prompt if True
        """
        if preserve_system:
            self.context.clear_buffer()
        else:
            self.context.reset()

    def fork(self) -> "ContextAwareChatToolAgent":
        """
        Create agent copy with independent context for parallel exploration.
        """
        return ContextAwareChatToolAgent(
            chat_api=self.chat_api,
            context=self.context.fork(),
            log_output=self.log_output,
            log_to_file=False
        )

    def compress_context(self, target_size: int) -> None:
        """
        Reduce context to manage token limits.
        """
        self.context.compress(target_size)

    def get_context_size(self) -> int:
        """
        Get current context message count.
        """
        return len(self.context.get_messages())

    def get_context_token_estimate(self) -> int:
        """
        Estimate tokens in current context.
        """
        return self.context.get_token_estimate()


class AsyncContextAwareChatToolAgent(AsyncChatToolAgent):
    """
    Async version with context management.
    """

    def __init__(
            self,
            chat_api: AsyncChatAPIProvider,
            context: Optional[AgentContext] = None,
            debug_output: bool = False
    ):
        super().__init__(chat_api, debug_output)
        self.context = context or AgentContext()
        self.auto_update_context = True

    def set_system_prompt(self, prompt: str) -> None:
        """Configure agent behavior via system prompt."""
        self.context.set_system_prompt(prompt)

    def prepare_messages_with_context(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Combine context with new messages."""
        context_messages = self.context.get_messages()

        if not context_messages:
            return messages

        if messages and messages[0].role == context_messages[-1].role:
            if messages[0].id == context_messages[-1].id:
                return context_messages + messages[1:]

        return context_messages + messages

    async def step(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[ProviderSettings] = None,
            reset_last_messages_buffer: bool = True,
            use_context: bool = True,
    ) -> ChatMessage:
        """Enhanced async step with context."""
        prepared_messages = (
            self.prepare_messages_with_context(messages)
            if use_context
            else messages
        )

        result = await super().step(
            prepared_messages,
            tool_registry,
            settings,
            reset_last_messages_buffer
        )

        if self.auto_update_context and use_context:
            self.context.extend(messages)
            self.context.append(result)

        return result

    async def get_response(
            self,
            messages: List[ChatMessage],
            tool_registry: ToolRegistry = None,
            settings: Optional[ProviderSettings] = None,
            reset_last_messages_buffer: bool = True,
            use_context: bool = True,
    ) -> ChatResponse:
        """Get async response with context management."""
        prepared_messages = (
            self.prepare_messages_with_context(messages)
            if use_context
            else messages
        )

        response = await super().get_response(
            prepared_messages,
            tool_registry,
            settings,
            reset_last_messages_buffer
        )

        if self.auto_update_context and use_context:
            self.context.extend(messages)
            for msg in response.messages:
                if msg not in self.context.message_buffer:
                    self.context.append(msg)

        return response

    async def chat(
            self,
            user_input: str,
            tool_registry: Optional[ToolRegistry] = None,
            settings: Optional[ProviderSettings] = None
    ) -> str:
        """Simple async chat interface."""
        user_message = ChatMessage.create_user_message(user_input)
        response = await self.get_response(
            [user_message],
            tool_registry,
            settings,
            use_context=True
        )
        return response.response

    def reset_context(self, preserve_system: bool = True) -> None:
        """Clear conversation history."""
        if preserve_system:
            self.context.clear_buffer()
        else:
            self.context.reset()

    def fork(self) -> "AsyncContextAwareChatToolAgent":
        """Create async agent copy with independent context."""
        return AsyncContextAwareChatToolAgent(
            chat_api=self.chat_api,
            context=self.context.fork(),
            debug_output=self.debug_output
        )
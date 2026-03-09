# summarization.py — Summarization providers for context trimming.
from abc import ABC, abstractmethod
from typing import List, Optional

from ToolAgents.data_models.messages import (
    ChatMessage,
    TextContent,
    ToolCallContent,
    ToolCallResultContent,
    BinaryContent,
)


class SummarizationProvider(ABC):
    """Abstract base for producing text summaries of message lists.

    Used by SummarizeAndTrimStrategy to condense older messages
    into a compact representation before dropping them.
    """

    @abstractmethod
    def summarize(self, messages: List[ChatMessage]) -> str:
        """Produce a text summary of the given messages.

        Args:
            messages: The messages to summarize.

        Returns:
            A string summary capturing the key information.
        """
        ...


class LLMSummarizationProvider(SummarizationProvider):
    """Uses a ToolAgents LLM provider to generate summaries.

    Pass any configured provider (OpenAI, Anthropic, etc.) along with a
    model name. The provider will be called with a summarization prompt
    containing the messages to condense.

    Example:
        from ToolAgents.provider import OpenAIChatApi
        provider = OpenAIChatApi(api_key="...")
        summarizer = LLMSummarizationProvider(
            provider=provider,
            model="gpt-4o-mini",
            max_tokens=500,
        )
    """

    def __init__(self, provider, model: str, max_tokens: int = 500):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens

    def summarize(self, messages: List[ChatMessage]) -> str:
        # Build a text representation of messages for the summarization prompt
        text_parts = []
        for msg in messages:
            role = msg.role.value
            content_text = msg.get_as_text()
            if content_text.strip():
                text_parts.append(f"[{role}]: {content_text}")

        conversation_text = "\n".join(text_parts)

        summarization_prompt = (
            "Summarize the following conversation concisely, preserving key facts, "
            "decisions, and any important context that would be needed to continue "
            "the conversation. Focus on what matters for future interactions.\n\n"
            f"{conversation_text}\n\n"
            "Summary:"
        )

        # Use the provider to generate the summary
        summary_messages = [
            ChatMessage.create_system_message(
                "You are a concise summarizer. Produce brief, factual summaries."
            ),
            ChatMessage.create_user_message(summarization_prompt),
        ]

        from ToolAgents.provider.llm_provider import ProviderSettings

        settings = ProviderSettings()
        settings.set_max_tokens(self.max_tokens)

        request = self.provider.message_converter.prepare_request(
            model=self.model,
            messages=summary_messages,
            settings=settings,
        )

        response = self.provider.get_response(request)
        result_message = self.provider.response_converter.from_provider_response(response)

        # Extract text from the response
        for content in result_message.content:
            if isinstance(content, TextContent):
                return content.content.strip()

        return ""


class RuleBasedSummarizationProvider(SummarizationProvider):
    """Simple rule-based summarization without any LLM call.

    Extracts the first N characters of each message's text content and
    joins them with role prefixes. Useful as a lightweight fallback when
    no LLM is available for summarization.

    Example:
        summarizer = RuleBasedSummarizationProvider(max_chars_per_message=200)
        summary = summarizer.summarize(old_messages)
    """

    def __init__(self, max_chars_per_message: int = 200):
        self.max_chars_per_message = max_chars_per_message

    def summarize(self, messages: List[ChatMessage]) -> str:
        parts = []
        for msg in messages:
            role = msg.role.value
            text = msg.get_as_text()
            if text.strip():
                truncated = text[: self.max_chars_per_message]
                if len(text) > self.max_chars_per_message:
                    truncated += "..."
                parts.append(f"[{role}]: {truncated}")

        if not parts:
            return "Previous conversation context (no text content)."

        return "Previous conversation summary:\n" + "\n".join(parts)

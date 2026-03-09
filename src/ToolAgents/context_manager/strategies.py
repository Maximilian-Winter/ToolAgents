# strategies.py — Context trimming strategies.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    ToolCallContent,
    ToolCallResultContent,
    ContentType,
)
from .models import ContextManagerConfig, ContextState


@dataclass
class MessageGroup:
    """A group of message indices that must be kept or removed together.

    Attributes:
        indices: List of message indices in the original message list.
        removable: Whether this group can be removed during trimming.
    """

    indices: List[int]
    removable: bool


def find_atomic_groups(
    messages: List[ChatMessage],
    pinned_ids: Set[str],
    always_keep_system: bool = True,
) -> List[MessageGroup]:
    """Group messages into atomic units for trimming.

    Rules:
    - System messages are non-removable (if always_keep_system).
    - Pinned messages are non-removable.
    - An assistant message containing ToolCallContent is grouped with all
      immediately following tool-result messages (role=Tool or containing
      ToolCallResultContent). They form a single atomic group.
    - All other messages are singleton removable groups.
    """
    groups: List[MessageGroup] = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]

        # Check if non-removable
        is_pinned = msg.id in pinned_ids
        is_system = msg.role == ChatMessageRole.System and always_keep_system

        if is_pinned or is_system:
            groups.append(MessageGroup(indices=[i], removable=False))
            i += 1
            continue

        # Check if this is an assistant message with tool calls
        has_tool_calls = any(
            isinstance(c, ToolCallContent) for c in msg.content
        )

        if msg.role == ChatMessageRole.Assistant and has_tool_calls:
            # Start an atomic group: assistant + following tool result messages
            group_indices = [i]
            j = i + 1
            while j < n:
                next_msg = messages[j]
                # Tool result messages follow the assistant tool-call message
                is_tool_result = (
                    next_msg.role == ChatMessageRole.Tool
                    or any(
                        isinstance(c, ToolCallResultContent)
                        for c in next_msg.content
                    )
                )
                if is_tool_result:
                    group_indices.append(j)
                    j += 1
                else:
                    break

            # Check if any message in the group is pinned
            group_pinned = any(messages[idx].id in pinned_ids for idx in group_indices)
            groups.append(
                MessageGroup(indices=group_indices, removable=not group_pinned)
            )
            i = j
        else:
            groups.append(MessageGroup(indices=[i], removable=True))
            i += 1

    return groups


class ContextStrategy(ABC):
    """Abstract base class for context trimming strategies.

    A strategy receives the full message list and returns two lists:
    the messages to keep and the messages that were trimmed.
    """

    @abstractmethod
    def trim(
        self,
        messages: List[ChatMessage],
        state: ContextState,
        config: ContextManagerConfig,
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """Trim messages to fit within the context window.

        Args:
            messages: The full list of messages.
            state: Current context manager state (includes pinned IDs).
            config: Context manager configuration.

        Returns:
            Tuple of (kept_messages, trimmed_messages).
        """
        ...


class SlidingWindowStrategy(ContextStrategy):
    """Drop the oldest removable message groups first.

    Preserves system messages, pinned messages, and tool call/result atomic pairs
    (removing them as complete units when needed). Works from the oldest messages
    forward, removing groups until the estimated context size drops below the limit.

    Since we don't have per-message token counts, we estimate by removing a
    proportional fraction: each removable group is assumed to contribute equally
    to the total context size.
    """

    def trim(
        self,
        messages: List[ChatMessage],
        state: ContextState,
        config: ContextManagerConfig,
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        if len(messages) == 0:
            return messages, []

        groups = find_atomic_groups(
            messages, state.pinned_message_ids, config.always_keep_system
        )

        # Count removable messages to estimate per-message token contribution
        removable_groups = [g for g in groups if g.removable]
        if not removable_groups:
            return messages, []

        total_removable_messages = sum(len(g.indices) for g in removable_groups)
        if total_removable_messages == 0:
            return messages, []

        effective_limit = config.max_context_tokens - config.reserve_tokens
        excess = state.current_context_tokens - effective_limit
        if excess <= 0:
            return messages, []

        # Estimate tokens per removable message
        tokens_per_message = state.current_context_tokens / max(len(messages), 1)

        # Remove oldest removable groups until we've shed enough estimated tokens
        tokens_removed = 0
        indices_to_remove: Set[int] = set()

        for group in groups:
            if not group.removable:
                continue
            if tokens_removed >= excess:
                break
            for idx in group.indices:
                indices_to_remove.add(idx)
            tokens_removed += len(group.indices) * tokens_per_message

        kept = [msg for i, msg in enumerate(messages) if i not in indices_to_remove]
        trimmed = [msg for i, msg in enumerate(messages) if i in indices_to_remove]

        return kept, trimmed


class KeepLastNTurnsStrategy(ContextStrategy):
    """Keep only the last N user-assistant turn pairs.

    A 'turn' is identified as a user message followed by an assistant response
    (which may include tool call/result pairs before the next user message).
    System messages and pinned messages are always kept.
    """

    def trim(
        self,
        messages: List[ChatMessage],
        state: ContextState,
        config: ContextManagerConfig,
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        if len(messages) == 0:
            return messages, []

        n_turns = config.keep_last_n

        # Find turn boundaries (each turn starts with a user message)
        turn_starts: List[int] = []
        for i, msg in enumerate(messages):
            if msg.role == ChatMessageRole.User:
                # Skip pinned user messages that aren't part of natural turns
                turn_starts.append(i)

        if len(turn_starts) <= n_turns:
            return messages, []

        # The cutoff: keep messages from the (len - n_turns)th user message onward
        cutoff_index = turn_starts[-n_turns]

        kept: List[ChatMessage] = []
        trimmed: List[ChatMessage] = []

        for i, msg in enumerate(messages):
            # Always keep system messages and pinned messages
            is_system = msg.role == ChatMessageRole.System and config.always_keep_system
            is_pinned = msg.id in state.pinned_message_ids

            if is_system or is_pinned or i >= cutoff_index:
                kept.append(msg)
            else:
                trimmed.append(msg)

        return kept, trimmed


class SummarizeAndTrimStrategy(ContextStrategy):
    """Summarize older messages and replace them with a summary, keeping recent ones.

    Splits messages into an "old" portion (to be summarized) and a "recent" portion
    (to be kept as-is). The old portion is replaced with a single assistant message
    containing the summary text.

    Requires a SummarizationProvider to be set on the ContextManager.
    Falls back to SlidingWindowStrategy if no summarizer is available.
    """

    def __init__(self, summarizer=None):
        self._summarizer = summarizer

    @property
    def summarizer(self):
        return self._summarizer

    @summarizer.setter
    def summarizer(self, value):
        self._summarizer = value

    def trim(
        self,
        messages: List[ChatMessage],
        state: ContextState,
        config: ContextManagerConfig,
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        if self._summarizer is None:
            # Fall back to sliding window if no summarizer configured
            fallback = SlidingWindowStrategy()
            return fallback.trim(messages, state, config)

        if len(messages) == 0:
            return messages, []

        groups = find_atomic_groups(
            messages, state.pinned_message_ids, config.always_keep_system
        )

        # Find the split point: keep the last keep_last_n turns worth of groups
        removable_groups = [g for g in groups if g.removable]
        if not removable_groups:
            return messages, []

        # We'll summarize the first half of removable groups
        n_to_summarize = max(1, len(removable_groups) // 2)
        groups_to_summarize = removable_groups[:n_to_summarize]

        indices_to_summarize: Set[int] = set()
        for group in groups_to_summarize:
            for idx in group.indices:
                indices_to_summarize.add(idx)

        if not indices_to_summarize:
            return messages, []

        # Extract messages to summarize
        messages_for_summary = [
            messages[i] for i in sorted(indices_to_summarize)
        ]

        # Generate summary
        summary_text = self._summarizer.summarize(messages_for_summary)

        # Build the result: non-summarized messages + summary inserted after system messages
        summary_message = ChatMessage.create_assistant_message(
            f"[Context Summary] {summary_text}"
        )

        kept: List[ChatMessage] = []
        trimmed: List[ChatMessage] = []
        summary_inserted = False

        for i, msg in enumerate(messages):
            if i in indices_to_summarize:
                trimmed.append(msg)
                # Insert summary after the last summarized message
                if not summary_inserted and (
                    i == max(indices_to_summarize)
                ):
                    kept.append(summary_message)
                    summary_inserted = True
            else:
                kept.append(msg)

        # If we haven't inserted the summary yet (edge case), prepend after system
        if not summary_inserted and summary_text:
            # Find first non-system position
            insert_pos = 0
            for j, msg in enumerate(kept):
                if msg.role != ChatMessageRole.System:
                    insert_pos = j
                    break
            kept.insert(insert_pos, summary_message)

        state.summaries_generated += 1

        return kept, trimmed

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from ToolAgents.data_models.messages import ChatMessage

logger = logging.getLogger(__name__)


class ExpiryAction(str, Enum):
    """What to do when a smart message's TTL reaches zero.

    Values:
        REMOVE: Silently drop the message from the conversation.
        SUMMARIZE: Replace the message with a compact summary.
        ARCHIVE: Move the message content to an archival store.
        CUSTOM: Call the message's on_expire_callback.
    """

    REMOVE = "remove"
    SUMMARIZE = "summarize"
    ARCHIVE = "archive"
    CUSTOM = "custom"


@dataclass
class MessageLifecycle:
    """Lifecycle metadata attached to a smart message.

    Attributes:
        ttl: Turns to live. Decremented each turn by the manager.
            None means the message is permanent (never expires from lifecycle).
            0 means the message expires on the next tick.
        turns_alive: How many turns this message has been active. Incremented
            each turn by the manager. Useful for age-based logic in callbacks.
        pinned: If True, the message is exempt from both lifecycle expiry
            AND context trimming. It stays in the conversation until explicitly
            unpinned and expired, or removed by the user.
        on_expire: What to do when TTL reaches 0. Defaults to REMOVE.
        on_expire_callback: Called when the message expires, if on_expire is
            CUSTOM (or in addition to other actions if desired). Receives the
            ChatMessage and this lifecycle as arguments.
        on_tick_callback: Called every turn while the message is alive.
            Receives the ChatMessage and this lifecycle as arguments.
            Use for decay effects, logging, or conditional early expiry.
        summarize_fn: Called when on_expire is SUMMARIZE. Receives the
            ChatMessage and should return a replacement ChatMessage (or None
            to just remove). If not set, a default text truncation is used.
        archive_fn: Called when on_expire is ARCHIVE. Receives the
            ChatMessage. Should handle storing it externally. If not set,
            the expired message data is placed in the manager's archive list.
        metadata: Arbitrary user-defined metadata for this lifecycle.
            Useful for tagging messages with source, purpose, etc.
    """

    ttl: Optional[int] = None
    turns_alive: int = 0
    pinned: bool = False
    on_expire: ExpiryAction = ExpiryAction.REMOVE
    on_expire_callback: Optional[Callable[["ChatMessage", "MessageLifecycle"], None]] = None
    on_tick_callback: Optional[Callable[["ChatMessage", "MessageLifecycle"], None]] = None
    summarize_fn: Optional[Callable[["ChatMessage"], Optional["ChatMessage"]]] = None
    archive_fn: Optional[Callable[["ChatMessage"], None]] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_permanent(self) -> bool:
        """True if this message has no TTL (lives forever unless removed)."""
        return self.ttl is None

    @property
    def is_expired(self) -> bool:
        """True if the TTL has reached 0 or below (and the message is not permanent)."""
        if self.ttl is None:
            return False
        if self.pinned:
            return False
        return self.ttl <= 0

    def copy(self) -> "MessageLifecycle":
        """Create a shallow copy of this lifecycle."""
        return MessageLifecycle(
            ttl=self.ttl,
            turns_alive=self.turns_alive,
            pinned=self.pinned,
            on_expire=self.on_expire,
            on_expire_callback=self.on_expire_callback,
            on_tick_callback=self.on_tick_callback,
            summarize_fn=self.summarize_fn,
            archive_fn=self.archive_fn,
            metadata=dict(self.metadata),
        )


@dataclass
class SmartMessage:
    """A ChatMessage paired with lifecycle metadata.

    This is the internal wrapper used by SmartMessageManager. Users interact
    with it through the manager's API, not directly.

    Attributes:
        message: The underlying ChatMessage.
        lifecycle: The lifecycle configuration and state.
    """

    message: "ChatMessage"
    lifecycle: MessageLifecycle


class ExpiryResult:
    """Collects the results of a tick() call for the harness to process.

    Attributes:
        removed: Messages that were silently removed.
        summarized: Tuples of (original_message, replacement_message).
            replacement_message may be None if summarization produced nothing.
        archived: Messages that should be stored in archival memory.
        custom: Messages that triggered custom callbacks.
    """

    def __init__(self):
        self.removed: List["ChatMessage"] = []
        self.summarized: List[tuple["ChatMessage", Optional["ChatMessage"]]] = []
        self.archived: List["ChatMessage"] = []
        self.custom: List["ChatMessage"] = []

    @property
    def has_changes(self) -> bool:
        """True if any messages expired this tick."""
        return bool(self.removed or self.summarized or self.archived or self.custom)

    def __repr__(self) -> str:
        return (
            f"ExpiryResult(removed={len(self.removed)}, "
            f"summarized={len(self.summarized)}, "
            f"archived={len(self.archived)}, "
            f"custom={len(self.custom)})"
        )


class SmartMessageManager:
    """Manages conversation messages with lifecycle awareness.

    Each message can optionally carry a MessageLifecycle that controls how
    long it stays in the conversation and what happens when it expires.
    Messages without a lifecycle behave as normal permanent messages.

    The manager operates independently of context trimming. The harness calls
    tick() before each LLM call, and uses get_active_messages() to build the
    message list. Trimming then operates on the already-filtered list.

    Usage:
        manager = SmartMessageManager()

        # Add a normal permanent message
        manager.add_message(user_msg)

        # Add an ephemeral message
        manager.add_message(
            system_context_msg,
            lifecycle=MessageLifecycle(ttl=3, on_expire=ExpiryAction.REMOVE)
        )

        # Each turn:
        expiry_result = manager.tick()
        if expiry_result.has_changes:
            # Handle archived messages, etc.
            for msg in expiry_result.archived:
                archival_memory.store(msg.get_as_text())

        active = manager.get_active_messages()
        # ... send active messages to LLM
    """

    def __init__(self) -> None:
        self._messages: List[SmartMessage] = []
        self._archive: List["ChatMessage"] = []
        self._tick_count: int = 0

    # --- Message Management ---

    def add_message(
        self,
        message: "ChatMessage",
        lifecycle: Optional[MessageLifecycle] = None,
    ) -> SmartMessage:
        """Add a message to the managed conversation.

        Args:
            message: The ChatMessage to add.
            lifecycle: Optional lifecycle configuration. If None, the message
                is permanent (no TTL, no expiry behavior).

        Returns:
            The SmartMessage wrapper.
        """
        if lifecycle is None:
            lifecycle = MessageLifecycle()  # permanent by default

        smart_msg = SmartMessage(message=message, lifecycle=lifecycle)
        self._messages.append(smart_msg)

        logger.debug(
            "Added smart message id=%s, ttl=%s, on_expire=%s",
            message.id, lifecycle.ttl, lifecycle.on_expire.value,
        )
        return smart_msg

    def add_messages(
        self,
        messages: List["ChatMessage"],
        lifecycle: Optional[MessageLifecycle] = None,
    ) -> List[SmartMessage]:
        """Add multiple messages with the same lifecycle configuration.

        If a lifecycle is provided, each message gets its own copy so TTL
        tracking is independent.

        Args:
            messages: The ChatMessages to add.
            lifecycle: Optional shared lifecycle template (each message gets a copy).

        Returns:
            List of SmartMessage wrappers.
        """
        result = []
        for msg in messages:
            lc = lifecycle.copy() if lifecycle is not None else None
            result.append(self.add_message(msg, lc))
        return result

    def remove_message(self, message_id: str) -> Optional["ChatMessage"]:
        """Remove a message by its ChatMessage id.

        Args:
            message_id: The id of the ChatMessage to remove.

        Returns:
            The removed ChatMessage, or None if not found.
        """
        for i, sm in enumerate(self._messages):
            if sm.message.id == message_id:
                removed = self._messages.pop(i)
                return removed.message
        return None

    def pin_message(self, message_id: str) -> bool:
        """Pin a message so it never expires and is exempt from trimming.

        Args:
            message_id: The id of the ChatMessage to pin.

        Returns:
            True if the message was found and pinned.
        """
        for sm in self._messages:
            if sm.message.id == message_id:
                sm.lifecycle.pinned = True
                return True
        return False

    def unpin_message(self, message_id: str) -> bool:
        """Unpin a message, allowing normal lifecycle processing.

        Args:
            message_id: The id of the ChatMessage to unpin.

        Returns:
            True if the message was found and unpinned.
        """
        for sm in self._messages:
            if sm.message.id == message_id:
                sm.lifecycle.pinned = False
                return True
        return False

    def set_ttl(self, message_id: str, ttl: Optional[int]) -> bool:
        """Update the TTL of an existing message.

        Args:
            message_id: The id of the ChatMessage.
            ttl: New TTL value (None for permanent).

        Returns:
            True if the message was found and updated.
        """
        for sm in self._messages:
            if sm.message.id == message_id:
                sm.lifecycle.ttl = ttl
                return True
        return False

    def get_lifecycle(self, message_id: str) -> Optional[MessageLifecycle]:
        """Get the lifecycle for a message.

        Args:
            message_id: The id of the ChatMessage.

        Returns:
            The MessageLifecycle, or None if the message is not found.
        """
        for sm in self._messages:
            if sm.message.id == message_id:
                return sm.lifecycle
        return None

    # --- Turn Processing ---

    def tick(self) -> ExpiryResult:
        """Process one turn of lifecycle for all messages.

        For each message:
        1. Increment turns_alive.
        2. Call on_tick_callback if set.
        3. If TTL is set and not pinned, decrement TTL.
        4. If TTL reaches 0, process expiry according to on_expire action.

        Messages that expire are removed from the active list.
        Summarized messages are replaced in-place.

        Returns:
            ExpiryResult describing what changed this tick.
        """
        self._tick_count += 1
        result = ExpiryResult()
        surviving: List[SmartMessage] = []

        for sm in self._messages:
            # Increment age
            sm.lifecycle.turns_alive += 1

            # Call per-turn callback
            if sm.lifecycle.on_tick_callback is not None:
                try:
                    sm.lifecycle.on_tick_callback(sm.message, sm.lifecycle)
                except Exception as e:
                    logger.error(
                        "Error in on_tick_callback for message %s: %s",
                        sm.message.id, e,
                    )

            # Pinned messages are never expired by lifecycle
            if sm.lifecycle.pinned:
                surviving.append(sm)
                continue

            # Permanent messages (no TTL) always survive
            if sm.lifecycle.ttl is None:
                surviving.append(sm)
                continue

            # Decrement TTL
            sm.lifecycle.ttl -= 1

            if sm.lifecycle.ttl > 0:
                # Still alive
                surviving.append(sm)
                continue

            # TTL reached 0 — process expiry
            logger.debug(
                "Message %s expired (on_expire=%s, turns_alive=%d)",
                sm.message.id, sm.lifecycle.on_expire.value, sm.lifecycle.turns_alive,
            )

            if sm.lifecycle.on_expire == ExpiryAction.REMOVE:
                result.removed.append(sm.message)

            elif sm.lifecycle.on_expire == ExpiryAction.SUMMARIZE:
                replacement = self._handle_summarize(sm)
                result.summarized.append((sm.message, replacement))
                if replacement is not None:
                    # Replace in-place with a new permanent smart message
                    surviving.append(SmartMessage(
                        message=replacement,
                        lifecycle=MessageLifecycle(),  # permanent
                    ))

            elif sm.lifecycle.on_expire == ExpiryAction.ARCHIVE:
                self._handle_archive(sm)
                result.archived.append(sm.message)

            elif sm.lifecycle.on_expire == ExpiryAction.CUSTOM:
                self._handle_custom(sm)
                result.custom.append(sm.message)

        self._messages = surviving

        if result.has_changes:
            logger.info("Tick %d: %s", self._tick_count, result)

        return result

    # --- Query ---

    def get_active_messages(self) -> List["ChatMessage"]:
        """Get all currently active (non-expired) messages.

        Returns:
            List of ChatMessage instances in insertion order.
        """
        return [sm.message for sm in self._messages]

    def get_smart_messages(self) -> List[SmartMessage]:
        """Get all active SmartMessage wrappers (message + lifecycle).

        Returns:
            List of SmartMessage instances in insertion order.
        """
        return list(self._messages)

    def get_pinned_message_ids(self) -> set:
        """Get the set of message IDs that are currently pinned.

        Useful for passing to the context manager's pinned set.

        Returns:
            Set of pinned message ID strings.
        """
        return {sm.message.id for sm in self._messages if sm.lifecycle.pinned}

    @property
    def archive(self) -> List["ChatMessage"]:
        """Messages that were archived via the ARCHIVE expiry action."""
        return list(self._archive)

    @property
    def tick_count(self) -> int:
        """Number of ticks processed so far."""
        return self._tick_count

    @property
    def message_count(self) -> int:
        """Number of currently active messages."""
        return len(self._messages)

    # --- Internal ---

    def _handle_summarize(self, sm: SmartMessage) -> Optional["ChatMessage"]:
        """Process SUMMARIZE expiry action.

        If the message has a custom summarize_fn, use it. Otherwise,
        create a default truncated summary.

        Returns:
            A replacement ChatMessage, or None if summarization produced nothing.
        """
        if sm.lifecycle.summarize_fn is not None:
            try:
                return sm.lifecycle.summarize_fn(sm.message)
            except Exception as e:
                logger.error(
                    "Error in summarize_fn for message %s: %s",
                    sm.message.id, e,
                )
                return None

        # Default: truncate to first 200 characters
        text = sm.message.get_as_text()
        if not text:
            return None

        truncated = text[:200]
        if len(text) > 200:
            truncated += "..."

        # Import here to avoid circular dependency at module level
        from ToolAgents.data_models.messages import ChatMessage

        return ChatMessage.create_assistant_message(
            f"[Summary of expired context] {truncated}"
        )

    def _handle_archive(self, sm: SmartMessage) -> None:
        """Process ARCHIVE expiry action.

        If the message has a custom archive_fn, call it. Otherwise,
        store in the manager's internal archive list.
        """
        if sm.lifecycle.archive_fn is not None:
            try:
                sm.lifecycle.archive_fn(sm.message)
                return
            except Exception as e:
                logger.error(
                    "Error in archive_fn for message %s: %s",
                    sm.message.id, e,
                )
        # Default: store internally
        self._archive.append(sm.message)

    def _handle_custom(self, sm: SmartMessage) -> None:
        """Process CUSTOM expiry action."""
        if sm.lifecycle.on_expire_callback is not None:
            try:
                sm.lifecycle.on_expire_callback(sm.message, sm.lifecycle)
            except Exception as e:
                logger.error(
                    "Error in on_expire_callback for message %s: %s",
                    sm.message.id, e,
                )

    def clear(self) -> None:
        """Remove all messages and reset tick count."""
        self._messages.clear()
        self._archive.clear()
        self._tick_count = 0

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        permanent = sum(1 for sm in self._messages if sm.lifecycle.is_permanent)
        ephemeral = len(self._messages) - permanent
        pinned = sum(1 for sm in self._messages if sm.lifecycle.pinned)
        return (
            f"SmartMessageManager({len(self._messages)} messages: "
            f"{permanent} permanent, {ephemeral} ephemeral, {pinned} pinned)"
        )

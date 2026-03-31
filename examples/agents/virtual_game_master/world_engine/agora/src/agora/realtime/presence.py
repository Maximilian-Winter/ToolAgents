"""Tracks agent liveness (last activity) and typing indicators."""

import asyncio


class PresenceTracker:
    """Tracks agent liveness (last activity) and typing indicators.

    Typing auto-expires after TYPING_TTL seconds. Presence is updated on
    any API activity (post, poll, wait, mark_read, typing).
    """

    TYPING_TTL = 10.0  # seconds before typing indicator expires
    IDLE_THRESHOLD = 120.0  # seconds before agent is considered idle

    def __init__(self):
        # agent_name -> last activity timestamp
        self._last_seen: dict[str, float] = {}
        # (room_name, agent_name) -> typing start timestamp
        self._typing: dict[tuple[str, str], float] = {}

    def touch(self, agent_name: str) -> None:
        """Update last-seen timestamp for an agent."""
        self._last_seen[agent_name] = asyncio.get_event_loop().time()

    def set_typing(self, room_name: str, agent_name: str) -> None:
        """Mark agent as typing in a room."""
        self._typing[(room_name, agent_name)] = asyncio.get_event_loop().time()
        self.touch(agent_name)

    def clear_typing(self, room_name: str, agent_name: str) -> None:
        """Clear typing indicator (e.g. after posting a message)."""
        self._typing.pop((room_name, agent_name), None)

    def get_typing(self, room_name: str) -> list[str]:
        """Return list of agents currently typing in a room."""
        now = asyncio.get_event_loop().time()
        typing = []
        expired = []
        for (rm, agent), ts in self._typing.items():
            if rm != room_name:
                continue
            if now - ts > self.TYPING_TTL:
                expired.append((rm, agent))
            else:
                typing.append(agent)
        for key in expired:
            del self._typing[key]
        return typing

    def get_status(self, agent_name: str) -> str:
        """Return 'online', 'idle', or 'offline'."""
        ts = self._last_seen.get(agent_name)
        if ts is None:
            return "offline"
        elapsed = asyncio.get_event_loop().time() - ts
        if elapsed < self.IDLE_THRESHOLD:
            return "online"
        return "idle"

    def get_all_statuses(self) -> dict[str, str]:
        """Return {agent_name: status} for all known agents."""
        return {name: self.get_status(name) for name in self._last_seen}


presence = PresenceTracker()

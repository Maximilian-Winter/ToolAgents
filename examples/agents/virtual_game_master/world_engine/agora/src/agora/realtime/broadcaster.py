"""In-memory pub/sub for pushing events to SSE and long-poll listeners."""

import asyncio


class RoomBroadcaster:
    """In-memory pub/sub for pushing events to SSE and long-poll listeners.

    Each subscriber gets an asyncio.Queue. When an event is published to a
    room, it is pushed to every subscriber queue for that room.
    """

    def __init__(self):
        # room_name -> list[asyncio.Queue]
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, room_name: str) -> asyncio.Queue:
        """Create a new subscriber queue for a room."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(room_name, []).append(queue)
        return queue

    def unsubscribe(self, room_name: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        subs = self._subscribers.get(room_name, [])
        try:
            subs.remove(queue)
        except ValueError:
            pass
        if not subs:
            self._subscribers.pop(room_name, None)

    def publish(self, room_name: str, event_type: str, data: dict) -> None:
        """Push an event to all subscribers of a room.

        event_type: 'message', 'reaction', 'receipt', etc.
        data: JSON-serializable dict.
        """
        payload = {"event": event_type, "data": data}
        for queue in self._subscribers.get(room_name, []):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # Slow consumer -- drop the event

    @property
    def subscriber_count(self) -> dict[str, int]:
        """Return {room_name: count} for diagnostics."""
        return {k: len(v) for k, v in self._subscribers.items()}


broadcaster = RoomBroadcaster()

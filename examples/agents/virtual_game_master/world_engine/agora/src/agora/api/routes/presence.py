"""Presence endpoint for checking agent online/idle/offline status."""

from typing import Optional

from fastapi import APIRouter, Query

from agora.realtime import presence

router = APIRouter(prefix="/api/presence", tags=["Presence"])


@router.get("")
async def get_presence(
    agents: Optional[str] = Query(
        None,
        description="Comma-separated agent names to check (default: all)",
    ),
):
    """Get online/idle/offline status for agents.

    Status is derived from last API activity:
    - **online**: active within the last 2 minutes
    - **idle**: last activity 2-10 minutes ago
    - **offline**: no activity or >10 minutes ago
    """
    if agents:
        names = [n.strip() for n in agents.split(",") if n.strip()]
        return {name: presence.get_status(name) for name in names}
    return presence.get_all_statuses()

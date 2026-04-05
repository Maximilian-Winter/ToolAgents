"""
GM Tool — Python Client

Async client for the GM Tool REST API.

Usage:
    from client import GMClient

    async with GMClient("http://localhost:8000") as gm:
        campaigns = await gm.campaigns.list()
        campaign = await gm.campaigns.create(name="My Campaign", game_system="D&D 5e")

        loc = await gm.locations.create(campaign.id, name="Waterdeep", location_type="city")
        resolved = await gm.locations.resolve_path(campaign.id, "waterdeep")

        npc = await gm.npcs.create(campaign.id, name="Vajra", title="The Blackstaff")
        await gm.tags.apply(campaign.id, tag_id=1, target_type="npc", target_id=npc.id)
"""

from client.core import GMClient

__all__ = ["GMClient"]

"""
Demo: populate a small world and exercise the path system.

Run with:  python -m asyncio demo.py
       or: python demo.py
"""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models import (
    Campaign,
    ConnectionType,
    Location,
    LocationConnection,
    LocationType,
    NPC,
    NPCLocationAssociation,
    NPCRelationship,
    NPCRelationshipType,
    NPCStatus,
    init_db,
)

DATABASE_URL = "sqlite+aiosqlite:///demo.db"


async def main():
    engine = await init_db(DATABASE_URL)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        async with session.begin():

            # --- Campaign ---
            campaign = Campaign(
                name="Dragon Heist",
                game_system="D&D 5e",
                description="A treasure hunt through the streets of Waterdeep.",
            )
            session.add(campaign)
            await session.flush()  # get campaign.id

            # --- Locations (hierarchy) ---
            sword_coast = Location(
                campaign_id=campaign.id,
                name="Sword Coast",
                location_type=LocationType.REGION,
                description="A stretch of coastline along the western edge of Faerûn.",
            )
            session.add(sword_coast)
            await session.flush()

            waterdeep = Location(
                campaign_id=campaign.id,
                parent_id=sword_coast.id,
                name="Waterdeep",
                location_type=LocationType.CITY,
                description="The City of Splendors, the greatest city in the North.",
            )
            session.add(waterdeep)
            await session.flush()

            castle_ward = Location(
                campaign_id=campaign.id,
                parent_id=waterdeep.id,
                name="Castle Ward",
                location_type=LocationType.DISTRICT,
                description="Home to the city's nobles and the seat of government.",
            )
            session.add(castle_ward)
            await session.flush()

            blackstaff_tower = Location(
                campaign_id=campaign.id,
                parent_id=castle_ward.id,
                name="Blackstaff Tower",
                location_type=LocationType.BUILDING,
                description="The imposing tower of the Blackstaff, archmage of Waterdeep.",
                secrets="Contains a hidden portal to Undermountain.",
            )
            session.add(blackstaff_tower)

            dock_ward = Location(
                campaign_id=campaign.id,
                parent_id=waterdeep.id,
                name="Dock Ward",
                location_type=LocationType.DISTRICT,
                description="The roughest, most dangerous district in Waterdeep.",
            )
            session.add(dock_ward)
            await session.flush()

            skewered_dragon = Location(
                campaign_id=campaign.id,
                parent_id=dock_ward.id,
                name="The Skewered Dragon",
                location_type=LocationType.BUILDING,
                description="A seedy tavern frequented by sailors and smugglers.",
            )
            session.add(skewered_dragon)
            await session.flush()

            # --- Lateral connection ---
            secret_tunnel = LocationConnection(
                from_location_id=blackstaff_tower.id,
                to_location_id=skewered_dragon.id,
                connection_type=ConnectionType.SECRET,
                description="An ancient tunnel beneath the city, known only to the Blackstaff.",
                is_secret=True,
                is_bidirectional=True,
            )
            session.add(secret_tunnel)

            # --- NPCs ---
            vajra = NPC(
                campaign_id=campaign.id,
                primary_location_id=blackstaff_tower.id,
                name="Vajra Safahr",
                title="The Blackstaff",
                description="The young and talented archmage of Waterdeep.",
                personality="Intense, focused, carries the weight of responsibility.",
                motivations="Protect Waterdeep from threats both magical and mundane.",
                secrets="Hears the whispers of all previous Blackstaffs in her mind.",
                status=NPCStatus.ALIVE,
            )
            session.add(vajra)

            durnan = NPC(
                campaign_id=campaign.id,
                primary_location_id=skewered_dragon.id,
                name="Durnan",
                title="Proprietor of the Yawning Portal",
                description="A retired adventurer who runs Waterdeep's most famous tavern.",
                personality="Gruff, no-nonsense, but fair.",
                motivations="Keep the peace in his establishment. Guard the entrance to Undermountain.",
                status=NPCStatus.ALIVE,
            )
            session.add(durnan)

            renaer = NPC(
                campaign_id=campaign.id,
                name="Renaer Neverember",
                title="Noble",
                description="Son of the former Open Lord, Dagult Neverember.",
                personality="Charming, idealistic, haunted by his father's legacy.",
                motivations="Prove himself independent of his father. Help the common people.",
                secrets="Knows more about the hidden gold than he lets on.",
                status=NPCStatus.ALIVE,
            )
            session.add(renaer)
            await session.flush()

            # --- NPC ↔ Location associations ---
            session.add_all(
                [
                    NPCLocationAssociation(
                        npc_id=vajra.id,
                        location_id=blackstaff_tower.id,
                        role="owner",
                        notes="Lives and works here.",
                    ),
                    NPCLocationAssociation(
                        npc_id=durnan.id,
                        location_id=skewered_dragon.id,
                        role="patron",
                        notes="Frequents this tavern for information.",
                    ),
                    NPCLocationAssociation(
                        npc_id=renaer.id,
                        location_id=castle_ward.id,
                        role="resident",
                        notes="Has a manor in Castle Ward.",
                    ),
                    NPCLocationAssociation(
                        npc_id=renaer.id,
                        location_id=skewered_dragon.id,
                        role="visitor",
                        notes="Slums it here occasionally to gather rumors.",
                    ),
                ]
            )

            # --- NPC ↔ NPC relationships ---
            session.add_all(
                [
                    NPCRelationship(
                        from_npc_id=vajra.id,
                        to_npc_id=durnan.id,
                        relationship_type=NPCRelationshipType.ALLY,
                        description="Mutual respect; Durnan guards the Undermountain entrance that Vajra monitors magically.",
                    ),
                    NPCRelationship(
                        from_npc_id=renaer.id,
                        to_npc_id=vajra.id,
                        relationship_type=NPCRelationshipType.CONTACT,
                        description="Renaer occasionally passes information to the Blackstaff.",
                        is_secret=True,
                    ),
                ]
            )

        # --- commit done, now query ---

        print("=" * 60)
        print("PATH RESOLUTION DEMO")
        print("=" * 60)

        test_paths = [
            "sword-coast",
            "sword-coast/waterdeep",
            "sword-coast/waterdeep/castle-ward",
            "sword-coast/waterdeep/castle-ward/blackstaff-tower",
            "sword-coast/waterdeep/dock-ward/the-skewered-dragon",
            "sword-coast/waterdeep/nonexistent",
        ]

        for path in test_paths:
            loc = await Location.resolve_path(session, campaign.id, path)
            if loc:
                full_path = await Location.get_location_path(session, loc.id)
                print(f"  ✓ {path}")
                print(f"    → {loc.name} (type: {loc.location_type.value}, path: {full_path})")
            else:
                print(f"  ✗ {path} — not found")
            print()

        print("=" * 60)
        print("LOCATION SEARCH DEMO")
        print("=" * 60)

        results = await Location.search(session, campaign.id, "ward")
        for loc in results:
            full_path = await Location.get_location_path(session, loc.id)
            print(f"  • {loc.name} — {full_path}")
        print()

        print("=" * 60)
        print("NPC SEARCH DEMO")
        print("=" * 60)

        npcs = await NPC.search(session, campaign.id, "ren")
        for npc in npcs:
            print(f"  • {npc.name} ({npc.title}) — {npc.status.value}")
        print()

    await engine.dispose()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

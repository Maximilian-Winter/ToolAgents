"""
GM Tool — FastAPI Application

Run with:
    uvicorn app:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import init_db
from routers import campaigns, locations, notes, npcs, player_characters, sessions, tags, world_lore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup."""
    await init_db()
    yield


app = FastAPI(
    title="GM Tool",
    description="A Game Master's tool for organizing worldbuilding, locations, and NPCs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(campaigns.router)
app.include_router(locations.router)
app.include_router(npcs.router)
app.include_router(player_characters.router)
app.include_router(sessions.router)
app.include_router(notes.router)
app.include_router(tags.router)
app.include_router(world_lore.router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "GM Tool",
        "version": "0.1.0",
        "docs": "/docs",
    }

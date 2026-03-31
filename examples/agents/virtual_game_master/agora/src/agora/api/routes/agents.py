from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.engine import get_db
from agora.db.models.agent import Agent, AgentPersona
from agora.api.deps import hash_token
from agora.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentOut,
    PersonaCreate,
    PersonaUpdate,
    PersonaOut,
)

# ── Agent routes ──────────────────────────────────────────────────────────────

agents_router = APIRouter(prefix="/api/agents", tags=["Agents"])


@agents_router.post("", response_model=AgentOut, status_code=201)
async def register_agent(
    body: AgentCreate,
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Agent).where(Agent.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Agent '{body.name}' already exists")

    agent = Agent(
        name=body.name,
        display_name=body.display_name,
        role=body.role,
        token_hash=hash_token(body.token) if body.token else None,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return agent


@agents_router.get("", response_model=list[AgentOut])
async def list_agents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).order_by(Agent.created_at.desc()))
    return result.scalars().all()


@agents_router.get("/{name}", response_model=AgentOut)
async def get_agent(name: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.name == name))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent '{name}' not found")
    return agent


@agents_router.patch("/{name}", response_model=AgentOut)
async def update_agent(
    name: str,
    body: AgentUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Agent).where(Agent.name == name))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent '{name}' not found")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(agent, key, value)

    await db.commit()
    await db.refresh(agent)
    return agent


@agents_router.delete("/{name}", status_code=204)
async def delete_agent(name: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.name == name))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent '{name}' not found")
    await db.delete(agent)
    await db.commit()


# ── Persona routes ────────────────────────────────────────────────────────────

personas_router = APIRouter(prefix="/api/personas", tags=["Personas"])


@personas_router.post("", response_model=PersonaOut, status_code=201)
async def create_persona(
    body: PersonaCreate,
    db: AsyncSession = Depends(get_db),
):
    persona = AgentPersona(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
    )
    db.add(persona)
    await db.commit()
    await db.refresh(persona)
    return persona


@personas_router.get("", response_model=list[PersonaOut])
async def list_personas(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AgentPersona).order_by(AgentPersona.created_at.desc())
    )
    return result.scalars().all()


@personas_router.get("/{persona_id}", response_model=PersonaOut)
async def get_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AgentPersona).where(AgentPersona.id == persona_id)
    )
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(404, f"Persona {persona_id} not found")
    return persona


@personas_router.patch("/{persona_id}", response_model=PersonaOut)
async def update_persona(
    persona_id: int,
    body: PersonaUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentPersona).where(AgentPersona.id == persona_id)
    )
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(404, f"Persona {persona_id} not found")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(persona, key, value)

    await db.commit()
    await db.refresh(persona)
    return persona


@personas_router.delete("/{persona_id}", status_code=204)
async def delete_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AgentPersona).where(AgentPersona.id == persona_id)
    )
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(404, f"Persona {persona_id} not found")
    await db.delete(persona)
    await db.commit()

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agora.db.engine import get_db
from agora.db.models.agent import Agent
from agora.db.models.project_agent import ProjectAgent
from agora.api.deps import require_project
from agora.schemas.project_agent import (
    ProjectAgentAdd,
    ProjectAgentUpdate,
    ProjectAgentOut,
)

router = APIRouter(
    prefix="/api/projects/{project_slug}/agents",
    tags=["Project Agents"],
)


def _to_out(pa: ProjectAgent) -> ProjectAgentOut:
    """Convert a ProjectAgent (with loaded agent relationship) to output schema."""
    return ProjectAgentOut(
        id=pa.id,
        project_id=pa.project_id,
        agent_id=pa.agent_id,
        system_prompt=pa.system_prompt,
        initial_task=pa.initial_task,
        model=pa.model,
        allowed_tools=pa.allowed_tools,
        prompt_source=pa.prompt_source,
        runtime=pa.runtime, extra_flags=pa.extra_flags,
        added_at=pa.added_at,
        agent_name=pa.agent.name,
        agent_display_name=pa.agent.display_name,
        agent_role=pa.agent.role,
    )


@router.post("", response_model=ProjectAgentOut, status_code=201)
async def add_agent_to_project(
    project_slug: str,
    body: ProjectAgentAdd,
    db: AsyncSession = Depends(get_db),
):
    """Add a global agent preset to this project with optional config overrides."""
    project = await require_project(project_slug, db)

    # Find the agent by name
    result = await db.execute(select(Agent).where(Agent.name == body.agent_name))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent '{body.agent_name}' not found")

    # Check if already added
    existing = await db.execute(
        select(ProjectAgent).where(
            ProjectAgent.project_id == project.id,
            ProjectAgent.agent_id == agent.id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Agent '{body.agent_name}' is already in this project")

    pa = ProjectAgent(
        project_id=project.id,
        agent_id=agent.id,
        system_prompt=body.system_prompt,
        initial_task=body.initial_task,
        model=body.model,
        allowed_tools=body.allowed_tools,
        prompt_source=body.prompt_source,
        runtime=body.runtime, extra_flags=body.extra_flags,
    )
    db.add(pa)
    await db.commit()
    await db.refresh(pa)

    # Load the agent relationship for the response
    result = await db.execute(
        select(ProjectAgent)
        .where(ProjectAgent.id == pa.id)
        .options(selectinload(ProjectAgent.agent))
    )
    pa = result.scalar_one()
    return _to_out(pa)


@router.get("", response_model=list[ProjectAgentOut])
async def list_project_agents(
    project_slug: str,
    db: AsyncSession = Depends(get_db),
):
    """List all agents assigned to this project with their configs."""
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(ProjectAgent)
        .where(ProjectAgent.project_id == project.id)
        .options(selectinload(ProjectAgent.agent))
        .order_by(ProjectAgent.added_at)
    )
    return [_to_out(pa) for pa in result.scalars().all()]


@router.get("/{agent_name}", response_model=ProjectAgentOut)
async def get_project_agent(
    project_slug: str,
    agent_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific agent's project config."""
    project = await require_project(project_slug, db)

    result = await db.execute(
        select(ProjectAgent)
        .join(Agent)
        .where(ProjectAgent.project_id == project.id, Agent.name == agent_name)
        .options(selectinload(ProjectAgent.agent))
    )
    pa = result.scalar_one_or_none()
    if not pa:
        raise HTTPException(404, f"Agent '{agent_name}' is not in project '{project_slug}'")
    return _to_out(pa)


@router.patch("/{agent_name}", response_model=ProjectAgentOut)
async def update_project_agent(
    project_slug: str,
    agent_name: str,
    body: ProjectAgentUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a project agent's configuration."""
    project = await require_project(project_slug, db)

    result = await db.execute(
        select(ProjectAgent)
        .join(Agent)
        .where(ProjectAgent.project_id == project.id, Agent.name == agent_name)
        .options(selectinload(ProjectAgent.agent))
    )
    pa = result.scalar_one_or_none()
    if not pa:
        raise HTTPException(404, f"Agent '{agent_name}' is not in project '{project_slug}'")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(pa, key, value)

    await db.commit()
    await db.refresh(pa)

    # Reload with relationship
    result = await db.execute(
        select(ProjectAgent)
        .where(ProjectAgent.id == pa.id)
        .options(selectinload(ProjectAgent.agent))
    )
    pa = result.scalar_one()
    return _to_out(pa)


@router.delete("/{agent_name}", status_code=204)
async def remove_agent_from_project(
    project_slug: str,
    agent_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Remove an agent from this project."""
    project = await require_project(project_slug, db)

    result = await db.execute(
        select(ProjectAgent)
        .join(Agent)
        .where(ProjectAgent.project_id == project.id, Agent.name == agent_name)
    )
    pa = result.scalar_one_or_none()
    if not pa:
        raise HTTPException(404, f"Agent '{agent_name}' is not in project '{project_slug}'")

    await db.delete(pa)
    await db.commit()

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.engine import get_db
from agora.db.models.agent import Agent
from agora.db.models.team import Team, TeamMember
from agora.api.deps import require_project
from agora.schemas.team import (
    TeamCreate,
    TeamUpdate,
    TeamOut,
    TeamMemberAdd,
    TeamMemberOut,
)

router = APIRouter(
    prefix="/api/projects/{project_slug}/teams",
    tags=["Teams"],
)


@router.post("", response_model=TeamOut, status_code=201)
async def create_team(
    project_slug: str,
    body: TeamCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    team = Team(
        project_id=project.id,
        name=body.name,
        description=body.description,
    )
    db.add(team)
    await db.commit()
    await db.refresh(team)
    return team


@router.get("", response_model=list[TeamOut])
async def list_teams(
    project_slug: str,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Team).where(Team.project_id == project.id).order_by(Team.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{team_id}", response_model=TeamOut)
async def get_team(
    project_slug: str,
    team_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Team).where(Team.id == team_id, Team.project_id == project.id)
    )
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(404, f"Team {team_id} not found in project '{project_slug}'")
    return team


@router.patch("/{team_id}", response_model=TeamOut)
async def update_team(
    project_slug: str,
    team_id: int,
    body: TeamUpdate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Team).where(Team.id == team_id, Team.project_id == project.id)
    )
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(404, f"Team {team_id} not found in project '{project_slug}'")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(team, key, value)

    await db.commit()
    await db.refresh(team)
    return team


@router.delete("/{team_id}", status_code=204)
async def delete_team(
    project_slug: str,
    team_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Team).where(Team.id == team_id, Team.project_id == project.id)
    )
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(404, f"Team {team_id} not found in project '{project_slug}'")
    await db.delete(team)
    await db.commit()


@router.post("/{team_id}/members", response_model=TeamMemberOut, status_code=201)
async def add_team_member(
    project_slug: str,
    team_id: int,
    body: TeamMemberAdd,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    # Verify team exists in project
    result = await db.execute(
        select(Team).where(Team.id == team_id, Team.project_id == project.id)
    )
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(404, f"Team {team_id} not found in project '{project_slug}'")

    # Verify agent exists
    agent_result = await db.execute(select(Agent).where(Agent.id == body.agent_id))
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent {body.agent_id} not found")

    # Check if already a member
    existing = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.agent_id == body.agent_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Agent {body.agent_id} is already a member of team {team_id}")

    member = TeamMember(
        team_id=team_id,
        agent_id=body.agent_id,
        role_in_team=body.role_in_team,
    )
    db.add(member)
    await db.commit()
    await db.refresh(member)

    # Return with agent_name
    return TeamMemberOut(
        id=member.id,
        team_id=member.team_id,
        agent_id=member.agent_id,
        role_in_team=member.role_in_team,
        joined_at=member.joined_at,
        agent_name=agent.name,
    )


@router.delete("/{team_id}/members/{agent_id}", status_code=204)
async def remove_team_member(
    project_slug: str,
    team_id: int,
    agent_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    # Verify team exists in project
    result = await db.execute(
        select(Team).where(Team.id == team_id, Team.project_id == project.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(404, f"Team {team_id} not found in project '{project_slug}'")

    member_result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.agent_id == agent_id,
        )
    )
    member = member_result.scalar_one_or_none()
    if not member:
        raise HTTPException(404, f"Agent {agent_id} is not a member of team {team_id}")
    await db.delete(member)
    await db.commit()

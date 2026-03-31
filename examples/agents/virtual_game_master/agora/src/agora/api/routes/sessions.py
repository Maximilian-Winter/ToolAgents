import secrets

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from agora.db.engine import get_db
from agora.db.models.agent import Agent, AgentSession
from agora.db.models.project import Project
from agora.api.deps import require_agent
from agora.schemas.session import LoginRequest, LoginResponse, SessionInfo

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    agent = await require_agent(body.name, body.token, db)

    # Validate project slug if provided
    if body.project:
        result = await db.execute(
            select(Project).where(Project.slug == body.project)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(404, f"Project '{body.project}' not found")

    session_token = secrets.token_hex(32)
    session = AgentSession(
        agent_id=agent.id,
        session_token=session_token,
        project_slug=body.project,
    )
    db.add(session)
    await db.commit()

    return LoginResponse(
        session_token=session_token,
        agent_name=agent.name,
        project=body.project,
    )


@router.post("/logout", status_code=204)
async def logout(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization header with Bearer token required")
    token = authorization[7:]
    result = await db.execute(
        select(AgentSession).where(AgentSession.session_token == token)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(401, "Invalid or expired session token")
    await db.delete(session)
    await db.commit()


@router.get("/me", response_model=SessionInfo)
async def get_session_info(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization header with Bearer token required")
    token = authorization[7:]
    result = await db.execute(
        select(AgentSession).where(AgentSession.session_token == token)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(401, "Invalid or expired session token")

    # Get agent name
    agent_result = await db.execute(select(Agent).where(Agent.id == session.agent_id))
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(401, "Agent for this session no longer exists")

    return SessionInfo(
        agent_name=agent.name,
        project=session.project_slug,
        created_at=session.created_at,
    )

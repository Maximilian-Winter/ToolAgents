import hashlib
from typing import Optional

from fastapi import Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agora.db.engine import get_db
from agora.db.models.project import Project
from agora.db.models.agent import Agent, AgentSession


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def require_agent(name: str, token: Optional[str], db: AsyncSession) -> Agent:
    """Look up agent by name and verify token. Returns Agent or raises 404/403."""
    result = await db.execute(select(Agent).where(Agent.name == name))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent '{name}' is not registered")
    if agent.token_hash is None and token is None:
        return agent
    if agent.token_hash is None and token is not None:
        raise HTTPException(403, f"Agent '{name}' was registered without a token")
    if agent.token_hash is not None and token is None:
        raise HTTPException(403, f"Agent '{name}' requires a token")
    if hash_token(token) != agent.token_hash:
        raise HTTPException(403, "Invalid token")
    return agent


async def require_project(slug: str, db: AsyncSession) -> Project:
    """Look up project by slug or raise 404."""
    result = await db.execute(select(Project).where(Project.slug == slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{slug}' not found")
    return project


async def get_current_agent_from_session(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> Optional[Agent]:
    """Extract agent from Bearer token in Authorization header. Returns None if no header."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    result = await db.execute(
        select(AgentSession).where(AgentSession.session_token == token)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(401, "Invalid or expired session token")
    # Update last_used_at
    from datetime import datetime, timezone

    session.last_used_at = datetime.now(timezone.utc)
    agent_result = await db.execute(select(Agent).where(Agent.id == session.agent_id))
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(401, "Agent for this session no longer exists")
    await db.commit()
    return agent

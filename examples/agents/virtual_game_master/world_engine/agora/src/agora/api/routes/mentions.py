"""Reverse lookup endpoint for cross-reference mentions."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.engine import get_db
from agora.db.models.mention import Mention
from agora.db.models.project import Project
from agora.schemas.mention import MentionOut

router = APIRouter(prefix="/api/projects/{project_slug}/mentions", tags=["Mentions"])


@router.get("", response_model=list[MentionOut])
async def get_mentions(
    project_slug: str,
    kb_path: Optional[str] = Query(None),
    issue_number: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.slug == project_slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{project_slug}' not found")

    stmt = select(Mention).where(Mention.project_id == project.id)

    if kb_path:
        stmt = stmt.where(Mention.mention_type == "kb", Mention.target_path == kb_path)
    elif issue_number is not None:
        stmt = stmt.where(Mention.mention_type == "issue", Mention.target_issue_number == issue_number)
    else:
        raise HTTPException(400, "Provide either kb_path or issue_number query parameter")

    stmt = stmt.order_by(Mention.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

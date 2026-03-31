import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func

from agora.db.engine import get_db
from agora.db.models.project import Project
from agora.db.models.chat import Room
from agora.db.models.task import Issue
from agora.db.models.enums import IssueState
from agora.api.deps import require_project
from agora.schemas.project import ProjectCreate, ProjectUpdate, ProjectOut

router = APIRouter(prefix="/api/projects", tags=["Projects"])


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate,
    db: AsyncSession = Depends(get_db),
):
    slug = slugify(body.name)

    # Check uniqueness of name and slug
    existing = await db.execute(
        select(Project).where((Project.name == body.name) | (Project.slug == slug))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Project with name '{body.name}' or slug '{slug}' already exists")

    project = Project(
        name=body.name,
        slug=slug,
        description=body.description,
        working_dir=body.working_dir,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    return result.scalars().all()


@router.get("/{slug}", response_model=ProjectOut)
async def get_project(slug: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.slug == slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{slug}' not found")
    return project


@router.patch("/{slug}", response_model=ProjectOut)
async def update_project(
    slug: str,
    body: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.slug == slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{slug}' not found")

    update_data = body.model_dump(exclude_unset=True)

    if "name" in update_data and update_data["name"] is not None:
        new_slug = slugify(update_data["name"])
        # Check uniqueness of new name/slug (excluding current project)
        existing = await db.execute(
            select(Project).where(
                ((Project.name == update_data["name"]) | (Project.slug == new_slug))
                & (Project.id != project.id)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                409,
                f"Project with name '{update_data['name']}' or slug '{new_slug}' already exists",
            )
        project.slug = new_slug

    for key, value in update_data.items():
        setattr(project, key, value)

    await db.commit()
    await db.refresh(project)
    return project


@router.get("/{slug}/stats")
async def get_project_stats(slug: str, db: AsyncSession = Depends(get_db)):
    project = await require_project(slug, db)

    room_count_result = await db.execute(
        select(func.count()).select_from(Room).where(Room.project_id == project.id)
    )
    room_count = room_count_result.scalar_one()

    total_issue_result = await db.execute(
        select(func.count()).select_from(Issue).where(Issue.project_id == project.id)
    )
    total_issue_count = total_issue_result.scalar_one()

    open_issue_result = await db.execute(
        select(func.count())
        .select_from(Issue)
        .where(Issue.project_id == project.id, Issue.state == IssueState.open)
    )
    open_issue_count = open_issue_result.scalar_one()

    return {
        "room_count": room_count,
        "open_issue_count": open_issue_count,
        "total_issue_count": total_issue_count,
    }


@router.delete("/{slug}", status_code=204)
async def delete_project(slug: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.slug == slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{slug}' not found")
    await db.delete(project)
    await db.commit()

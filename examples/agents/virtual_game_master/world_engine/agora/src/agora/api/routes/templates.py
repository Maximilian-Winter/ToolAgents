"""CRUD for document templates + render endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.engine import get_db
from agora.db.models.template import DocumentTemplate
from agora.db.models.project import Project
from agora.schemas.template import (
    TemplateCreate,
    TemplateUpdate,
    TemplateOut,
    RenderRequest,
    RenderResponse,
)
from agora.services.template_engine import build_context, render_template

global_templates_router = APIRouter(prefix="/api/templates", tags=["templates"])


@global_templates_router.post("", response_model=TemplateOut, status_code=201)
async def create_global_template(body: TemplateCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(DocumentTemplate).where(
            DocumentTemplate.name == body.name,
            DocumentTemplate.project_id.is_(None),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Global template '{body.name}' already exists")

    template = DocumentTemplate(**body.model_dump(), project_id=None)
    db.add(template)
    await db.commit()
    await db.refresh(template)
    return template


@global_templates_router.get("", response_model=list[TemplateOut])
async def list_global_templates(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DocumentTemplate)
        .where(DocumentTemplate.project_id.is_(None))
        .order_by(DocumentTemplate.name)
    )
    return result.scalars().all()


@global_templates_router.get("/{template_id}", response_model=TemplateOut)
async def get_template(template_id: int, db: AsyncSession = Depends(get_db)):
    template = await db.get(DocumentTemplate, template_id)
    if not template:
        raise HTTPException(404, "Template not found")
    return template


@global_templates_router.patch("/{template_id}", response_model=TemplateOut)
async def update_template(
    template_id: int, body: TemplateUpdate, db: AsyncSession = Depends(get_db)
):
    template = await db.get(DocumentTemplate, template_id)
    if not template:
        raise HTTPException(404, "Template not found")

    updates = body.model_dump(exclude_unset=True)
    if "name" in updates:
        existing = await db.execute(
            select(DocumentTemplate).where(
                DocumentTemplate.name == updates["name"],
                DocumentTemplate.project_id == template.project_id,
                DocumentTemplate.id != template_id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(409, f"Template '{updates['name']}' already exists in this scope")

    for key, val in updates.items():
        setattr(template, key, val)
    await db.commit()
    await db.refresh(template)
    return template


@global_templates_router.delete("/{template_id}", status_code=204)
async def delete_template(template_id: int, db: AsyncSession = Depends(get_db)):
    template = await db.get(DocumentTemplate, template_id)
    if not template:
        raise HTTPException(404, "Template not found")
    await db.delete(template)
    await db.commit()


@global_templates_router.post("/{template_id}/render", response_model=RenderResponse)
async def render_template_endpoint(
    template_id: int, body: RenderRequest, request: Request, db: AsyncSession = Depends(get_db)
):
    template = await db.get(DocumentTemplate, template_id)
    if not template:
        raise HTTPException(404, "Template not found")

    server_url = str(request.base_url).rstrip("/")
    context = await build_context(body.project_slug, body.agent_name, server_url, db)
    rendered, unresolved = render_template(template.content, context)
    return RenderResponse(rendered_content=rendered, unresolved_variables=unresolved)


project_templates_router = APIRouter(
    prefix="/api/projects/{project_slug}/templates", tags=["templates"]
)


async def _get_project(project_slug: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.slug == project_slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{project_slug}' not found")
    return project


@project_templates_router.post("", response_model=TemplateOut, status_code=201)
async def create_project_template(
    project_slug: str, body: TemplateCreate, db: AsyncSession = Depends(get_db)
):
    project = await _get_project(project_slug, db)

    existing = await db.execute(
        select(DocumentTemplate).where(
            DocumentTemplate.name == body.name,
            DocumentTemplate.project_id == project.id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Template '{body.name}' already exists in this project")

    template = DocumentTemplate(**body.model_dump(), project_id=project.id)
    db.add(template)
    await db.commit()
    await db.refresh(template)
    return template


@project_templates_router.get("", response_model=list[TemplateOut])
async def list_project_templates(project_slug: str, db: AsyncSession = Depends(get_db)):
    project = await _get_project(project_slug, db)

    result = await db.execute(
        select(DocumentTemplate).where(
            or_(
                DocumentTemplate.project_id.is_(None),
                DocumentTemplate.project_id == project.id,
            )
        ).order_by(DocumentTemplate.name)
    )
    all_templates = result.scalars().all()

    by_name: dict[str, DocumentTemplate] = {}
    for t in all_templates:
        if t.name not in by_name or t.project_id is not None:
            by_name[t.name] = t

    return list(by_name.values())

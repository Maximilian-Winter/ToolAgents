"""CRUD for custom field definitions and values on agents/projects."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.engine import get_db
from agora.db.models.custom_field import CustomFieldDefinition, CustomFieldValue
from agora.db.models.agent import Agent
from agora.db.models.project import Project
from agora.schemas.custom_field import (
    CustomFieldDefinitionCreate,
    CustomFieldDefinitionUpdate,
    CustomFieldDefinitionOut,
    CustomFieldValueSet,
)
from agora.services.field_validation import validate_field_value

# ── Field Definition CRUD ──────────────────────────────────────────

definitions_router = APIRouter(prefix="/api/custom-fields", tags=["custom-fields"])


@definitions_router.post("", response_model=CustomFieldDefinitionOut, status_code=201)
async def create_field_definition(
    body: CustomFieldDefinitionCreate, db: AsyncSession = Depends(get_db)
):
    existing = await db.execute(
        select(CustomFieldDefinition).where(
            CustomFieldDefinition.name == body.name,
            CustomFieldDefinition.entity_type == body.entity_type,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Field '{body.name}' already exists for {body.entity_type}")

    field_def = CustomFieldDefinition(**body.model_dump())
    db.add(field_def)
    await db.commit()
    await db.refresh(field_def)
    return field_def


@definitions_router.get("", response_model=list[CustomFieldDefinitionOut])
async def list_field_definitions(
    entity_type: str | None = Query(None, pattern=r"^(agent|project)$"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(CustomFieldDefinition).order_by(CustomFieldDefinition.sort_order)
    if entity_type:
        stmt = stmt.where(CustomFieldDefinition.entity_type == entity_type)
    result = await db.execute(stmt)
    return result.scalars().all()


@definitions_router.get("/{field_id}", response_model=CustomFieldDefinitionOut)
async def get_field_definition(field_id: int, db: AsyncSession = Depends(get_db)):
    field_def = await db.get(CustomFieldDefinition, field_id)
    if not field_def:
        raise HTTPException(404, "Field definition not found")
    return field_def


@definitions_router.patch("/{field_id}", response_model=CustomFieldDefinitionOut)
async def update_field_definition(
    field_id: int, body: CustomFieldDefinitionUpdate, db: AsyncSession = Depends(get_db)
):
    field_def = await db.get(CustomFieldDefinition, field_id)
    if not field_def:
        raise HTTPException(404, "Field definition not found")
    for key, val in body.model_dump(exclude_unset=True).items():
        setattr(field_def, key, val)
    await db.commit()
    await db.refresh(field_def)
    return field_def


@definitions_router.delete("/{field_id}", status_code=204)
async def delete_field_definition(field_id: int, db: AsyncSession = Depends(get_db)):
    field_def = await db.get(CustomFieldDefinition, field_id)
    if not field_def:
        raise HTTPException(404, "Field definition not found")
    await db.delete(field_def)
    await db.commit()


# ── Field Values on Agents ──────────────────────────────────────────

agent_fields_router = APIRouter(prefix="/api/agents/{agent_name}/fields", tags=["custom-fields"])


async def _get_agent_by_name(agent_name: str, db: AsyncSession) -> Agent:
    result = await db.execute(select(Agent).where(Agent.name == agent_name))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, f"Agent '{agent_name}' not found")
    return agent


async def _get_field_values(entity_type: str, entity_id: int, db: AsyncSession) -> dict[str, str]:
    stmt = (
        select(CustomFieldDefinition.name, CustomFieldValue.value)
        .join(CustomFieldValue, CustomFieldDefinition.id == CustomFieldValue.field_id)
        .where(
            CustomFieldDefinition.entity_type == entity_type,
            CustomFieldValue.entity_id == entity_id,
        )
    )
    result = await db.execute(stmt)
    return {row.name: row.value for row in result.all()}


async def _set_field_values(
    entity_type: str, entity_id: int, fields: dict[str, str], db: AsyncSession
) -> dict[str, str]:
    result_values = {}
    for field_name, raw_value in fields.items():
        stmt = select(CustomFieldDefinition).where(
            CustomFieldDefinition.name == field_name,
            CustomFieldDefinition.entity_type == entity_type,
        )
        result = await db.execute(stmt)
        field_def = result.scalar_one_or_none()
        if not field_def:
            raise HTTPException(422, f"Unknown {entity_type} field: '{field_name}'")

        try:
            validated = validate_field_value(raw_value, field_def.field_type, field_def.options_json)
        except ValueError as e:
            raise HTTPException(422, f"Field '{field_name}': {e}")

        existing = await db.execute(
            select(CustomFieldValue).where(
                CustomFieldValue.field_id == field_def.id,
                CustomFieldValue.entity_id == entity_id,
            )
        )
        fv = existing.scalar_one_or_none()
        if fv:
            fv.value = validated
        else:
            fv = CustomFieldValue(field_id=field_def.id, entity_id=entity_id, value=validated)
            db.add(fv)
        result_values[field_name] = validated

    await db.commit()
    return result_values


@agent_fields_router.get("")
async def get_agent_fields(agent_name: str, db: AsyncSession = Depends(get_db)):
    agent = await _get_agent_by_name(agent_name, db)
    return await _get_field_values("agent", agent.id, db)


@agent_fields_router.put("")
async def set_agent_fields(agent_name: str, body: dict[str, str], db: AsyncSession = Depends(get_db)):
    agent = await _get_agent_by_name(agent_name, db)
    return await _set_field_values("agent", agent.id, body, db)


@agent_fields_router.put("/{field_name}")
async def set_agent_field(
    agent_name: str, field_name: str, body: CustomFieldValueSet, db: AsyncSession = Depends(get_db)
):
    agent = await _get_agent_by_name(agent_name, db)
    result = await _set_field_values("agent", agent.id, {field_name: body.value}, db)
    return result


# ── Field Values on Projects ──────────────────────────────────────────

project_fields_router = APIRouter(prefix="/api/projects/{project_slug}/fields", tags=["custom-fields"])


async def _get_project_by_slug(project_slug: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.slug == project_slug))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project '{project_slug}' not found")
    return project


@project_fields_router.get("")
async def get_project_fields(project_slug: str, db: AsyncSession = Depends(get_db)):
    project = await _get_project_by_slug(project_slug, db)
    return await _get_field_values("project", project.id, db)


@project_fields_router.put("")
async def set_project_fields(
    project_slug: str, body: dict[str, str], db: AsyncSession = Depends(get_db)
):
    project = await _get_project_by_slug(project_slug, db)
    return await _set_field_values("project", project.id, body, db)


@project_fields_router.put("/{field_name}")
async def set_project_field(
    project_slug: str, field_name: str, body: CustomFieldValueSet, db: AsyncSession = Depends(get_db)
):
    project = await _get_project_by_slug(project_slug, db)
    result = await _set_field_values("project", project.id, {field_name: body.value}, db)
    return result

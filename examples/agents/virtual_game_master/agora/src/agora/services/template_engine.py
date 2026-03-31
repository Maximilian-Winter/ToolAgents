"""Template rendering engine with {{variable}} substitution."""

import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.models.agent import Agent, AgentPersona
from agora.db.models.project import Project
from agora.db.models.project_agent import ProjectAgent
from agora.db.models.custom_field import CustomFieldDefinition, CustomFieldValue

VARIABLE_PATTERN = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


async def build_context(
    project_slug: str,
    agent_name: Optional[str],
    server_url: str,
    db: AsyncSession,
) -> dict[str, str]:
    context: dict[str, str] = {}

    context["platform.server_url"] = server_url
    context["platform.date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    result = await db.execute(select(Project).where(Project.slug == project_slug))
    project = result.scalar_one_or_none()
    if not project:
        return context

    context["project.name"] = project.name or ""
    context["project.slug"] = project.slug or ""
    context["project.description"] = project.description or ""
    context["project.working_dir"] = project.working_dir or ""

    proj_fields = await _get_custom_fields("project", project.id, db)
    for fname, fval in proj_fields.items():
        context[f"project.fields.{fname}"] = fval

    if agent_name:
        agent_result = await db.execute(select(Agent).where(Agent.name == agent_name))
        agent = agent_result.scalar_one_or_none()
        if agent:
            context["agent.name"] = agent.name or ""
            context["agent.display_name"] = agent.display_name or ""
            context["agent.role"] = agent.role or ""

            agent_fields = await _get_custom_fields("agent", agent.id, db)
            for fname, fval in agent_fields.items():
                context[f"agent.fields.{fname}"] = fval

            pa_result = await db.execute(
                select(ProjectAgent)
                .where(ProjectAgent.project_id == project.id, ProjectAgent.agent_id == agent.id)
            )
            pa = pa_result.scalar_one_or_none()
            if pa:
                system_prompt = pa.system_prompt or ""
                if agent.persona_id:
                    persona_result = await db.execute(
                        select(AgentPersona).where(AgentPersona.id == agent.persona_id)
                    )
                    persona = persona_result.scalar_one_or_none()
                    if persona and persona.system_prompt:
                        if pa.prompt_source == "override":
                            system_prompt = pa.system_prompt or persona.system_prompt or ""
                        else:
                            parts = [p for p in [persona.system_prompt, pa.system_prompt] if p]
                            system_prompt = "\n\n".join(parts)

                context["agent.system_prompt"] = system_prompt
                context["agent.initial_task"] = pa.initial_task or ""
                context["agent.model"] = pa.model or ""
                context["agent.prompt_source"] = pa.prompt_source or "append"
                context["agent.runtime"] = getattr(pa, "runtime", None) or ""

    return context


async def _get_custom_fields(entity_type: str, entity_id: int, db: AsyncSession) -> dict[str, str]:
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


def render_template(template_content: str, context: dict[str, str]) -> tuple[str, list[str]]:
    unresolved: list[str] = []

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in context:
            return context[var_name]
        unresolved.append(var_name)
        return match.group(0)

    rendered = VARIABLE_PATTERN.sub(replace_var, template_content)
    return rendered, unresolved

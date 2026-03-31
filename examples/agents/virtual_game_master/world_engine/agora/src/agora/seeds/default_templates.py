"""Seed default global document templates."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.models.template import DocumentTemplate


CLAUDE_CODE_STARTUP_SH = """#!/bin/bash
# Startup script for {{agent.name}} on project {{project.name}}
cd "{{project.working_dir}}"

agora login {{agent.name}} --project {{project.slug}} --server {{platform.server_url}}

claude \\
  --{{agent.prompt_source}}-system-prompt "{{agent.system_prompt}}" \\
  --model {{agent.model}} \\
  "{{agent.initial_task}}"
"""

CLAUDE_CODE_STARTUP_BAT = """@echo off
REM Startup script for {{agent.name}} on project {{project.name}}
cd /d "{{project.working_dir}}"

call agora login {{agent.name}} --project {{project.slug}} --server {{platform.server_url}}

claude ^
  --{{agent.prompt_source}}-system-prompt "{{agent.system_prompt}}" ^
  --model {{agent.model}} ^
  "{{agent.initial_task}}"
"""

SYSTEM_PROMPT_TEMPLATE = """You are {{agent.display_name}}, working on the {{project.name}} project.

Role: {{agent.role}}

Project: {{project.description}}
"""


DEFAULT_TEMPLATES = [
    {
        "name": "Claude Code Startup Script (Unix)",
        "description": "Shell script to launch a Claude Code agent",
        "type_tag": "startup-script",
        "content": CLAUDE_CODE_STARTUP_SH.strip(),
    },
    {
        "name": "Claude Code Startup Script (Windows)",
        "description": "Batch script to launch a Claude Code agent",
        "type_tag": "startup-script",
        "content": CLAUDE_CODE_STARTUP_BAT.strip(),
    },
    {
        "name": "Agent System Prompt",
        "description": "Basic system prompt template for any agent",
        "type_tag": "system-prompt",
        "content": SYSTEM_PROMPT_TEMPLATE.strip(),
    },
]


async def seed_default_templates(db: AsyncSession) -> None:
    """Insert default global templates if they don't already exist."""
    for tmpl_data in DEFAULT_TEMPLATES:
        existing = await db.execute(
            select(DocumentTemplate).where(
                DocumentTemplate.name == tmpl_data["name"],
                DocumentTemplate.project_id.is_(None),
            )
        )
        if not existing.scalar_one_or_none():
            db.add(DocumentTemplate(**tmpl_data, project_id=None))
    await db.commit()

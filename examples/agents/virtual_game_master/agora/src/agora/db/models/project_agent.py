from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from agora.db.base import Base


class ProjectAgent(Base):
    """Links a global agent preset to a project with per-project configuration.

    Agents are global templates. When added to a project, a ProjectAgent record
    stores the project-specific overrides for system prompt, model, tools, etc.
    """

    __tablename__ = "project_agents"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    # Per-project configuration overrides
    system_prompt = Column(Text, nullable=True)
    initial_task = Column(Text, nullable=True)
    model = Column(String(100), nullable=True)  # e.g. "claude-sonnet-4-5-20250514"
    allowed_tools = Column(Text, nullable=True)  # comma-separated tool names
    prompt_source = Column(
        String(20), nullable=False, default="append"
    )  # "append" (--append-system-prompt) or "override" (--system-prompt)
    runtime = Column(String(50), nullable=True)  # e.g. "claude-code", "aider", "custom"
    extra_flags = Column(Text, nullable=True)  # JSON object for runtime-agnostic flags

    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("project_id", "agent_id", name="uq_project_agent"),
    )

    project = relationship("Project", back_populates="project_agents")
    agent = relationship("Agent", back_populates="project_assignments")

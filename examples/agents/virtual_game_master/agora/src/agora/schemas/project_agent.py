from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional


class ProjectAgentAdd(BaseModel):
    """Add a global agent preset to this project with optional config overrides."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent_name: str = Field(..., min_length=1, max_length=100)
    system_prompt: Optional[str] = None
    initial_task: Optional[str] = None
    model: Optional[str] = Field(None, max_length=100)
    allowed_tools: Optional[str] = None
    prompt_source: str = Field("append", pattern=r"^(append|override)$")
    runtime: Optional[str] = Field(None, max_length=50)
    extra_flags: Optional[str] = None  # JSON string


class ProjectAgentUpdate(BaseModel):
    """Update per-project agent configuration."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    system_prompt: Optional[str] = None
    initial_task: Optional[str] = None
    model: Optional[str] = Field(None, max_length=100)
    allowed_tools: Optional[str] = None
    prompt_source: Optional[str] = Field(None, pattern=r"^(append|override)$")
    runtime: Optional[str] = Field(None, max_length=50)
    extra_flags: Optional[str] = None


class ProjectAgentOut(BaseModel):
    """ProjectAgent with nested agent info for API responses."""

    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    agent_id: int
    system_prompt: Optional[str]
    initial_task: Optional[str]
    model: Optional[str]
    allowed_tools: Optional[str]
    prompt_source: str
    runtime: Optional[str]
    extra_flags: Optional[str]
    added_at: datetime
    # Flattened agent fields
    agent_name: str
    agent_display_name: Optional[str]
    agent_role: Optional[str]

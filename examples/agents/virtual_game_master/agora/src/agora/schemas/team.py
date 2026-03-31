from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional


class TeamCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class TeamUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    name: str
    description: Optional[str]
    created_at: datetime


class TeamMemberAdd(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent_id: int
    role_in_team: Optional[str] = Field(None, max_length=200)


class TeamMemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    team_id: int
    agent_id: int
    role_in_team: Optional[str]
    joined_at: datetime
    agent_name: str

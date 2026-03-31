from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TemplateCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    type_tag: Optional[str] = Field(None, max_length=100)
    content: str = Field(..., min_length=1)


class TemplateUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    type_tag: Optional[str] = Field(None, max_length=100)
    content: Optional[str] = Field(None, min_length=1)


class TemplateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: Optional[str]
    type_tag: Optional[str]
    content: str
    project_id: Optional[int]
    created_at: datetime
    updated_at: datetime


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_slug: str
    agent_name: Optional[str] = None


class RenderResponse(BaseModel):
    rendered_content: str
    unresolved_variables: list[str]

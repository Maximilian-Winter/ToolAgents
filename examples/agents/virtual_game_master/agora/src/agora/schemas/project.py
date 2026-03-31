from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional


class ProjectCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    working_dir: Optional[str] = Field(None, max_length=500)


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    working_dir: Optional[str] = Field(None, max_length=500)


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    slug: str
    description: Optional[str]
    working_dir: Optional[str]
    created_at: datetime
    updated_at: datetime

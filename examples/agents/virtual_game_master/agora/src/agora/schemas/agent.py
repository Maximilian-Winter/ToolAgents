from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional


# --- Agent schemas ---


class AgentCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    role: Optional[str] = Field(None, max_length=200)
    token: Optional[str] = None  # NOT stored directly; hashed on registration


class AgentUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    display_name: Optional[str] = Field(None, max_length=200)
    role: Optional[str] = Field(None, max_length=200)
    persona_id: Optional[int] = None


class AgentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    display_name: Optional[str]
    role: Optional[str]
    persona_id: Optional[int]
    created_at: datetime


# --- Persona schemas ---


class PersonaCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: str = Field(..., min_length=1)


class PersonaUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: Optional[str] = Field(None, min_length=1)


class PersonaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: Optional[str]
    system_prompt: str
    created_at: datetime
    updated_at: datetime

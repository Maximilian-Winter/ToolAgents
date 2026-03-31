from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    token: Optional[str] = None
    project: Optional[str] = None  # project slug


class LoginResponse(BaseModel):
    session_token: str
    agent_name: str
    project: Optional[str] = None


class SessionInfo(BaseModel):
    agent_name: str
    project: Optional[str] = None
    created_at: datetime

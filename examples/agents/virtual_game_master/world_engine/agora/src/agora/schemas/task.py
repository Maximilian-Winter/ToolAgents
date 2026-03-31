from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional

from agora.db.models.enums import IssueState, Priority


# --- Issues ---
class IssueCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1, max_length=500)
    body: Optional[str] = None
    priority: Priority = Priority.none
    assignee: Optional[str] = Field(None, max_length=100)  # agent name
    reporter: str = Field(..., min_length=1, max_length=100)  # agent or user name
    milestone_id: Optional[int] = None
    labels: Optional[list[str]] = None  # label names to attach


class IssueUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    body: Optional[str] = None
    state: Optional[IssueState] = None
    priority: Optional[Priority] = None
    assignee: Optional[str] = Field(None, max_length=100)
    milestone_id: Optional[int] = None


class IssueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    number: int
    title: str
    body: Optional[str]
    state: IssueState
    priority: Priority
    assignee: Optional[str]
    reporter: str
    milestone_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]
    labels: list["LabelOut"] = []
    comment_count: int = 0


# --- Comments ---
class CommentCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    author: str = Field(..., min_length=1, max_length=100)
    body: str = Field(..., min_length=1)


class CommentUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    body: str = Field(..., min_length=1)


class CommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    issue_id: int
    author: str
    body: str
    created_at: datetime
    updated_at: datetime


# --- Labels ---
class LabelCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    color: Optional[str] = Field(None, max_length=7)  # hex color
    description: Optional[str] = Field(None, max_length=500)


class LabelUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    color: Optional[str] = Field(None, max_length=7)
    description: Optional[str] = Field(None, max_length=500)


class LabelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    name: str
    color: Optional[str]
    description: Optional[str]


# --- Milestones ---
class MilestoneCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None


class MilestoneUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    state: Optional[IssueState] = None


class MilestoneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    title: str
    description: Optional[str]
    due_date: Optional[datetime]
    state: IssueState
    created_at: datetime
    open_issues: int = 0
    closed_issues: int = 0


# --- Activity ---
class ActivityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    issue_id: int
    actor: str
    action: str
    detail_json: Optional[str]
    created_at: datetime


# --- Dependencies ---
class DependencyCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    depends_on_number: int  # issue number (not id) in the same project


class DependencyOut(BaseModel):
    issue_number: int
    depends_on_number: int


# --- Label attachment ---
class LabelAttach(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    label_name: str = Field(..., min_length=1, max_length=100)

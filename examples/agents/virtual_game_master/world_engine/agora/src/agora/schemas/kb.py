"""Pydantic schemas for knowledge base documents."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class KBDocumentCreate(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)
    title: Optional[str] = Field(None, max_length=200)
    tags: Optional[str] = None  # comma-separated
    content: str = Field(..., min_length=0)
    author: str = Field(..., min_length=1, max_length=100)


class KBDocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    path: str
    title: str
    tags: Optional[str] = None
    content: str
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime


class KBDocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    path: str
    title: str
    tags: Optional[str] = None
    updated_by: str
    updated_at: datetime


class KBDocumentMove(BaseModel):
    new_path: str = Field(..., min_length=1, max_length=500)


class KBSearchResult(BaseModel):
    path: str
    title: str
    snippet: str
    rank: float


class KBTreeNode(BaseModel):
    name: str
    path: Optional[str] = None  # only for leaf nodes (files)
    title: Optional[str] = None  # only for leaf nodes
    children: Optional[list["KBTreeNode"]] = None  # only for directory nodes

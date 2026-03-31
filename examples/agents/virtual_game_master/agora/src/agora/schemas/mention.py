"""Pydantic schemas for cross-reference mentions."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MentionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_type: str
    source_id: int
    mention_type: str
    target_path: Optional[str] = None
    target_issue_number: Optional[int] = None
    created_at: datetime

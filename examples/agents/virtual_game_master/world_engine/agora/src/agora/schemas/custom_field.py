import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomFieldDefinitionCreate(BaseModel):
    """Create a new custom field definition."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(..., min_length=1, max_length=200)
    field_type: str = Field(..., pattern=r"^(string|number|boolean|enum)$")
    entity_type: str = Field(..., pattern=r"^(agent|project)$")
    options_json: Optional[str] = None  # JSON array string for enum choices
    default_value: Optional[str] = None
    required: bool = False
    sort_order: int = 0

    @field_validator("options_json")
    @classmethod
    def validate_options_json(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, list) or not all(isinstance(i, str) for i in parsed):
                    raise ValueError("options_json must be a JSON array of strings")
            except json.JSONDecodeError:
                raise ValueError("options_json must be valid JSON")
        return v


class CustomFieldDefinitionUpdate(BaseModel):
    """Update a custom field definition."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    label: Optional[str] = Field(None, min_length=1, max_length=200)
    options_json: Optional[str] = None
    default_value: Optional[str] = None
    required: Optional[bool] = None
    sort_order: Optional[int] = None

    @field_validator("options_json")
    @classmethod
    def validate_options_json(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, list) or not all(isinstance(i, str) for i in parsed):
                    raise ValueError("options_json must be a JSON array of strings")
            except json.JSONDecodeError:
                raise ValueError("options_json must be valid JSON")
        return v


class CustomFieldDefinitionOut(BaseModel):
    """Custom field definition response."""

    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    label: str
    field_type: str
    entity_type: str
    options_json: Optional[str]
    default_value: Optional[str]
    required: bool
    sort_order: int
    created_at: datetime
    updated_at: datetime


class CustomFieldValueSet(BaseModel):
    """Set a single custom field value."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    value: str

"""Validates custom field values against their field definitions."""

import json
from typing import Optional


def validate_field_value(value: str, field_type: str, options_json: Optional[str] = None) -> str:
    """Validate and normalize a field value based on its type.

    Args:
        value: The raw string value to validate.
        field_type: One of "string", "number", "boolean", "enum".
        options_json: JSON array of valid enum choices (required for enum type).

    Returns:
        The normalized value string.

    Raises:
        ValueError: If the value is invalid for the field type.
    """
    if field_type == "string":
        return value

    if field_type == "number":
        try:
            float(value)
        except ValueError:
            raise ValueError(f"Value '{value}' is not a valid number")
        return value

    if field_type == "boolean":
        lower = value.lower()
        if lower not in ("true", "false"):
            raise ValueError(f"Value '{value}' is not a valid boolean (must be 'true' or 'false')")
        return lower

    if field_type == "enum":
        if options_json is None:
            raise ValueError("Enum field has no options defined")
        options = json.loads(options_json)
        if value not in options:
            raise ValueError(f"Value '{value}' is not one of the allowed options: {options}")
        return value

    raise ValueError(f"Unknown field type: {field_type}")

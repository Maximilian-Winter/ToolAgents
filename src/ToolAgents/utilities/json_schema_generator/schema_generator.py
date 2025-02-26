from collections import OrderedDict
from inspect import isclass
from enum import Enum
from pydantic import BaseModel, Field
from typing import Union, get_origin, get_args, List, Any, Type
from types import NoneType


# New models for customization

class AdditionalFieldPosition(Enum):
    before = "before"
    after = "after"


class AdditionalSchemaField(BaseModel):
    """
    Represents an additional schema field.
    """
    name: str = Field(..., description="The name of the field.")
    title: str = Field(..., description="The title of the field.")
    description: str = Field(..., description="The description of the field.")
    type: str = Field(..., description="The type of the field.")
    required: bool = Field(..., description="Whether the field is required.")
    position: AdditionalFieldPosition = Field(
        default=AdditionalFieldPosition.after,
        description="The position at which the field should be inserted."
    )


class SchemaObject(BaseModel):
    """
    Represents a schema object.
    """
    model: Type[BaseModel]
    additional_fields: List[AdditionalSchemaField] = []


class OuterSchemaObject(BaseModel):
    """
    Represents an outer object around a schema.
    """
    name: str = Field(..., description="The name of the outer object.")
    description: str = Field(..., description="The description of the outer object.")
    schemas: List[SchemaObject] = Field(..., description="The list of schema objects.")
    additional_fields: List[AdditionalSchemaField] = Field(default_factory=list)
    type: str = Field(..., description="The type of the outer object.")


# Original custom JSON schema generator (refined for clarity)

def get_json_type(annotation):
    """Map Python basic types to JSON types."""
    mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        NoneType: "null",
    }
    return mapping.get(annotation)


def refine_schema(schema: dict, model: Type[BaseModel]) -> dict:
    """
    Refine the generated schema based on the model's annotations and field details.
    Recursively handles Enums, Unions, lists/sets, dictionaries, and nested Pydantic models.
    """
    if "properties" not in schema:
        return schema

    for name, prop in schema["properties"].items():
        field = model.model_fields.get(name)
        if not field:
            continue

        # Update title and description
        prop["title"] = name.replace("_", " ").title()
        prop["description"] = field.description or ""

        annotation = field.annotation
        origin = get_origin(annotation)

        # Handle Enums
        if isclass(annotation) and issubclass(annotation, Enum):
            prop.pop("allOf", None)
            prop["enum"] = [e.value for e in annotation]
            first_value = next(iter(annotation)).value  # Assume uniform type
            prop["type"] = get_json_type(type(first_value))
            prop.pop("$ref", None)
            continue

        # Handle Unions (including Optional)
        if origin is Union:
            types = get_args(annotation)
            anyof_list = []
            for sub_type in types:
                type_str = get_json_type(sub_type)
                if sub_type is NoneType:
                    anyof_list.append({"type": "null"})
                elif isclass(sub_type) and issubclass(sub_type, BaseModel):
                    sub_schema = refine_schema(sub_type.model_json_schema(), sub_type)
                    anyof_list.append(sub_schema)
                elif type_str:
                    anyof_list.append({"type": type_str})
            prop["anyOf"] = anyof_list
            continue

        # Handle lists and sets
        if origin in [list, set]:
            item_type = get_args(annotation)[0]
            if isclass(item_type) and issubclass(item_type, BaseModel):
                prop["items"] = refine_schema(item_type.model_json_schema(), item_type)
            elif item_type is Any:
                prop["items"] = {
                    "type": "object",
                    "anyOf": [{"type": t} for t in ["boolean", "number", "null", "string"]],
                }
            else:
                item_origin = get_origin(item_type)
                if item_origin is Union:
                    types = get_args(item_type)
                    anyof_list = []
                    for sub_type in types:
                        type_str = get_json_type(sub_type)
                        if sub_type is NoneType:
                            anyof_list.append({"type": "null"})
                        elif isclass(sub_type) and issubclass(sub_type, BaseModel):
                            sub_schema = refine_schema(sub_type.model_json_schema(), sub_type)
                            anyof_list.append(sub_schema)
                        elif type_str:
                            anyof_list.append({"type": type_str})
                    prop["items"] = {"anyOf": anyof_list}
                else:
                    type_str = get_json_type(item_type)
                    if type_str:
                        prop["items"] = {"type": type_str}
            prop["minItems"] = 1
            continue

        # Handle dictionaries
        if origin is dict:
            _, value_type = get_args(annotation)
            if isclass(value_type) and issubclass(value_type, BaseModel):
                prop["additionalProperties"] = refine_schema(value_type.model_json_schema(), value_type)
            else:
                value_type_str = get_json_type(value_type)
                prop["additionalProperties"] = {"type": value_type_str} if value_type_str else {}
            prop["type"] = "object"
            continue

        # Handle nested Pydantic models
        if isclass(annotation) and issubclass(annotation, BaseModel):
            nested_schema = refine_schema(annotation.model_json_schema(), annotation)
            prop.update(nested_schema)

    # Set top-level title and description
    schema["title"] = model.__name__
    schema["description"] = (model.__doc__ or "").strip()

    # Define required fields based on the model definition
    required_fields = [name for name, field in model.model_fields.items() if field.is_required()]
    if required_fields:
        schema["required"] = required_fields

    # Clean up unwanted keys
    schema.pop("$defs", None)
    schema.pop("$ref", None)

    return schema


def custom_json_schema(model: Type[BaseModel]) -> dict:
    """
    Generate a custom JSON schema for a given Pydantic model.
    """
    base_schema = model.model_json_schema()
    return refine_schema(base_schema, model)


# New helper to merge additional schema fields into an existing schema

def insert_additional_fields(schema: dict, additional_fields: List[AdditionalSchemaField]) -> dict:
    """
    Insert additional fields into the schema's properties.
    Fields with position 'before' will be added at the start,
    and those with 'after' will be appended at the end.
    """
    if "properties" not in schema:
        schema["properties"] = {}

    # Convert properties to an OrderedDict to preserve insertion order.
    orig_props = OrderedDict(schema["properties"])
    before = OrderedDict()
    after = OrderedDict()
    before_required_fields = []
    after_required_fields = []
    # Create property definitions for additional fields.
    for field in additional_fields:
        field_schema = {
            "description": field.description,
            "title": field.title,
            "type": field.type,
        }
        if field.position == AdditionalFieldPosition.before:
            before[field.name] = field_schema
            if field.required:
                before_required_fields.append(field.name)
        else:
            after[field.name] = field_schema
            if field.required:
                after_required_fields.append(field.name)
    # Merge additional fields with original properties.
    merged = OrderedDict()
    merged.update(before)
    merged.update(orig_props)
    merged.update(after)
    schema["properties"] = dict(merged)
    before_required_fields.extend(schema["required"])
    schema["required"] = before_required_fields
    schema["required"].extend(after_required_fields)
    return schema


# New function to generate a JSON schema for a single SchemaObject

def generate_schema_object(schema_obj: SchemaObject) -> dict:
    """
    Generate the JSON schema for a SchemaObject by processing its model
    and inserting any additional fields.
    """
    # Generate the base schema from the model.
    base_schema = custom_json_schema(schema_obj.model)
    # Insert additional fields (if any) into the schema.
    return insert_additional_fields(base_schema, schema_obj.additional_fields)


# New function to generate the outer JSON schema from an OuterSchemaObject

def generate_outer_json_schema(outer_obj: OuterSchemaObject) -> dict:
    """
    Generate the outer JSON schema based on an OuterSchemaObject.
    Combines inner schemas (from each SchemaObject) using "anyOf" if needed,
    and then merges in outer additional fields.
    """
    # Process each inner schema.
    inner_schemas = [generate_schema_object(s) for s in outer_obj.schemas]

    # Combine inner schemas: if only one, use it directly; otherwise use "anyOf".
    if len(inner_schemas) == 1:
        combined_schema = inner_schemas[0]
    else:
        combined_schema = {"anyOf": inner_schemas}

    # Insert additional outer fields.
    combined_schema = insert_additional_fields(combined_schema, outer_obj.additional_fields)

    # Set the outer schema details.
    combined_schema["title"] = outer_obj.name
    combined_schema["description"] = outer_obj.description
    combined_schema["type"] = outer_obj.type

    return combined_schema

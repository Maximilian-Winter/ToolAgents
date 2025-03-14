from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import create_model
import typing


def json_schema_to_python_types(schema):
    type_map = {
        "any": Any,
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(schema, Any)


def list_to_enum(enum_name, values):
    return Enum(enum_name, {value: value for value in values})


def resolve_ref(ref: str, schema: dict, definitions: dict) -> dict:
    """Resolve a $ref reference to its definition."""
    if ref.startswith("#/$defs/"):
        def_name = ref.replace("#/$defs/", "")
        return definitions.get(def_name, {})
    return {}


def convert_mcp_input_schema_to_pydantic_model(
        dictionary: dict[str, Any],
        model_name: str = "CustomModel",
        docs: dict[str, str] = None,
        docs_function: dict[str, str] = None,
        definitions: dict[str, Any] = None,
        created_models: dict[str, Any] = None,
) -> type[Any]:
    """
    Convert a JSON schema to a Pydantic model, handling $defs and $ref.

    Args:
        dictionary: The JSON schema dictionary
        model_name: Name for the generated model
        docs: Documentation for fields
        docs_function: Function documentation
        definitions: Dictionary of schema definitions
        created_models: Dictionary of already created models to avoid duplicates

    Returns:
        A Pydantic model class
    """
    if docs is None:
        docs = {}
    if docs_function is None:
        docs_function = {}
    if definitions is None:
        # Extract all definitions from $defs
        definitions = dictionary.get("$defs", {})
    if created_models is None:
        created_models = {}

    # Check if we've already created this model
    if model_name in created_models:
        return created_models[model_name]

    fields: dict[str, Any] = {}

    # Process properties
    for field_name, field_data in dictionary.get("properties", {}).items():
        # Handle $ref references
        if "$ref" in field_data:
            ref_path = field_data["$ref"]
            ref_data = resolve_ref(ref_path, dictionary, definitions)

            # If this is a reference to an enum
            if "enum" in ref_data:
                enum_values = ref_data.get("enum", [])
                enum_name = ref_data.get("title", field_name)
                fields[field_name] = (list_to_enum(enum_name, enum_values), ...)

                # Add description if available
                if field_data.get("description"):
                    docs[field_name] = field_data["description"]
                elif ref_data.get("description"):
                    docs[field_name] = ref_data["description"]

            # If this is a reference to an object
            elif ref_data.get("type") == "object":
                # Create a submodel for the referenced object
                submodel_name = ref_data.get("title", f"{model_name}_{field_name}")
                submodel = convert_mcp_input_schema_to_pydantic_model(
                    ref_data,
                    submodel_name,
                    docs,
                    docs_function,
                    definitions,
                    created_models
                )
                fields[field_name] = (submodel, ...)

                # Add description if available
                if field_data.get("description"):
                    docs[field_name] = field_data["description"]
                elif ref_data.get("description"):
                    docs[field_name] = ref_data["description"]

            continue

        # Handle direct field types
        field_type = field_data.get("type", "string")

        if field_data.get("description", None):
            docs[field_name] = field_data["description"]

        if field_data.get("enum", []):
            fields[field_name] = (
                list_to_enum(field_name, field_data.get("enum", [])),
                ...,
            )
        elif field_type == "array":
            items = field_data.get("items", {})
            if items != {}:
                # Handle array items that might be references
                if "$ref" in items:
                    ref_path = items["$ref"]
                    ref_data = resolve_ref(ref_path, dictionary, definitions)

                    if "enum" in ref_data:
                        enum_values = ref_data.get("enum", [])
                        enum_name = ref_data.get("title", f"{field_name}_item")
                        enum_type = list_to_enum(enum_name, enum_values)
                        fields[field_name] = (List[enum_type], ...)
                    else:
                        ref_model_name = ref_data.get("title", f"{model_name}_{field_name}_item")
                        ref_model = convert_mcp_input_schema_to_pydantic_model(
                            ref_data,
                            ref_model_name,
                            docs,
                            docs_function,
                            definitions,
                            created_models
                        )
                        fields[field_name] = (List[ref_model], ...)
                else:
                    array = {"properties": items}
                    array_type = convert_mcp_input_schema_to_pydantic_model(
                        array,
                        f"{model_name}_{field_name}_items",
                        docs,
                        docs_function,
                        definitions,
                        created_models
                    )
                    fields[field_name] = (List[array_type], ...)
            else:
                fields[field_name] = (list, ...)
        elif field_type == "object":
            submodel = convert_mcp_input_schema_to_pydantic_model(
                field_data,
                f"{model_name}_{field_name}",
                docs,
                docs_function,
                definitions,
                created_models
            )
            fields[field_name] = (submodel, ...)
        else:
            python_type = json_schema_to_python_types(field_type)
            fields[field_name] = (python_type, ...)

    # Process function
    if "function" in dictionary:
        for field_name, field_data in dictionary.get("function", {}).items():
            if field_name == "name":
                model_name = field_data
            elif field_name == "description":
                docs_function["__doc__"] = field_data
            elif field_name == "parameters":
                return convert_mcp_input_schema_to_pydantic_model(
                    field_data,
                    f"{model_name}",
                    docs,
                    docs_function,
                    definitions,
                    created_models
                )

    # Process parameters
    if "parameters" in dictionary:
        field_data = {"function": dictionary}
        return convert_mcp_input_schema_to_pydantic_model(
            field_data,
            f"{model_name}",
            docs,
            docs_function,
            definitions,
            created_models
        )

    # Handle required fields
    if "required" in dictionary:
        required = dictionary.get("required", [])
        for key, field in list(fields.items()):
            if key not in required:
                fields[key] = (Optional[field[0]], None)

    # Create the model
    custom_model = create_model(model_name, **fields)

    # Store the created model to avoid duplicates
    created_models[model_name] = custom_model

    # Add documentation
    if "__doc__" in docs_function:
        custom_model.__doc__ = docs_function["__doc__"]
    for field_name, doc in docs.items():
        if field_name in custom_model.model_fields:
            custom_model.model_fields[field_name].description = doc

    return custom_model


def convert_mcp_input_json_schema(schema: dict) -> type[Any]:
    """
    Main entry point to convert a JSON schema with $defs to a Pydantic model.

    Args:
        schema: The JSON schema dictionary

    Returns:
        A Pydantic model class
    """
    # Extract title for the root model
    model_name = schema.get("title", "RootModel")

    # Process all definitions first to make them available for references
    definitions = schema.get("$defs", {})
    created_models = {}

    # Create the root model
    return convert_mcp_input_schema_to_pydantic_model(
        schema,
        model_name,
        definitions=definitions,
        created_models=created_models
    )

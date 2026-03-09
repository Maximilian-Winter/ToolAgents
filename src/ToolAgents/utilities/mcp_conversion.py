from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set, Literal, TypeVar, Generic, get_type_hints, get_origin, \
    get_args
from pydantic import create_model, Field, validator, root_validator
import typing
import re


def json_schema_to_python_types(schema_type, format_type=None):
    """Convert JSON schema types to Python types with format support."""
    type_map = {
        "any": Any,
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": None,
    }

    # Handle format-specific types
    if schema_type == "string" and format_type:
        format_map = {
            "date": str,  # Could use datetime.date if needed
            "date-time": str,  # Could use datetime.datetime if needed
            "time": str,  # Could use datetime.time if needed
            "email": str,
            "ipv4": str,
            "ipv6": str,
            "uuid": str,
            "uri": str,
            "hostname": str,
            "binary": bytes,
        }
        return format_map.get(format_type, str)

    return type_map.get(schema_type, Any)


def list_to_enum(enum_name, values, descriptions=None):
    """Create an Enum from a list of values with optional descriptions."""
    if descriptions and len(descriptions) == len(values):
        # Create an Enum with docstrings for each value
        enum_dict = {}
        for i, value in enumerate(values):
            member = value if isinstance(value, str) else f"VALUE_{i}"
            enum_dict[member] = value

        enum_class = Enum(enum_name, enum_dict)

        # Add docstrings to enum values
        for i, value in enumerate(values):
            member = value if isinstance(value, str) else f"VALUE_{i}"
            if hasattr(enum_class, member):
                enum_class[member].__doc__ = descriptions[i]

        return enum_class
    else:
        # Simple Enum creation
        if all(isinstance(v, str) for v in values):
            return Enum(enum_name, {value: value for value in values})
        else:
            # Handle non-string enum values
            return Enum(enum_name, {f"VALUE_{i}": value for i, value in enumerate(values)})


def resolve_ref(ref: str, schema: dict, definitions: dict) -> dict:
    """
    Resolve a $ref reference to its definition.

    Args:
        ref: Reference string (e.g., "#/$defs/MyType")
        schema: The full schema
        definitions: Dictionary of definitions

    Returns:
        The resolved schema definition
    """
    if not ref.startswith("#/"):
        # External references not supported yet
        return {}

    # Handle both $defs and definitions for compatibility
    path_parts = ref.lstrip("#/").split("/")

    current = schema
    for part in path_parts:
        if part == "$defs" and "$defs" in current:
            current = current["$defs"]
        elif part == "definitions" and "definitions" in current:
            current = current["definitions"]
        elif part in current:
            current = current[part]
        else:
            # Check in our extracted definitions
            if part in definitions:
                return definitions[part]
            return {}

    return current


def get_sub_schema_name(model_name, field_name=None, sub_index=None):
    """Generate a unique name for a sub-schema."""
    if field_name:
        name = f"{model_name}_{field_name}"
    else:
        name = model_name

    if sub_index is not None:
        name = f"{name}_Option{sub_index}"

    return name


def convert_schema_to_pydantic_model(
        dictionary: dict[str, Any],
        model_name: str = "CustomModel",
        docs: dict[str, str] = None,
        docs_function: dict[str, str] = None,
        definitions: dict[str, Any] = None,
        created_models: dict[str, Any] = None,
) -> type[Any]:
    """
    Convert a JSON schema to a Pydantic model, handling complex schema features.

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
        # Extract all definitions from both $defs and definitions
        definitions = {**dictionary.get("$defs", {}), **dictionary.get("definitions", {})}
    if created_models is None:
        created_models = {}

    # Check if we've already created this model
    if model_name in created_models:
        return created_models[model_name]

    # Handle schema combinations (allOf, anyOf, oneOf)
    if "allOf" in dictionary:
        # Merge all schemas in allOf
        merged_schema = {}
        for sub_schema in dictionary["allOf"]:
            if "$ref" in sub_schema:
                ref_schema = resolve_ref(sub_schema["$ref"], dictionary, definitions)
                sub_schema = ref_schema

            # Merge properties
            if "properties" in sub_schema:
                merged_schema.setdefault("properties", {}).update(sub_schema["properties"])

            # Merge required fields
            if "required" in sub_schema:
                merged_schema.setdefault("required", []).extend(sub_schema["required"])

            # Copy other attributes
            for key, value in sub_schema.items():
                if key not in ["properties", "required", "$ref"]:
                    merged_schema[key] = value

        # Process the merged schema
        return convert_schema_to_pydantic_model(
            merged_schema,
            model_name,
            docs,
            docs_function,
            definitions,
            created_models
        )

    elif "oneOf" in dictionary or "anyOf" in dictionary:
        # For oneOf/anyOf, create a Union type of all possible schemas
        schemas = dictionary.get("oneOf", dictionary.get("anyOf", []))
        union_types = []

        for i, sub_schema in enumerate(schemas):
            if "$ref" in sub_schema:
                ref_schema = resolve_ref(sub_schema["$ref"], dictionary, definitions)
                sub_schema = ref_schema
                sub_name = ref_schema.get("title", f"{model_name}_Option{i}")
            else:
                sub_name = f"{model_name}_Option{i}"

            sub_model = convert_schema_to_pydantic_model(
                sub_schema,
                sub_name,
                docs,
                docs_function,
                definitions,
                created_models
            )
            union_types.append(sub_model)

        # Create a simple wrapper model with a single field of Union type
        fields = {"value": (Union[tuple(union_types)], ...)}
        custom_model = create_model(model_name, **fields)
        created_models[model_name] = custom_model
        return custom_model

    fields: dict[str, Any] = {}
    validators = {}
    field_validators = {}

    # Process schema description
    if "description" in dictionary:
        docs_function["__doc__"] = dictionary["description"]

    # Process properties
    for field_name, field_data in dictionary.get("properties", {}).items():
        field_type = None
        field_default = ...  # Ellipsis means required

        # Handle $ref references
        if "$ref" in field_data:
            ref_path = field_data["$ref"]
            ref_data = resolve_ref(ref_path, dictionary, definitions)

            # If this is a reference to an enum
            if "enum" in ref_data:
                enum_values = ref_data.get("enum", [])
                enum_descriptions = ref_data.get("enumDescriptions", [])
                enum_name = ref_data.get("title", field_name)
                field_type = list_to_enum(enum_name, enum_values, enum_descriptions)

            # If this is a reference to an object
            elif ref_data.get("type") == "object" or "properties" in ref_data:
                # Create a submodel for the referenced object
                submodel_name = ref_data.get("title", f"{model_name}_{field_name}")
                field_type = convert_schema_to_pydantic_model(
                    ref_data,
                    submodel_name,
                    docs,
                    docs_function,
                    definitions,
                    created_models
                )
            # Other reference types
            else:
                ref_type = ref_data.get("type")
                if ref_type:
                    field_type = json_schema_to_python_types(ref_type, ref_data.get("format"))
                else:
                    # Handle enum directly in reference
                    if "enum" in ref_data:
                        enum_values = ref_data.get("enum", [])
                        enum_name = ref_data.get("title", field_name)
                        field_type = list_to_enum(enum_name, enum_values)
                    else:
                        # Default to Any if type can't be determined
                        field_type = Any

            # Add description if available
            if field_data.get("description"):
                docs[field_name] = field_data["description"]
            elif ref_data.get("description"):
                docs[field_name] = ref_data["description"]

            # Set default value if available
            if "default" in field_data:
                field_default = field_data["default"]
            elif "default" in ref_data:
                field_default = ref_data["default"]

        # Handle direct field types
        else:
            field_type_str = field_data.get("type", "string")

            # Store description
            if field_data.get("description"):
                docs[field_name] = field_data["description"]

            # Handle enums
            if "enum" in field_data:
                enum_values = field_data.get("enum", [])
                enum_descriptions = field_data.get("enumDescriptions", [])
                field_type = list_to_enum(field_name, enum_values, enum_descriptions)

            # Handle arrays
            elif field_type_str == "array":
                items = field_data.get("items", {})

                if items:
                    # Handle array items that might be references
                    if "$ref" in items:
                        ref_path = items["$ref"]
                        ref_data = resolve_ref(ref_path, dictionary, definitions)

                        if "enum" in ref_data:
                            enum_values = ref_data.get("enum", [])
                            enum_name = ref_data.get("title", f"{field_name}_item")
                            enum_type = list_to_enum(enum_name, enum_values)
                            field_type = List[enum_type]
                        else:
                            ref_model_name = ref_data.get("title", f"{model_name}_{field_name}_item")
                            ref_model = convert_schema_to_pydantic_model(
                                ref_data,
                                ref_model_name,
                                docs,
                                docs_function,
                                definitions,
                                created_models
                            )
                            field_type = List[ref_model]
                    # Complex array item definition
                    elif "type" in items or "properties" in items:
                        array_item_type = items.get("type", "object")

                        if array_item_type == "object" or "properties" in items:
                            array_model = convert_schema_to_pydantic_model(
                                items,
                                f"{model_name}_{field_name}_item",
                                docs,
                                docs_function,
                                definitions,
                                created_models
                            )
                            field_type = List[array_model]
                        else:
                            item_python_type = json_schema_to_python_types(array_item_type, items.get("format"))
                            field_type = List[item_python_type]
                    else:
                        # Default array of Any
                        field_type = List[Any]
                else:
                    # Default array of Any
                    field_type = List[Any]

                # Add array validators if needed
                if "uniqueItems" in field_data and field_data["uniqueItems"]:
                    # TODO: Add validator for unique items
                    pass

                if "minItems" in field_data or "maxItems" in field_data:
                    # TODO: Add validator for min/max items
                    pass

            # Handle objects
            elif field_type_str == "object" or "properties" in field_data:
                submodel = convert_schema_to_pydantic_model(
                    field_data,
                    f"{model_name}_{field_name}",
                    docs,
                    docs_function,
                    definitions,
                    created_models
                )
                field_type = submodel

            # Handle combinations (allOf, anyOf, oneOf) within a property
            elif any(k in field_data for k in ["allOf", "anyOf", "oneOf"]):
                submodel = convert_schema_to_pydantic_model(
                    field_data,
                    f"{model_name}_{field_name}",
                    docs,
                    docs_function,
                    definitions,
                    created_models
                )
                field_type = submodel

            # Handle primitive types with format
            else:
                format_type = field_data.get("format")
                field_type = json_schema_to_python_types(field_type_str, format_type)

            # Handle default values
            if "default" in field_data:
                field_default = field_data["default"]

        # Handle pattern validation
        if "pattern" in field_data:
            # TODO: Add regex validator
            pass

        # Add the field to our model
        if field_type is not None:
            fields[field_name] = (field_type, field_default)

    # Process function
    if "function" in dictionary:
        for field_name, field_data in dictionary.get("function", {}).items():
            if field_name == "name":
                model_name = field_data
            elif field_name == "description":
                docs_function["__doc__"] = field_data
            elif field_name == "parameters":
                return convert_schema_to_pydantic_model(
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
        return convert_schema_to_pydantic_model(
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
                field_type, _ = field
                fields[key] = (Optional[field_type], None)

    # Create the model
    model_config = {}

    if "additionalProperties" in dictionary:
        # Handle additionalProperties
        if dictionary["additionalProperties"] is True:
            # Allow any additional properties
            model_config["extra"] = "allow"
        elif dictionary["additionalProperties"] is False:
            # Forbid additional properties
            model_config["extra"] = "forbid"
        elif isinstance(dictionary["additionalProperties"], dict):
            # Type-constrained additional properties
            # TODO: Implement proper handling of typed additionalProperties
            model_config["extra"] = "allow"

    # Create the model with config if needed
    if model_config:
        custom_model = create_model(
            model_name,
            **fields
        )
    else:
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


def convert_json_schema(schema: dict) -> type[Any]:
    """
    Main entry point to convert a JSON schema to a Pydantic model.

    Args:
        schema: The JSON schema dictionary

    Returns:
        A Pydantic model class
    """
    # Extract title for the root model
    model_name = schema.get("title", "RootModel")

    # Process all definitions
    definitions = {**schema.get("$defs", {}), **schema.get("definitions", {})}
    created_models = {}

    # Create the root model
    return convert_schema_to_pydantic_model(
        schema,
        model_name,
        definitions=definitions,
        created_models=created_models
    )


# Usage example
if __name__ == "__main__":
    # Example schema with $defs and $ref
    example_schema = {
        "$defs": {
            "Calculation": {
                "description": "Represents a math operation on two numbers.",
                "properties": {
                    "number_one": {
                        "description": "First number.",
                        "title": "Number One",
                        "type": "number"
                    },
                    "operation": {
                        "$ref": "#/$defs/MathOperation",
                        "description": "Math operation to perform."
                    },
                    "number_two": {
                        "description": "Second number.",
                        "title": "Number Two",
                        "type": "number"
                    }
                },
                "required": [
                    "number_one",
                    "operation",
                    "number_two"
                ],
                "title": "Calculation",
                "type": "object"
            },
            "MathOperation": {
                "enum": [
                    "add",
                    "subtract",
                    "multiply",
                    "divide"
                ],
                "title": "MathOperation",
                "type": "string"
            }
        },
        "properties": {
            "calculation": {
                "$ref": "#/$defs/Calculation"
            }
        },
        "required": [
            "calculation"
        ],
        "title": "do_calculationArguments",
        "type": "object"
    }

    # Convert the schema to a Pydantic model
    CalculationModel = convert_json_schema(example_schema)

    # Print model details
    print(f"Model name: {CalculationModel.__name__}")
    print(f"Model fields: {CalculationModel.model_fields}")

    # Create an instance to test
    calc_instance = CalculationModel(
        calculation={
            "number_one": 10.5,
            "operation": "add",
            "number_two": 5.2
        }
    )
    print(f"Instance: {calc_instance}")

    # Example with more complex features
    complex_schema = {
        "title": "ComplexModel",
        "type": "object",
        "properties": {
            "simple_array": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True
            },
            "conditional_type": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"}
                ]
            },
            "email": {
                "type": "string",
                "format": "email"
            },
            "nested_object": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer", "default": 0}
                }
            }
        },
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zip": {"type": "string", "pattern": "^\\d{5}$"}
                }
            }
        }
    }

    # Convert the complex schema
    ComplexModel = convert_json_schema(complex_schema)
    print(f"\nComplex model name: {ComplexModel.__name__}")
    print(f"Complex model fields: {ComplexModel.model_fields}")

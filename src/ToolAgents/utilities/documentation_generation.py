import json
from enum import Enum
from inspect import isclass, getdoc
from types import UnionType, GenericAlias, NoneType
from typing import get_args, get_origin, Union, Any

from pydantic_core import PydanticUndefined

import inspect

from pydantic import BaseModel

from typing import get_args, get_origin, Union, Any, Generic
from types import GenericAlias


def generate_type_definitions(types: list[type]):
    output = []
    processed_types = set()

    def process_type(field_type):
        if field_type in processed_types:
            return
        processed_types.add(field_type)

        if isinstance(field_type, GenericAlias) or (inspect.isclass(field_type) and issubclass(field_type, Generic)):
            # Handle generic types
            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin and issubclass(origin, BaseModel):
                class_name = f"{origin.__name__}[{', '.join(arg.__name__ for arg in args)}]"
                extra_definitions, out = generate_class_definition(origin, class_name, args)
                for definition in extra_definitions:
                    if definition not in output:
                        output.append(definition.strip())
                if out not in output:
                    output.append(out.strip())

            # Process the type arguments recursively
            for arg in args:
                process_type(arg)

        elif isclass(field_type) and issubclass(field_type, BaseModel):
            extra_definitions, out = generate_class_definition(field_type, field_type.__name__)
            for definition in extra_definitions:
                if definition not in output:
                    output.append(definition.strip())
            if out not in output:
                output.append(out.strip())
        elif issubclass(field_type, Enum):
            enum_def = generate_enum_definition(field_type).strip()
            if enum_def not in output:
                output.append(enum_def)
        elif get_origin(field_type) == Union or isinstance(field_type, UnionType):
            for typ in get_args(field_type):
                process_type(typ)

    for field_type in types:
        process_type(field_type)

    return output


def generate_class_definition(model: type[BaseModel], class_name: str, type_params: list[type] = None) -> tuple[
    list[str], str]:
    definitions = []
    class_def = f"class {class_name}"
    if type_params:
        class_def += f"[{', '.join(param.__name__ for param in type_params)}]"
    class_def += ":\n"

    class_doc = getdoc(model)
    base_class_doc = getdoc(BaseModel)
    class_description = class_doc if class_doc and class_doc != base_class_doc else ""
    class_def += f'    """\n    {class_description}\n    Attributes:\n'

    for name, field_type in model.__annotations__.items():
        field_info = model.model_fields.get(name)
        field_description = field_info.description if field_info and field_info.description else ""
        type_annotation = get_type_annotation(field_type)
        if field_info.default is not PydanticUndefined and field_info.default is not None:
            continue
        class_def += f'        {name} ({type_annotation}): {field_description}\n'

        if isinstance(field_type, GenericAlias) or (inspect.isclass(field_type) and issubclass(field_type, Generic)):
            origin = get_origin(field_type)
            args = get_args(field_type)
            if origin and issubclass(origin, BaseModel):
                definitions.append(generate_class_definition(origin, origin.__name__, args)[1].strip())
            for arg in args:
                if isclass(arg) and issubclass(arg, BaseModel):
                    definitions.append(generate_class_definition(arg, arg.__name__)[1].strip())
                elif issubclass(arg, Enum):
                    definitions.append(generate_enum_definition(arg).strip())
        elif isclass(field_type) and issubclass(field_type, Enum):
            definitions.append(generate_enum_definition(field_type).strip())
        elif isclass(field_type) and issubclass(field_type, BaseModel):
            definitions.append(generate_class_definition(field_type, field_type.__name__)[1].strip())

    class_def += '    """\n'
    class_def += '    # Implementation omitted for brevity\n'
    class_def += '    pass\n\n'

    return definitions, class_def


def generate_function_definition(
        model: type[BaseModel],
        function_name: str,
        description: str = "",
) -> str:
    """
    Generate a Python function definition with class definitions for BaseModel parameters,
    using the docstring from the model's run method if available.

    Args:
        model (type[BaseModel]): The Pydantic model class.
        function_name (str): The name of the function.
        description (str, optional): A description of the function. Defaults to "".

    Returns:
        str: A string representation of the function definition with class definitions for BaseModel parameters.
    """
    output = ""
    parameters = []
    for name, field_type in model.__annotations__.items():
        parameters.append(f"{name}: {get_type_annotation(field_type)}")

    function_def = f"def {function_name}({', '.join(parameters)}):\n"

    # Check if the model has a run method and if it has a docstring
    run_method = getattr(model, 'run', None)
    if run_method and inspect.getdoc(run_method):
        function_def += f'    """\n{inspect.getdoc(run_method)}\n    """\n'
    else:
        function_def += f'    """\n'
        function_def += f'    {description}\n' if description else ''
        function_def += f'    Args:\n'

        for name, field_type in model.__annotations__.items():
            field_info = model.model_fields.get(name)
            field_description = field_info.description if field_info and field_info.description else "No description provided."
            function_def += f'        {name} ({get_type_annotation(field_type)}): {field_description}\n'
        function_def += (f'    \n'
                         f'    """\n'
                         #f'    # Implementation left out for brevity\n'
                         #f'    pass\n'
                         )

    return function_def


def generate_enum_definition(enum_class: type[Enum]) -> str:
    """
    Generate a string representation of an Enum class definition.

    Args:
        enum_class (type[Enum]): The Enum class to define.

    Returns:
        str: A string representation of the Enum class definition.
    """
    enum_def = f"class {enum_class.__name__}(Enum):\n"
    for member in enum_class:
        enum_def += f"    {member.name} = {member.value!r}\n"
    enum_def += "\n"
    return enum_def


def get_type_annotation(field_type: type[Any]) -> str:
    """
    Get a string representation of the type annotation for a field.

    Args:
        field_type (type[Any]): The type of the field.

    Returns:
        str: A string representation of the type annotation.
    """
    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        return f"list[{get_type_annotation(element_type)}]"
    elif get_origin(field_type) == Union or isinstance(field_type, UnionType):
        element_types = get_args(field_type)
        if len(element_types) == 2:
            if isclass(element_types[1]) and issubclass(NoneType, element_types[1]):
                return f"Optional[{get_type_annotation(element_types[0])}]"

        types = [get_type_annotation(t) for t in element_types]
        return f"Union[{', '.join(types)}]"
    elif isinstance(field_type, GenericAlias):
        if field_type.__origin__ == dict:
            key_type, value_type = get_args(field_type)
            return f"dict[{get_type_annotation(key_type)}, {get_type_annotation(value_type)}]"
    elif isclass(field_type) and issubclass(field_type, Enum):
        return field_type.__name__
    elif isclass(field_type) and issubclass(field_type, BaseModel):
        return field_type.__name__
    else:
        return field_type.__name__


def generate_markdown_documentation(
        pydantic_models: list[type[BaseModel]],
        model_prefix="Model",
        fields_prefix="Fields",
        documentation_with_field_description=True,
        ordered_json_mode=False,
) -> str:
    """
    Generate markdown documentation for a list of Pydantic models.

    Args:
        pydantic_models (list[type[BaseModel]]): list of Pydantic model classes.
        model_prefix (str): Prefix for the model section.
        fields_prefix (str): Prefix for the fields section.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation.
    """
    documentation = ""
    pyd_models = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"### {model_prefix} '{model.__name__}'\n"
        else:
            documentation += f"### '{model.__name__}'\n"

        # Handling multi-line model description with proper indentation

        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = (
            class_doc if class_doc and class_doc != base_class_doc else ""
        )
        if class_description != "":
            documentation += format_multiline_description(class_description, 0) + "\n"

        if add_prefix:
            # Indenting the fields section
            documentation += f"#### {fields_prefix}\n"
        else:
            documentation += f"#### Fields\n"
        if isclass(model) and issubclass(model, BaseModel):
            count = 1
            for name, field_type in model.__annotations__.items():
                # if name == "markdown_code_block":
                #    continue
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                if get_origin(field_type) == Union:
                    element_types = get_args(field_type)
                    for element_type in element_types:
                        if isclass(element_type) and issubclass(
                                element_type, BaseModel
                        ):
                            pyd_models.append((element_type, False))
                documentation += generate_field_markdown(
                    name
                    if not ordered_json_mode
                    else "{:03}".format(count) + "_" + name,
                    field_type,
                    model,
                    documentation_with_field_description=documentation_with_field_description,
                )
                count += 1
            if add_prefix:
                if documentation.endswith(f"#### {fields_prefix}\n"):
                    documentation += "none\n"
            else:
                if documentation.endswith("#### Fields\n"):
                    documentation += "none\n"
            documentation += "\n"

        if (
                hasattr(model, "Config")
                and hasattr(model.Config, "json_schema_extra")
                and "example" in model.Config.json_schema_extra
        ):
            documentation += f"  Expected Example Output for {model.__name__}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"

    return documentation


def generate_field_markdown(
        field_name: str,
        field_type: type[Any],
        model: type[BaseModel],
        depth=1,
        documentation_with_field_description=True,
) -> str:
    """
    Generate markdown documentation for a Pydantic model field.

    Args:
        field_name (str): Name of the field.
        field_type (type[Any]): Type of the field.
        model (type[BaseModel]): Pydantic model class.
        depth (int): Indentation depth in the documentation.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation for the field.
    """
    indent = ""

    field_info = model.model_fields.get(field_name)
    field_description = (
        field_info.description if field_info and field_info.description else ""
    )
    is_enum = False
    enum_values = None
    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        field_text = (
            f"{indent}{field_name} ({field_type.__name__} of {element_type.__name__})"
        )
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"
    elif get_origin(field_type) == Union or isinstance(field_type, UnionType):
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            if element_type.__name__ == "NoneType":
                types.append("null")
            else:
                types.append(element_type.__name__)
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    elif issubclass(field_type, Enum):
        enum_values = [f"'{str(member.value)}'" for member in field_type]
        is_enum = True
        field_text = f"{indent}{field_name}"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    else:
        field_text = f"{indent}{field_name} ({field_type.__name__})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if is_enum:

        field_text = field_text.strip() + field_description.strip() + f" Can be one of the following values: {' or '.join(enum_values)}" + "\n"
    elif field_description != "":
        field_text += field_description + "\n"

    # Check for and include field-specific examples if available
    if (
            hasattr(model, "Config")
            and hasattr(model.Config, "json_schema_extra")
            and "example" in model.Config.json_schema_extra
    ):
        field_example = model.Config.json_schema_extra["example"].get(field_name)
        if field_example is not None:
            example_text = (
                f"'{field_example}'"
                if isinstance(field_example, str)
                else field_example
            )
            field_text += f"{indent}  Example: {example_text}\n"

    if isclass(field_type) and issubclass(field_type, BaseModel):
        field_text += f"{indent}  Details:\n"
        for name, type_ in field_type.__annotations__.items():
            field_text += generate_field_markdown(name, type_, field_type, depth + 2)

    return field_text


def format_json_example(example: dict[str, Any], depth: int) -> str:
    """
    Format a JSON example into a readable string with indentation.

    Args:
        example (dict): JSON example to be formatted.
        depth (int): Indentation depth.

    Returns:
        str: Formatted JSON example string.
    """
    indent = "    " * depth
    formatted_example = "{\n"
    for key, value in example.items():
        value_text = f"'{value}'" if isinstance(value, str) else value
        formatted_example += f"{indent}{key}: {value_text},\n"
    formatted_example = formatted_example.rstrip(",\n") + "\n" + indent + "}"
    return formatted_example


def generate_text_documentation(
        pydantic_models: list[BaseModel],
        model_prefix="Output Model",
        fields_prefix="Fields",
        documentation_with_field_description=True,
        ordered_json_mode=False,
) -> str:
    """
    Generate markdown documentation for a list of Pydantic models.

    Args:
        pydantic_models (list[type[BaseModel]]): list of Pydantic model classes.
        model_prefix (str): Prefix for the model section.
        fields_prefix (str): Prefix for the fields section.
        documentation_with_field_description (bool): Include field descriptions in the documentation.
        ordered_json_mode (bool): Add ordering prefix for JSON schemas
    Returns:
        str: Generated text documentation.
    """
    documentation = ""
    pyd_models = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"{model_prefix}: {model.__name__}\n"
        else:
            documentation += f"Class: {model.__name__}\n"

        # Handling multi-line model description with proper indentation

        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = (
            class_doc if class_doc and class_doc != base_class_doc else ""
        )
        if class_description != "":
            documentation += "  Description: "
            documentation += format_multiline_description(class_description, 2) + "\n"

        if add_prefix:
            # Indenting the fields section
            documentation += f"  {fields_prefix}:\n"
        else:
            documentation += f"  Attributes:\n"
        if isclass(model) and issubclass(model, BaseModel):
            count = 1
            for name, field_type in model.__annotations__.items():
                # if name == "markdown_code_block":
                #    continue
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                    if get_origin(element_type) == Union or isinstance(
                            element_type, UnionType
                    ):
                        element_types = get_args(element_type)
                        for element_type in element_types:
                            if isclass(element_type) and issubclass(
                                    element_type, BaseModel
                            ):
                                pyd_models.append((element_type, False))
                            if get_origin(element_type) == list:
                                element_type = get_args(element_type)[0]
                                if isclass(element_type) and issubclass(
                                        element_type, BaseModel
                                ):
                                    pyd_models.append((element_type, False))
                if get_origin(field_type) == Union or isinstance(field_type, UnionType):
                    element_types = get_args(field_type)
                    for element_type in element_types:
                        if isclass(element_type) and issubclass(
                                element_type, BaseModel
                        ):
                            pyd_models.append((element_type, False))
                        if get_origin(element_type) == list:
                            element_type = get_args(element_type)[0]
                            if isclass(element_type) and issubclass(
                                    element_type, BaseModel
                            ):
                                pyd_models.append((element_type, False))
                if isclass(field_type) and issubclass(field_type, BaseModel):
                    pyd_models.append((field_type, False))
                documentation += generate_field_text(
                    name if not ordered_json_mode
                    else "{:03}".format(count) + "_" + name,
                    name,
                    field_type,
                    model,
                    documentation_with_field_description=documentation_with_field_description,
                )
                count += 1
            if add_prefix:
                if documentation.endswith(f"{fields_prefix}:\n"):
                    documentation += "    none\n"
            else:
                if documentation.endswith("fields:\n"):
                    documentation += "    none\n"
            documentation += "\n"

        if (
                hasattr(model, "Config")
                and hasattr(model.Config, "json_schema_extra")
                and "example" in model.Config.json_schema_extra
        ):
            documentation += f"  Expected Example Output for {model.__name__}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"

    return documentation


def generate_field_text(
        field_name: str,
        field_real_name: str,
        field_type: type[Any],
        model: type[BaseModel],
        depth=1,
        documentation_with_field_description=True,
) -> str:
    """
    Generate markdown documentation for a Pydantic model field.

    Args:
        field_name (str): Output Name of the field.
        field_real_name (str): Real Name of the field.:
        field_type (type[Any]): Type of the field.

        model (type[BaseModel]): Pydantic model class.
        depth (int): Indentation depth in the documentation.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation for the field.

    """
    indent = "    " * depth

    field_info = model.model_fields.get(field_real_name)
    field_description = (
        field_info.description if field_info and field_info.description else ""
    )
    field_text = ""
    is_enum = False
    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        if get_origin(element_type) == Union or isinstance(element_type, UnionType):
            element_types = get_args(element_type)
            types = []
            for element_type in element_types:
                if element_type.__name__ == "NoneType":
                    types.append("null")
                else:
                    if isclass(element_type) and issubclass(element_type, Enum):
                        enum_values = [
                            f"'{str(member.value)}'" for member in element_type
                        ]
                        for enum_value in enum_values:
                            types.append(enum_value)

                    else:
                        if get_origin(element_type) == list:
                            element_type = get_args(element_type)[0]
                            types.append(f"(list of {element_type.__name__})")
                        else:
                            types.append(element_type.__name__)
            field_text = f"({' or '.join(types)})"
            field_text = f"{indent}{field_name} ({field_type.__name__} of {field_text})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
        else:
            field_text = f"{indent}{field_name} ({field_type.__name__} of {element_type.__name__})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
    elif get_origin(field_type) == Union:
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            if element_type.__name__ == "NoneType":
                types.append("null")
            else:
                if isclass(element_type) and issubclass(element_type, Enum):
                    enum_values = [f"'{str(member.value)}'" for member in element_type]
                    for enum_value in enum_values:
                        types.append(enum_value)

                else:
                    if get_origin(element_type) == list:
                        element_type = get_args(element_type)[0]
                        types.append(f"(list of {element_type.__name__})")
                    else:
                        types.append(element_type.__name__)
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"
    elif isinstance(field_type, GenericAlias):
        if field_type.__origin__ == dict:
            key_type, value_type = get_args(field_type)

            additional_key_type = key_type.__name__
            additional_value_type = value_type.__name__
            field_text = f"{indent}{field_name} (dict of {additional_key_type} to {additional_value_type})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
        elif field_type.__origin__ == list:
            element_type = get_args(field_type)[0]
            field_text = f"{indent}{field_name} (list of {element_type.__name__})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
    elif isclass(field_type) and issubclass(field_type, Enum):
        enum_values = [f"'{str(member.value)}'" for member in field_type]
        is_enum = True
        field_text = f"{indent}{field_name} (enum)"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    else:
        field_text = f"{indent}{field_name} ({field_type.__name__})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if is_enum:

        field_text = field_text + field_description.strip() + f" Can be one of the following values: {' or '.join(enum_values)}" + "\n"
    elif field_description != "":
        field_text += field_description + "\n"

    # Check for and include field-specific examples if available
    if (
            hasattr(model, "Config")
            and hasattr(model.Config, "json_schema_extra")
            and "example" in model.Config.json_schema_extra
    ):
        field_example = model.Config.json_schema_extra["example"].get(field_real_name)
        if field_example is not None:
            example_text = (
                f"'{field_example}'"
                if isinstance(field_example, str)
                else field_example
            )
            field_text += f"{indent}  Example: {example_text}\n"

    return field_text


def format_multiline_description(description: str, indent_level: int) -> str:
    """
    Format a multiline description with proper indentation.

    Args:
        description (str): Multiline description.
        indent_level (int): Indentation level.

    Returns:
        str: Formatted multiline description.
    """
    indent = "  " * indent_level
    return description.replace("\n", "\n" + indent)

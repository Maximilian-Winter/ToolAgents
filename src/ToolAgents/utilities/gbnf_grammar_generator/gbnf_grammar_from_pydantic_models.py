from __future__ import annotations

import inspect
import re
from copy import copy
from enum import Enum
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
)

from docstring_parser import parse
from pydantic import BaseModel, create_model

from ToolAgents.utilities.llm_documentation import (
    generate_markdown_documentation,
    generate_text_documentation,
)
from ToolAgents.utilities.pydantic_utilites import (
    create_dynamic_models_from_dictionaries,
)

if TYPE_CHECKING:
    from types import GenericAlias
    from types import UnionType
else:
    # python 3.8 compat
    from typing import _GenericAlias as GenericAlias
    from types import UnionType


class PydanticDataType(Enum):
    """
    Defines the data types supported by the grammar_generator.

    Attributes:
        STRING (str): Represents a string data type.
        BOOLEAN (str): Represents a boolean data type.
        INTEGER (str): Represents an integer data type.
        FLOAT (str): Represents a float data type.
        OBJECT (str): Represents an object data type.
        ARRAY (str): Represents an array data type.
        ENUM (str): Represents an enum data type.
        CUSTOM_CLASS (str): Represents a custom class data type.
    """

    STRING = "string"
    TRIPLE_QUOTED_STRING = "triple_quoted_string"
    MARKDOWN_CODE_BLOCK = "markdown_code_block"
    BOOLEAN = "boolean"
    INTEGER = "number"
    FLOAT = "number"
    OBJECT = "object"
    ARRAY = "array"
    ENUM = "enum"
    ANY = "any"
    NULL = "null"
    CUSTOM_CLASS = "custom-class"
    CUSTOM_DICT = "custom-dict"
    SET = "set"


def map_pydantic_type_to_gbnf(pydantic_type: type[Any]) -> str:
    if isclass(pydantic_type) and issubclass(pydantic_type, str):
        return PydanticDataType.STRING.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, bool):
        return PydanticDataType.BOOLEAN.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, int):
        return PydanticDataType.INTEGER.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, float):
        return PydanticDataType.FLOAT.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, Enum):
        return PydanticDataType.ENUM.value

    elif isclass(pydantic_type) and issubclass(pydantic_type, BaseModel):
        return format_model_and_field_name(pydantic_type.__name__)
    elif get_origin(pydantic_type) is list:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-list"
    elif get_origin(pydantic_type) is set:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-set"
    elif get_origin(pydantic_type) is Union or isinstance(pydantic_type, UnionType):
        union_types = get_args(pydantic_type)
        union_rules = [map_pydantic_type_to_gbnf(ut) for ut in union_types]
        return f"union-{'-or-'.join(union_rules)}"
    elif get_origin(pydantic_type) is Optional:
        element_type = get_args(pydantic_type)[0]
        return f"optional-{map_pydantic_type_to_gbnf(element_type)}"
    elif isclass(pydantic_type):
        return f"{PydanticDataType.CUSTOM_CLASS.value}-{format_model_and_field_name(pydantic_type.__name__)}"
    elif get_origin(pydantic_type) is dict:
        key_type, value_type = get_args(pydantic_type)
        return f"custom-dict-key-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(key_type))}-value-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(value_type))}"
    else:
        return "unknown"


def format_model_and_field_name(model_name: str) -> str:
    parts = re.findall("[A-Z][^A-Z]*", model_name)
    if not parts:  # Check if the list is empty
        return model_name.lower().replace("_", "-")
    return "-".join(part.lower().replace("_", "-") for part in parts)


def generate_list_rule(element_type):
    """
    Generate a GBNF rule for a list of a given element type.

    :param element_type: The type of the elements in the list (e.g., 'string').
    :return: A string representing the GBNF rule for a list of the given type.
    """
    rule_name = f"{map_pydantic_type_to_gbnf(element_type)}-list"
    element_rule = map_pydantic_type_to_gbnf(element_type)
    list_rule = rf'{rule_name} ::= "["  {element_rule} (","  {element_rule})* "]"'
    return list_rule


def get_members_structure(cls, rule_name):
    if issubclass(cls, Enum):
        # Handle Enum types
        members = [
            f'"\\"{member.value}\\""' for name, member in cls.__members__.items()
        ]
        return f"{cls.__name__.lower()} ::= " + " | ".join(members)
    if cls.__annotations__ and cls.__annotations__ != {}:
        result = f'{rule_name} ::= "{{"'
        # Modify this comprehension
        members = [
            f'  "\\"{name}\\"" ":"  {map_pydantic_type_to_gbnf(param_type)}'
            for name, param_type in cls.__annotations__.items()
            if name != "self"
        ]

        result += '"," '.join(members)
        result += '  "}"'
        return result
    if rule_name == "custom-class-any":
        result = f"{rule_name} ::= "
        result += "value"
        return result

    init_signature = inspect.signature(cls.__init__)
    parameters = init_signature.parameters
    result = f'{rule_name} ::=  "{{"'
    # Modify this comprehension too
    members = [
        f'  "\\"{name}\\"" ":"  {map_pydantic_type_to_gbnf(param.annotation)}'
        for name, param in parameters.items()
        if name != "self" and param.annotation != inspect.Parameter.empty
    ]

    result += '", "'.join(members)
    result += '  "}"'
    return result


def regex_to_gbnf(regex_pattern: str) -> str:
    """
    Translate a basic regex pattern to a GBNF rule.
    Note: This function handles only a subset of simple regex patterns.
    """
    gbnf_rule = regex_pattern

    # Translate common regex components to GBNF
    gbnf_rule = gbnf_rule.replace("\\d", "[0-9]")
    gbnf_rule = gbnf_rule.replace("\\s", "[ \t\n]")

    # Handle quantifiers and other regex syntax that is similar in GBNF
    # (e.g., '*', '+', '?', character classes)

    return gbnf_rule


def generate_gbnf_integer_rules(max_digit=None, min_digit=None):
    """

    Generate GBNF Integer Rules

    Generates GBNF (Generalized Backus-Naur Form) rules for integers based on the given maximum and minimum digits.

    Parameters:
        max_digit (int): The maximum number of digits for the integer. Default is None.
        min_digit (int): The minimum number of digits for the integer. Default is None.

    Returns:
        integer_rule (str): The identifier for the integer rule generated.
        additional_rules (list): A list of additional rules generated based on the given maximum and minimum digits.

    """
    additional_rules = []

    # Define the rule identifier based on max_digit and min_digit
    integer_rule = "integer-part"
    if max_digit is not None:
        integer_rule += f"-max{max_digit}"
    if min_digit is not None:
        integer_rule += f"-min{min_digit}"

    # Handling Integer Rules
    if max_digit is not None or min_digit is not None:
        # Start with an empty rule part
        integer_rule_part = ""

        # Add mandatory digits as per min_digit
        if min_digit is not None:
            integer_rule_part += "[0-9] " * min_digit

        # Add optional digits up to max_digit
        if max_digit is not None:
            optional_digits = max_digit - (min_digit if min_digit is not None else 0)
            integer_rule_part += "".join(["[0-9]? " for _ in range(optional_digits)])

        # Trim the rule part and append it to additional rules
        integer_rule_part = integer_rule_part.strip()
        if integer_rule_part:
            additional_rules.append(f"{integer_rule} ::= {integer_rule_part}")

    return integer_rule, additional_rules


def generate_gbnf_float_rules(
    max_digit=None, min_digit=None, max_precision=None, min_precision=None
):
    """
    Generate GBNF float rules based on the given constraints.

    :param max_digit: Maximum number of digits in the integer part (default: None)
    :param min_digit: Minimum number of digits in the integer part (default: None)
    :param max_precision: Maximum number of digits in the fractional part (default: None)
    :param min_precision: Minimum number of digits in the fractional part (default: None)
    :return: A tuple containing the float rule and additional rules as a list

    Example Usage:
    max_digit = 3
    min_digit = 1
    max_precision = 2
    min_precision = 1
    generate_gbnf_float_rules(max_digit, min_digit, max_precision, min_precision)

    Output:
    ('float-3-1-2-1', ['integer-part-max3-min1 ::= [0-9] [0-9] [0-9]?', 'fractional-part-max2-min1 ::= [0-9] [0-9]?', 'float-3-1-2-1 ::= integer-part-max3-min1 "." fractional-part-max2-min
    *1'])

    Note:
    GBNF stands for Generalized Backus-Naur Form, which is a notation technique to specify the syntax of programming languages or other formal grammars.
    """
    additional_rules = []

    # Define the integer part rule
    integer_part_rule = (
        "integer-part"
        + (f"-max{max_digit}" if max_digit is not None else "")
        + (f"-min{min_digit}" if min_digit is not None else "")
    )

    # Define the fractional part rule based on precision constraints
    fractional_part_rule = "fractional-part"
    fractional_rule_part = ""
    if max_precision is not None or min_precision is not None:
        fractional_part_rule += (
            f"-max{max_precision}" if max_precision is not None else ""
        ) + (f"-min{min_precision}" if min_precision is not None else "")
        # Minimum number of digits
        fractional_rule_part = "[0-9]" * (
            min_precision if min_precision is not None else 1
        )
        # Optional additional digits
        fractional_rule_part += "".join(
            [" [0-9]?"]
            * (
                (max_precision - (min_precision if min_precision is not None else 1))
                if max_precision is not None
                else 0
            )
        )
        additional_rules.append(f"{fractional_part_rule} ::= {fractional_rule_part}")

    # Define the float rule
    float_rule = f"float-{max_digit if max_digit is not None else 'X'}-{min_digit if min_digit is not None else 'X'}-{max_precision if max_precision is not None else 'X'}-{min_precision if min_precision is not None else 'X'}"
    additional_rules.append(
        f'{float_rule} ::= {integer_part_rule} "." {fractional_part_rule}'
    )

    # Generating the integer part rule definition, if necessary
    if max_digit is not None or min_digit is not None:
        integer_rule_part = "[0-9]"
        if min_digit is not None and min_digit > 1:
            integer_rule_part += " [0-9]" * (min_digit - 1)
        if max_digit is not None:
            integer_rule_part += "".join(
                [" [0-9]?"] * (max_digit - (min_digit if min_digit is not None else 1))
            )
        additional_rules.append(f"{integer_part_rule} ::= {integer_rule_part.strip()}")

    return float_rule, additional_rules


def generate_gbnf_rule_for_type(
    model_name,
    field_name,
    field_type,
    is_optional,
    processed_models,
    created_rules,
    field_info=None,
) -> tuple[str, list[str]]:
    """
    Generate GBNF rule for a given field type.

    :param model_name: Name of the model.

    :param field_name: Name of the field.
    :param field_type: Type of the field.
    :param is_optional: Whether the field is optional.
    :param processed_models: List of processed models.
    :param created_rules: List of created rules.
    :param field_info: Additional information about the field (optional).

    :return: Tuple containing the GBNF type and a list of additional rules.
    :rtype: tuple[str, list]
    """
    rules = []

    field_name = format_model_and_field_name(field_name)
    gbnf_type = map_pydantic_type_to_gbnf(field_type)

    if isclass(field_type) and issubclass(field_type, BaseModel):
        nested_model_name = format_model_and_field_name(field_type.__name__)
        nested_model_rules, _ = generate_gbnf_grammar(
            field_type, processed_models, created_rules
        )
        rules.extend(nested_model_rules)
        gbnf_type, rules = nested_model_name, rules
    elif isclass(field_type) and issubclass(field_type, Enum):
        enum_values = [
            f'"\\"{e.value}\\""' for e in field_type
        ]  # Adding escaped quotes
        enum_rule = f"{model_name}-{field_name} ::= {' | '.join(enum_values)}"
        rules.append(enum_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules
    elif get_origin(field_type) == list:  # Array
        element_type = get_args(field_type)[0]
        element_rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name,
            f"{field_name}-element",
            element_type,
            is_optional,
            processed_models,
            created_rules,
        )
        rules.extend(additional_rules)
        array_rule = rf"""{model_name}-{field_name} ::= "[" ws ({element_rule_name})? ("," ws {element_rule_name})* ws "]" """
        rules.append(array_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules

    elif get_origin(field_type) == set or field_type == set:  # Array
        element_type = get_args(field_type)[0]
        element_rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name,
            f"{field_name}-element",
            element_type,
            is_optional,
            processed_models,
            created_rules,
        )
        rules.extend(additional_rules)
        array_rule = rf"""{model_name}-{field_name} ::= "[" ws ({element_rule_name})? ("," ws {element_rule_name})* ws "]" """
        rules.append(array_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules

    elif gbnf_type.startswith("custom-class-"):
        rules.append(get_members_structure(field_type, gbnf_type))
    elif gbnf_type.startswith("custom-dict-"):
        key_type, value_type = get_args(field_type)

        additional_key_type, additional_key_rules = generate_gbnf_rule_for_type(
            model_name,
            f"{field_name}-key-type",
            key_type,
            is_optional,
            processed_models,
            created_rules,
        )
        additional_value_type, additional_value_rules = generate_gbnf_rule_for_type(
            model_name,
            f"{field_name}-value-type",
            value_type,
            is_optional,
            processed_models,
            created_rules,
        )
        rules.extend(
            [
                rf'{gbnf_type} ::= "{{"  ( {additional_key_type} ": "  {additional_value_type} ("," "\n" ws {additional_key_type} ":"  {additional_value_type})*  )? "}}" '
            ]
        )
        rules.extend(additional_key_rules)
        rules.extend(additional_value_rules)
    elif gbnf_type.startswith("union-"):
        union_types = get_args(field_type)
        union_rules = []

        for union_type in union_types:
            if isinstance(union_type, GenericAlias):
                union_gbnf_type, union_rules_list = generate_gbnf_rule_for_type(
                    model_name,
                    field_name,
                    union_type,
                    False,
                    processed_models,
                    created_rules,
                )
                union_rules.append(union_gbnf_type)
                rules.extend(union_rules_list)

            elif not issubclass(union_type, type(None)):
                union_gbnf_type, union_rules_list = generate_gbnf_rule_for_type(
                    model_name,
                    field_name,
                    union_type,
                    False,
                    processed_models,
                    created_rules,
                )
                union_rules.append(union_gbnf_type)
                rules.extend(union_rules_list)

        # Defining the union grammar rule separately
        if len(union_rules) == 1:
            union_grammar_rule = f"{model_name}-{field_name}-optional ::= {' | '.join(union_rules)} | null"
        else:
            uni = []
            for rule in union_rules:
                if rule not in uni:
                    uni.append(rule)
            union_grammar_rule = (
                f"{model_name}-{field_name}-union ::= {' | '.join(uni)}"
            )
        rules.append(union_grammar_rule)
        if len(union_rules) == 1:
            gbnf_type = f"{model_name}-{field_name}-optional"
        else:
            gbnf_type = f"{model_name}-{field_name}-union"
    elif isclass(field_type) and issubclass(field_type, str):
        if (
            field_info
            and hasattr(field_info, "json_schema_extra")
            and field_info.json_schema_extra is not None
        ):
            triple_quoted_string = field_info.json_schema_extra.get(
                "triple_quoted_string", False
            )
            markdown_string = field_info.json_schema_extra.get(
                "markdown_code_block", False
            )

            gbnf_type = (
                PydanticDataType.TRIPLE_QUOTED_STRING.value
                if triple_quoted_string
                else PydanticDataType.STRING.value
            )
            gbnf_type = (
                PydanticDataType.MARKDOWN_CODE_BLOCK.value
                if markdown_string
                else gbnf_type
            )

        elif field_info and hasattr(field_info, "pattern"):
            # Convert regex pattern to grammar rule
            regex_pattern = field_info.regex.pattern
            gbnf_type = f"pattern-{field_name} ::= {regex_to_gbnf(regex_pattern)}"
        else:
            gbnf_type = PydanticDataType.STRING.value

    elif (
        isclass(field_type)
        and issubclass(field_type, float)
        and field_info
        and hasattr(field_info, "json_schema_extra")
        and field_info.json_schema_extra is not None
    ):
        # Retrieve precision attributes for floats
        max_precision = (
            field_info.json_schema_extra.get("max_precision")
            if field_info and hasattr(field_info, "json_schema_extra")
            else None
        )
        min_precision = (
            field_info.json_schema_extra.get("min_precision")
            if field_info and hasattr(field_info, "json_schema_extra")
            else None
        )
        max_digits = (
            field_info.json_schema_extra.get("max_digit")
            if field_info and hasattr(field_info, "json_schema_extra")
            else None
        )
        min_digits = (
            field_info.json_schema_extra.get("min_digit")
            if field_info and hasattr(field_info, "json_schema_extra")
            else None
        )

        # Generate GBNF rule for float with given attributes
        gbnf_type, rules = generate_gbnf_float_rules(
            max_digit=max_digits,
            min_digit=min_digits,
            max_precision=max_precision,
            min_precision=min_precision,
        )

    elif (
        isclass(field_type)
        and issubclass(field_type, int)
        and field_info
        and hasattr(field_info, "json_schema_extra")
        and field_info.json_schema_extra is not None
    ):
        # Retrieve digit attributes for integers
        max_digits = (
            field_info.json_schema_extra.get("max_digit")
            if field_info and hasattr(field_info, "json_schema_extra")
            else None
        )
        min_digits = (
            field_info.json_schema_extra.get("min_digit")
            if field_info and hasattr(field_info, "json_schema_extra")
            else None
        )

        # Generate GBNF rule for integer with given attributes
        gbnf_type, rules = generate_gbnf_integer_rules(
            max_digit=max_digits, min_digit=min_digits
        )
    else:
        gbnf_type, rules = gbnf_type, []

    return gbnf_type, rules


def generate_gbnf_grammar(
    model: type[BaseModel],
    processed_models: set[type[BaseModel]],
    created_rules: dict[str, list[str]],
) -> tuple[list[str], bool]:
    """

    Generate GBnF Grammar

    Generates a GBnF grammar for a given model.

    :param model: A Pydantic model class to generate the grammar for. Must be a subclass of BaseModel.
    :param processed_models: A set of already processed models to prevent infinite recursion.
    :param created_rules: A dict containing already created rules to prevent duplicates.
    :return: A list of GBnF grammar rules in string format. And two booleans indicating if an extra markdown or triple quoted string is in the grammar.
    Example Usage:
    ```
    model = MyModel
    processed_models = set()
    created_rules = dict()

    gbnf_grammar = generate_gbnf_grammar(model, processed_models, created_rules)
    ```
    """
    if model in processed_models:
        return [], False

    processed_models.add(model)
    model_name = format_model_and_field_name(model.__name__)

    if not issubclass(model, BaseModel):
        # For non-Pydantic classes, generate model_fields from __annotations__ or __init__
        if hasattr(model, "__annotations__") and model.__annotations__:
            model_fields = {
                name: (typ, ...) for name, typ in model.__annotations__.items()
            }
        else:
            init_signature = inspect.signature(model.__init__)
            parameters = init_signature.parameters
            model_fields = {
                name: (param.annotation, param.default)
                for name, param in parameters.items()
                if name != "self"
            }
    else:
        # For Pydantic models, use model_fields and check for ellipsis (required fields)
        model_fields = model.__annotations__

    model_rule_parts = []
    nested_rules = []
    has_markdown_code_block = False
    has_triple_quoted_string = False
    look_for_markdown_code_block = False
    look_for_triple_quoted_string = False
    for field_name, field_info in model_fields.items():
        if not issubclass(model, BaseModel):
            field_type, default_value = field_info
            # Check if the field is optional (not required)
            is_optional = (default_value is not inspect.Parameter.empty) and (
                default_value is not Ellipsis
            )
        else:
            field_type = field_info
            field_info = model.model_fields[field_name]
            is_optional = (
                field_info.is_required is False and get_origin(field_type) is Optional
            )
        rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name,
            format_model_and_field_name(field_name),
            field_type,
            is_optional,
            processed_models,
            created_rules,
            field_info,
        )
        look_for_markdown_code_block = (
            True if rule_name == "markdown_code_block" else False
        )
        look_for_triple_quoted_string = (
            True if rule_name == "triple_quoted_string" else False
        )
        if not look_for_markdown_code_block and not look_for_triple_quoted_string:
            if rule_name not in created_rules:
                created_rules[rule_name] = additional_rules
            model_rule_parts.append(
                f' ws "\\"{field_name}\\"" ": " {rule_name}'
            )  # Adding escaped quotes
            nested_rules.extend(additional_rules)
        else:
            has_triple_quoted_string = look_for_triple_quoted_string
            has_markdown_code_block = look_for_markdown_code_block

    fields_joined = r' "," '.join(model_rule_parts)
    if fields_joined != "":
        model_rule = rf'{model_name} ::= "{{" {fields_joined} ws "}}"'
    else:
        model_rule = rf'{model_name} ::= "{{" "}}"'

    has_special_string = False
    if has_triple_quoted_string:
        model_rule += '"\\n" ws "}"'
        model_rule += '"\\n" triple-quoted-string'
        has_special_string = True
    if has_markdown_code_block:
        model_rule += '"\\n" ws "}"'
        model_rule += '"\\n" markdown-code-block'
        has_special_string = True
    all_rules = [model_rule] + nested_rules

    return all_rules, has_special_string


def generate_gbnf_grammar_from_pydantic_models(
    models: list[BaseModel],
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    list_of_outputs: bool = False,
    add_inner_thoughts: bool = True,
    allow_only_inner_thoughts: bool = True,
    inner_thought_field_name: str = "chain_of_thought",
    add_request_heartbeat: bool = True,
    request_heartbeat_field_name: str = "request_heartbeat",
    request_heartbeat_models: list[str] = None,
) -> str:
    """
    Generate GBNF Grammar from Pydantic Models.

    This method takes a list of Pydantic models and uses them to generate a GBNF grammar string. The generated grammar string can be used for parsing and validating data using the generated
    * grammar.

    Args:
        models (list[type[BaseModel]]): A list of Pydantic models to generate the grammar from.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        list_of_outputs (str, optional): Allows a list of output objects
        add_inner_thoughts (bool, optional): Add inner thoughts to the grammar. Defaults to True.
        allow_only_inner_thoughts (bool, optional): Allow only inner thoughts. Defaults to True.
    Returns:
        str: The generated GBNF grammar string.

    Examples:
        models = [UserModel, PostModel]
        grammar = generate_gbnf_grammar_from_pydantic(models)
        print(grammar)
        # Output:
        # root ::= UserModel | PostModel
        # ...
    """
    if request_heartbeat_models is None:
        request_heartbeat_models = []
    processed_models: set[type[BaseModel]] = set()
    all_rules = []
    created_rules: dict[str, list[str]] = {}
    if outer_object_name is None:
        for model in models:
            model_rules, _ = generate_gbnf_grammar(
                model, processed_models, created_rules
            )
            all_rules.extend(model_rules)

        if list_of_outputs:
            root_rule = (
                r'root ::= (" "| "\n") "[" grammar-models ("," ws grammar-models)* "]"'
                + "\n"
            )
        else:
            root_rule = r'root ::= (" "| "\n") grammar-models' + "\n"
        root_rule += "grammar-models ::= " + " | ".join(
            [format_model_and_field_name(model.__name__) for model in models]
        )
        all_rules.insert(0, root_rule)
        return "\n".join(all_rules) + get_primitive_grammar("\n".join(all_rules))
    elif outer_object_name is not None:
        if list_of_outputs:
            root_rule = (
                rf'root ::= (" "| "\n")? "[" {format_model_and_field_name(outer_object_name)} ("," ws {format_model_and_field_name(outer_object_name)})* "]"'
                + "\n"
            )
        else:
            root_rule = f"root ::= {format_model_and_field_name(outer_object_name)}\n"

        if add_inner_thoughts:
            if allow_only_inner_thoughts:
                model_rule = rf'{format_model_and_field_name(outer_object_name)} ::= "{{" ws "\"{inner_thought_field_name}\""  ":" ws string (("," ws "\"{outer_object_name}\""  ":" ws grammar-models)? | ws "}}")'
            else:
                model_rule = rf'{format_model_and_field_name(outer_object_name)} ::= "{{" ws "\"{inner_thought_field_name}\""  ":" ws string "," ws "\"{outer_object_name}\""  ":" ws grammar-models '
        else:
            model_rule = rf'{format_model_and_field_name(outer_object_name)} ::= "{{" ws "\"{outer_object_name}\""  ": " grammar-models'

        fields_joined = " | ".join(
            [
                rf"{format_model_and_field_name(model.__name__)}-grammar-model"
                for model in models
            ]
        )

        grammar_model_rules = f"\ngrammar-models ::= {fields_joined}"
        mod_rules = []
        for model in models:
            mod_rule = (
                rf"{format_model_and_field_name(model.__name__)}-grammar-model ::= "
            )
            mod_rule += (
                rf'"\"{model.__name__}\"" "," ws "\"{outer_object_content}\"" ": " {format_model_and_field_name(model.__name__)}'
                + "\n"
            )
            mod_rules.append(mod_rule)
        grammar_model_rules += "\n" + "\n".join(mod_rules)

        for model in models:
            model_rules, has_special_string = generate_gbnf_grammar(
                model, processed_models, created_rules
            )
            if add_request_heartbeat and model.__name__ in request_heartbeat_models:
                model_rules[
                    0
                ] += rf' "," ws "\"{request_heartbeat_field_name}\""  ":" ws boolean '
            if not has_special_string:
                model_rules[0] += r' ws "}"'

            all_rules.extend(model_rules)

        all_rules.insert(0, root_rule + model_rule + grammar_model_rules)
        return "\n".join(all_rules) + get_primitive_grammar("\n".join(all_rules))


def get_primitive_grammar(grammar):
    """
    Returns the needed GBNF primitive grammar for a given GBNF grammar string.

    Args:
        grammar (str): The string containing the GBNF grammar.

    Returns:
        str: GBNF primitive grammar string.
    """
    type_list: list[type[object]] = []
    if "string-list" in grammar:
        type_list.append(str)
    if "boolean-list" in grammar:
        type_list.append(bool)
    if "integer-list" in grammar:
        type_list.append(int)
    if "float-list" in grammar:
        type_list.append(float)
    additional_grammar = [generate_list_rule(t) for t in type_list]
    primitive_grammar = r"""
boolean ::= "true" | "false"
null ::= "null"
string ::= "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\""
ws ::= (" ")?
number ::= "-"? ([0-9]+ | [0-9]+ "." [0-9]+) ([eE] [-+]? [0-9]+)?"""

    any_block = ""
    if "custom-class-any" in grammar:
        any_block = """
value ::= object | array | string | number | boolean | null

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}"

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]"
"""

    markdown_code_block_grammar = ""
    if "markdown-code-block" in grammar:
        markdown_code_block_grammar = r'''
markdown-code-block ::= opening-triple-ticks markdown-code-block-content closing-triple-ticks
markdown-code-block-content ::= ( [^`] | "`" [^`] |  "`"  "`" [^`]  )*
opening-triple-ticks ::= "```" "python" "\n" | "```" "c" "\n" | "```" "cpp" "\n" | "```" "txt" "\n" | "```" "text" "\n" | "```" "json" "\n" | "```" "javascript" "\n" | "```" "css" "\n" | "```" "html" "\n" | "```" "markdown" "\n"
closing-triple-ticks ::= "```" "\n"'''

    if "triple-quoted-string" in grammar:
        markdown_code_block_grammar = r"""
triple-quoted-string ::= triple-quotes triple-quoted-string-content triple-quotes
triple-quoted-string-content ::= ( [^'] | "'" [^'] |  "'"  "'" [^']  )*
triple-quotes ::= "'''" """
    return (
        "\n"
        + "\n".join(additional_grammar)
        + any_block
        + primitive_grammar
        + markdown_code_block_grammar
    )


def save_gbnf_grammar_and_documentation(
    grammar,
    documentation,
    grammar_file_path="./grammar.gbnf",
    documentation_file_path="./grammar_documentation.md",
):
    """
    Save GBNF grammar and documentation to specified files.

    Args:
        grammar (str): GBNF grammar string.
        documentation (str): Documentation string.
        grammar_file_path (str): File path to save the GBNF grammar.
        documentation_file_path (str): File path to save the documentation.

    Returns:
        None
    """
    try:
        with open(grammar_file_path, "w") as file:
            file.write(grammar + get_primitive_grammar(grammar))
        print(f"Grammar successfully saved to {grammar_file_path}")
    except IOError as e:
        print(f"An error occurred while saving the grammar file: {e}")

    try:
        with open(documentation_file_path, "w") as file:
            file.write(documentation)
        print(f"Documentation successfully saved to {documentation_file_path}")
    except IOError as e:
        print(f"An error occurred while saving the documentation file: {e}")


def remove_empty_lines(string):
    """
    Remove empty lines from a string.

    Args:
        string (str): Input string.

    Returns:
        str: String with empty lines removed.
    """
    lines = string.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_no_empty_lines = "\n".join(non_empty_lines)
    return string_no_empty_lines


def generate_and_save_gbnf_grammar_and_documentation(
    pydantic_model_list,
    grammar_file_path="./generated_grammar.gbnf",
    documentation_file_path="./generated_grammar_documentation.md",
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description=True,
    add_inner_thoughts: bool = False,
    allow_only_inner_thoughts: bool = False,
    inner_thoughts_field_name: str = "thoughts_and_reasoning",
    add_request_heartbeat: bool = False,
    request_heartbeat_field_name: str = "request_heartbeat",
    request_heartbeat_models: List[str] = None,
):
    """
    Generate GBNF grammar and documentation, and save them to specified files.

    Args:
        pydantic_model_list: List of Pydantic model classes.
        grammar_file_path (str): File path to save the generated GBNF grammar.
        documentation_file_path (str): File path to save the generated documentation.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        model_prefix (str): Prefix for the model section in the documentation.
        fields_prefix (str): Prefix for the fields section in the documentation.
        list_of_outputs (bool): Whether the output is a list of items.
        documentation_with_field_description (bool): Include field descriptions in the documentation.
        add_inner_thoughts (bool): Add inner thoughts to the grammar. This is useful for adding comments or reasoning to the output.
        allow_only_inner_thoughts (bool): Allow only inner thoughts. If True, only inner thoughts will be allowed in the output.
        inner_thoughts_field_name (str): Field name for inner thoughts. Default is "thoughts_and_reasoning".
        add_request_heartbeat (bool): Add request heartbeat to the grammar. This allows the LLM to decide when to return control to the system.
        request_heartbeat_field_name (str): Field name for request heartbeat. Default is "request_heartbeat".
        request_heartbeat_models (List[str]): List of models that will have a request heartbeat field.
    Returns:
        None
    """
    documentation = generate_markdown_documentation(
        pydantic_model_list,
        model_prefix,
        fields_prefix,
        documentation_with_field_description=documentation_with_field_description,
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(
        pydantic_model_list,
        outer_object_name,
        outer_object_content,
        list_of_outputs,
        add_inner_thoughts,
        allow_only_inner_thoughts,
        inner_thoughts_field_name,
        add_request_heartbeat,
        request_heartbeat_field_name,
        request_heartbeat_models,
    )
    grammar = remove_empty_lines(grammar)
    save_gbnf_grammar_and_documentation(
        grammar, documentation, grammar_file_path, documentation_file_path
    )


def generate_gbnf_grammar_and_documentation(
    pydantic_model_list,
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description=True,
    add_inner_thoughts: bool = False,
    allow_only_inner_thoughts: bool = False,
    inner_thoughts_field_name: str = "thoughts_and_reasoning",
    add_request_heartbeat: bool = False,
    request_heartbeat_field_name: str = "request_heartbeat",
    request_heartbeat_models: List[str] = None,
):
    """
    Generate GBNF grammar and documentation for a list of Pydantic models.

    Args:
        pydantic_model_list: List of Pydantic model classes.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. E.g., "function_parameters" or "params" for function calling.
        model_prefix (str): Prefix for the model section in the documentation.
        fields_prefix (str): Prefix for the fields section in the documentation.
        list_of_outputs (bool): Whether the output is a list of items.
        documentation_with_field_description (bool): Include field descriptions in the documentation.
        add_inner_thoughts (bool): Add inner thoughts to the grammar. This is useful for adding comments or reasoning to the output.
        allow_only_inner_thoughts (bool): Allow only inner thoughts. If True, only inner thoughts will be allowed in the output.
        inner_thoughts_field_name (str): Field name for inner thoughts. Default is "thoughts_and_reasoning".
        add_request_heartbeat (bool): Add request heartbeat to the grammar. This allows the LLM to decide when to return control to the system.
        request_heartbeat_field_name (str): Field name for request heartbeat. Default is "request_heartbeat".
        request_heartbeat_models (List[str]): List of models that will have a request heartbeat field.

    Returns:
        tuple: GBNF grammar string, documentation string.
    """
    documentation = generate_text_documentation(
        copy(pydantic_model_list),
        model_prefix,
        fields_prefix,
        documentation_with_field_description=documentation_with_field_description,
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(
        pydantic_model_list,
        outer_object_name,
        outer_object_content,
        list_of_outputs,
        add_inner_thoughts,
        allow_only_inner_thoughts,
        inner_thoughts_field_name,
        add_request_heartbeat,
        request_heartbeat_field_name,
        request_heartbeat_models,
    )
    grammar = remove_empty_lines(grammar + get_primitive_grammar(grammar))
    return grammar, documentation


def generate_gbnf_grammar_and_documentation_from_dictionaries(
    dictionaries: list[dict[str, Any]],
    outer_object_name: str | None = None,
    outer_object_content: str | None = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description=True,
    add_inner_thoughts: bool = False,
    allow_only_inner_thoughts: bool = False,
    inner_thoughts_field_name: str = "thoughts_and_reasoning",
    add_request_heartbeat: bool = False,
    request_heartbeat_field_name: str = "request_heartbeat",
    request_heartbeat_models: List[str] = None,
):
    """
    Generate GBNF grammar and documentation from a list of dictionaries.

    Args:
        dictionaries (list[dict]): List of dictionaries representing Pydantic models.
        outer_object_name (str): Outer object name for the GBNF grammar. If None, no outer object will be generated. Eg. "function" for function calling.
        outer_object_content (str): Content for the outer rule in the GBNF grammar. Eg. "function_parameters" or "params" for function calling.
        model_prefix (str): Prefix for the model section in the documentation.
        fields_prefix (str): Prefix for the fields section in the documentation.
        list_of_outputs (bool): Whether the output is a list of items.
        documentation_with_field_description (bool): Include field descriptions in the documentation.
        add_inner_thoughts (bool): Add inner thoughts to the grammar. This is useful for adding comments or reasoning to the output.
        allow_only_inner_thoughts (bool): Allow only inner thoughts. If True, only inner thoughts will be allowed in the output.
        inner_thoughts_field_name (str): Field name for inner thoughts. Default is "thoughts_and_reasoning".
        add_request_heartbeat (bool): Add request heartbeat to the grammar. This allows the LLM to decide when to return control to the system.
        request_heartbeat_field_name (str): Field name for request heartbeat. Default is "request_heartbeat".
        request_heartbeat_models (List[str]): List of models that will have a request heartbeat field.

    Returns:
        tuple: GBNF grammar string, documentation string.
    """
    pydantic_model_list = create_dynamic_models_from_dictionaries(dictionaries)
    documentation = generate_markdown_documentation(
        copy(pydantic_model_list),
        model_prefix,
        fields_prefix,
        documentation_with_field_description=documentation_with_field_description,
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(
        pydantic_model_list,
        outer_object_name,
        outer_object_content,
        list_of_outputs,
        add_inner_thoughts,
        allow_only_inner_thoughts,
        inner_thoughts_field_name,
        add_request_heartbeat,
        request_heartbeat_field_name,
        request_heartbeat_models,
    )
    grammar = remove_empty_lines(grammar + get_primitive_grammar(grammar))
    return grammar, documentation

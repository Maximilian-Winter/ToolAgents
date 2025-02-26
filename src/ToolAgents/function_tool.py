import inspect
import json
import re

import typing

from typing import Type,  Dict


from abc import ABC, abstractmethod
from typing import Any, List, Union, Callable, Tuple, TypeVar


from docstring_parser import DocstringStyle, parse
from pydantic import BaseModel, create_model, Field
from pydantic_core import PydanticUndefined

from ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_from_pydantic_models

from ToolAgents.utilities.llm_documentation.documentation_generation import generate_text_documentation, generate_function_definition


def format_model_and_field_name(model_name: str) -> str:
    parts = re.findall("[A-Z][^A-Z]*", model_name)
    if not parts:  # Check if the list is empty
        return model_name.lower().replace("_", "-")
    return "-".join(part.lower().replace("_", "-") for part in parts)


def create_dynamic_model_from_function(
        func: Callable[..., Any],
        add_inner_thoughts: bool = False,
        inner_thoughts_field_name: str = "inner_thoughts",
):
    """
    Creates a dynamic Pydantic model from a given function's type hints and adds the function as a 'run' method.

    Args:
        func (Callable): A function with type hints from which to create the model.
        add_inner_thoughts (bool): Add an 'inner_thoughts' field at the params level to the model. Default is False.
        inner_thoughts_field_name (str): Field name for inner thoughts. Default is "inner_thoughts".

    Returns:
        A dynamic Pydantic model class with the provided function as a 'run' method.
    """

    # Get the signature of the function
    sig = inspect.signature(func)

    # Parse the docstring
    assert func.__doc__ is not None
    docstring = parse(func.__doc__, style=DocstringStyle.AUTO)
    dynamic_fields = {}
    param_docs = []
    if add_inner_thoughts:
        dynamic_fields[inner_thoughts_field_name] = (str, None)
    for param in sig.parameters.values():
        # Exclude 'self' parameter
        if param.name == "self":
            continue

        # Assert that the parameter has a type annotation
        if param.annotation == inspect.Parameter.empty:
            raise TypeError(
                f"Parameter '{param.name}' in function '{func.__name__}' lacks a type annotation"
            )

        # Find the parameter's description in the docstring
        param_doc = next(
            (d for d in docstring.params if d.arg_name == param.name), None
        )

        # Assert that the parameter has a description
        if not param_doc or not param_doc.description:
            # Find the parameter's description in the docstring
            param_doc = next(
                (d for d in docstring2.params if d.arg_name == param.name), None
            )
            if not param_doc or not param_doc.description:
                raise ValueError(
                    f"Parameter '{param.name}' in function '{func.__name__}' lacks a description in the docstring"
                )

        # Add parameter details to the schema
        param_docs.append((param.name, param_doc))
        if param.default == inspect.Parameter.empty:
            default_value = ...
        else:
            default_value = param.default
        dynamic_fields[param.name] = (
            param.annotation if param.annotation != inspect.Parameter.empty else str,
            default_value,
        )

    # Creating the dynamic model
    dynamic_model = create_model(f"{func.__name__}", **dynamic_fields)  # type: ignore[call-overload]
    if add_inner_thoughts:
        dynamic_model.model_fields[
            inner_thoughts_field_name
        ].description = "Deep inner monologue private to you only."
    for name, param_doc in param_docs:
        dynamic_model.model_fields[name].description = param_doc.description

    dynamic_model.__doc__ = (
            (docstring.short_description if docstring.short_description is not None else "")
            + "\n"
            + (docstring.long_description if docstring.long_description is not None else "")
    )

    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name, _ in dynamic_fields.items()}
        return func(**func_args)

    # Adding the wrapped function as a 'run' method
    setattr(dynamic_model, "run", run_method_wrapper)
    return dynamic_model


def add_run_method_to_dynamic_model(model: type[BaseModel], func: Callable[..., Any]):
    """
    Add a 'run' method to a dynamic Pydantic model, using the provided function.

    Args:
        model (type[BaseModel]): Dynamic Pydantic model class.
        func (Callable): Function to be added as a 'run' method to the model.

    Returns:
        type[BaseModel]: Pydantic model class with the added 'run' method.
    """

    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name in model.model_fields}
        return func(**func_args)

    # Adding the wrapped function as a 'run' method
    setattr(model, "run", run_method_wrapper)

    return model


def create_dynamic_models_from_dictionaries(dictionaries: list[dict[str, Any]]):
    """
    Create a list of dynamic Pydantic model classes from a list of dictionaries.

    Args:
        dictionaries (list[dict]): List of dictionaries representing model structures.

    Returns:
        list[Type[BaseModel]]: List of generated dynamic Pydantic model classes.
    """
    dynamic_models = []
    for func in dictionaries:
        model_name = format_model_and_field_name(func.get("name", ""))
        dyn_model = convert_dictionary_to_pydantic_model(func, model_name)
        dynamic_models.append(dyn_model)
    return dynamic_models


def map_grammar_names_to_pydantic_model_class(pydantic_model_list):
    output = {}
    for model in pydantic_model_list:
        output[format_model_and_field_name(model.__name__)] = model

    return output


from enum import Enum


def json_schema_to_python_types(schema):
    type_map = {
        "any": Any,
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
    }
    return type_map[schema]


def list_to_enum(enum_name, values):
    return Enum(enum_name, {value: value for value in values})


def convert_dictionary_to_pydantic_model(
        dictionary: dict[str, Any],
        model_name: str = "CustomModel",
        docs: dict[str, str] = None,
        docs_function: dict[str, str] = None,
) -> type[Any]:
    """
    Convert a dictionary to a Pydantic model class.

    Args:
        dictionary (dict): Dictionary representing the model structure.
        model_name (str): Name of the generated Pydantic model.

    Returns:
        type[BaseModel]: Generated Pydantic model class.
    """
    fields: dict[str, Any] = {}
    if docs is None:
        docs = {}
    if docs_function is None:
        docs_function = {}
    if "properties" in dictionary:
        for field_name, field_data in dictionary.get("properties", {}).items():
            if field_data == "object":
                submodel = convert_dictionary_to_pydantic_model(
                    dictionary, f"{model_name}_{field_name}", docs, docs_function
                )
                fields[field_name] = (submodel, ...)

            else:
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
                        array = {"properties": items}
                        array_type = convert_dictionary_to_pydantic_model(
                            array,
                            f"{model_name}_{field_name}_items",
                            docs,
                            docs_function,
                        )
                        fields[field_name] = (List[array_type], ...)  # type: ignore[valid-type]
                    else:
                        fields[field_name] = (list, ...)
                elif field_type == "object":
                    submodel = convert_dictionary_to_pydantic_model(
                        field_data, f"{model_name}_{field_name}", docs, docs_function
                    )
                    fields[field_name] = (submodel, ...)
                elif field_type == "required":
                    required = field_data
                    for key, field in fields.items():
                        if key not in required:
                            fields[key] = (typing.Optional[fields[key][0]], ...)
                else:
                    field_type = json_schema_to_python_types(field_type)
                    fields[field_name] = (field_type, ...)
    if "function" in dictionary:
        for field_name, field_data in dictionary.get("function", {}).items():
            if field_name == "name":
                model_name = field_data
            elif field_name == "description":
                docs_function["__doc__"] = field_data
            elif field_name == "parameters":
                return convert_dictionary_to_pydantic_model(
                    field_data, f"{model_name}", docs, docs_function
                )

    if "parameters" in dictionary:
        field_data = {"function": dictionary}
        return convert_dictionary_to_pydantic_model(
            field_data, f"{model_name}", docs, docs_function
        )
    if "required" in dictionary:
        required = dictionary.get("required", [])
        for key, field in fields.items():
            if key not in required:
                fields[key] = (typing.Optional[fields[key][0]], ...)
    custom_model = create_model(model_name, **fields)

    if "__doc__" in docs_function:
        custom_model.__doc__ = docs_function["__doc__"]
    for field_name, doc in docs.items():
        custom_model.model_fields[field_name].description = doc

    return custom_model


def get_enum_type(enum):
    """Determine the JSON schema type for an enum based on its members."""
    enum_values = [e.value for e in enum]
    if all(isinstance(e, int) for e in enum_values):
        return {"type": "integer", "enum": enum_values}
    elif all(isinstance(e, float) for e in enum_values):
        return {"type": "number", "enum": enum_values}
    else:
        return {"type": "string", "enum": enum_values}


def py_type_to_json_type(schema):
    type_map = {
        Any: {"type": "any"},
        str: {"type": "string"},
        float: {"type": "number"},
        int: {"type": "integer"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return type_map[schema]


def get_openai_type(py_type):
    """Map Python types to JSON schema types and handle special cases like Enums, Lists, and Unions."""
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        # Handle Enum types by determining their actual value types
        return get_enum_type(py_type)
    elif inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        # Handle nested Pydantic models by recursive call
        return {
            "type": "object",
            "properties": pydantic_model_to_openai_function_definition(py_type)[
                "function"
            ]["parameters"]["properties"],
        }
    elif hasattr(py_type, "__origin__"):
        if py_type.__origin__ is Union:
            # Filter out NoneType to handle optional fields
            non_none_types = [get_openai_type(t) for t in py_type.__args__ if t is not type(None)]
            return non_none_types
        elif py_type.__origin__ is Dict or py_type.__origin__ is dict:
            # Handle lists by identifying the type of list items
            return {"type": "object"}
        elif py_type.__origin__ is List or py_type.__origin__ is list:
            # Handle lists by identifying the type of list items
            return {"type": "array", "items": get_openai_type(py_type.__args__[0])}
    else:
        # Default type handling
        return py_type_to_json_type(py_type)


def pydantic_model_to_openai_function_definition(pydantic_model: Type[BaseModel]):
    model_schema = pydantic_model.model_json_schema()
    properties = model_schema["properties"]
    required_fields = model_schema.get("required", [])
    class_doc = inspect.getdoc(pydantic_model)
    base_class_doc = inspect.getdoc(BaseModel)
    class_description = class_doc if class_doc and class_doc != base_class_doc else ""

    function_definition = {
        "type": "function",
        "function": {
            "name": pydantic_model.__name__,
            "description": class_description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": required_fields,
            },
        },
    }

    type_hints = typing.get_type_hints(pydantic_model)
    for prop_name, prop_meta in properties.items():
        prop_type = type_hints[prop_name]

        openai_type = get_openai_type(prop_type)
        field_info = pydantic_model.model_fields.get(prop_name)
        field_description = (
            field_info.description if field_info and field_info.description else ""
        )
        if isinstance(openai_type,
                      dict) and field_info.default is not PydanticUndefined and field_info.default is not None:
            continue
        if isinstance(openai_type, list) and len(openai_type) > 1:
            # Handling Union types specifically
            function_definition["function"]["parameters"]["properties"][prop_name] = {
                "type": "union",
                "options": openai_type
            }
            if len(field_description) > 0:
                function_definition["function"]["parameters"]["properties"][prop_name]["description"] = field_description
        elif isinstance(openai_type, list):
            if len(field_description) > 0:
                openai_type[0]["description"] = field_description
            function_definition["function"]["parameters"]["properties"][prop_name] = openai_type[0]
        else:
            function_definition["function"]["parameters"]["properties"][prop_name] = {
                **openai_type
            }
            if len(field_description) > 0:
                function_definition["function"]["parameters"]["properties"][prop_name]["description"] = field_description

    return function_definition


def add_field_to_model(
        model_class: Type[BaseModel],
        field_name: str,
        field_type: Type[Any],
        default: Any = None,
        required: bool = True
) -> Type[BaseModel]:
    """
    Adds a new field to an existing Pydantic model class.

    Args:
        model_class: The original Pydantic model class
        field_name: Name of the field to add
        field_type: Type of the field (e.g., str, int, etc.)
        default: Default value for the field (None if not provided)
        required: Whether the field is required (True by default)

    Returns:
        A new Pydantic model class with the added field

    Example:
        class UserBase(BaseModel):
            name: str
            age: int

        UpdatedUser = add_field_to_model(
            UserBase,
            "email",
            str,
            default="user@example.com",
            required=False
        )
    """
    # Get existing model fields
    existing_fields = model_class.__annotations__

    # Create field configuration
    field_config = {}
    if not required:
        if default is not None:
            field_config["default"] = default
        else:
            field_config["default"] = None
    elif default is not None:
        field_config["default"] = default

    # Create the new field with Field configuration
    new_field = Field(**field_config)

    # Create new annotations dictionary with the added field
    new_annotations = {
        **existing_fields,
        field_name: field_type
    }

    # Create new namespace for the model
    namespace = {
        "__annotations__": new_annotations,
        field_name: new_field,
        **{k: v for k, v in model_class.__dict__.items()
           if not k.startswith("_") and k != "model_fields"}
    }

    # Create new model class
    new_model = type(
        f"{model_class.__name__}With{field_name.title()}",
        (BaseModel,),
        namespace
    )

    return new_model

T = TypeVar('T')
U = TypeVar('U')


class BaseProcessor(ABC):
    """
    Abstract base class for all processors (pre and post).
    """

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process the input data and return transformed data.

        Args:
            data: Input data to process

        Returns:
            Processed data
        """
        pass

    def __call__(self, data: Any) -> Any:
        """
        Make the processor callable.

        Args:
            data: Input data to process

        Returns:
            Processed data
        """
        return self.process(data)


class PreProcessor(BaseProcessor):
    """
    Abstract base class for preprocessing parameters before function execution.
    """

    @abstractmethod
    def process(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Process the input parameters before function execution.

        Args:
            parameters: Dictionary of input parameters

        Returns:
            Processed parameters dictionary
        """
        pass


class PostProcessor(BaseProcessor):
    """
    Abstract base class for postprocessing function results.
    """

    @abstractmethod
    def process(self, result: Any) -> Any:
        """
        Process the function result.

        Args:
            result: Function execution result

        Returns:
            Processed result
        """
        pass


class FunctionTool:
    """
    Class representing a function tool for a LLM.

    Args:
        function_tool: The function tool, can be either a pydantic model with a run method,
            a python function or a tuple of a OpenAI tool specification and a function as callback.
        pre_processors: Single preprocessor or list of preprocessors to apply before function execution
        post_processors: Single postprocessor or list of postprocessors to apply after function execution
        debug_mode: Enable debug mode to print parameters
        **additional_parameters: Additional parameters to pass to the call if function is called.
    """

    def __init__(
            self,
            function_tool: Union[BaseModel, Callable, Tuple[Dict[str, Any], Callable]],
            pre_processors: Union[PreProcessor, List[PreProcessor], Callable, List[Callable], None] = None,
            post_processors: Union[PostProcessor, List[PostProcessor], Callable, List[Callable], None] = None,
            debug_mode: bool = False,
            **additional_parameters,
    ):
        # Initialize function tool as before...
        if isinstance(function_tool, type) and issubclass(function_tool, BaseModel):
            self.model = function_tool
        elif (
                isinstance(function_tool, tuple)
                and len(function_tool) == 2
                and isinstance(function_tool[0], dict)
                and callable(function_tool[1])
        ):
            models = create_dynamic_models_from_dictionaries([function_tool[0]])
            self.model = add_run_method_to_dynamic_model(models[0], function_tool[1])
        elif callable(function_tool):
            self.model = create_dynamic_model_from_function(function_tool)
        else:
            raise ValueError("Invalid function_tool type provided")

        # Initialize processors
        self.pre_processors = self._normalize_processors(pre_processors)
        self.post_processors = self._normalize_processors(post_processors)

        self.debug_mode = debug_mode
        self.additional_parameters = additional_parameters if additional_parameters else {}

    @staticmethod
    def _normalize_processors(
            processors: Union[BaseProcessor, List[BaseProcessor], Callable, List[Callable], None]
    ) -> List[BaseProcessor]:
        """
        Normalize processors input to a list of BaseProcessor instances.

        Args:
            processors: Single processor or list of processors

        Returns:
            List of BaseProcessor instances
        """
        if processors is None:
            return []

        if not isinstance(processors, list):
            processors = [processors]

        normalized = []
        for proc in processors:
            if isinstance(proc, BaseProcessor):
                normalized.append(proc)
            elif callable(proc):
                # Wrap callable in an anonymous processor class
                normalized.append(
                    type('CallableProcessor', (BaseProcessor,), {
                        'process': staticmethod(proc)
                    })()
                )
            else:
                raise ValueError(f"Invalid processor type: {type(proc)}")

        return normalized

    def set_name(self, new_name: str):
        self.model.__name__ = new_name

    def set_keyword_argument(self, key: str, value: Any):
        self.additional_parameters[key] = value

    def get_python_documentation(self):
        return generate_function_definition(self.model, self.model.__name__, self.model.__doc__)

    def get_text_documentation(self):
        return generate_text_documentation([self.model], "Function", "Arguments")

    @staticmethod
    def from_pydantic_model_and_callable(
            pydantic_model: BaseModel, tool_function: Callable
    ):
        """
        Converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
        Args:
            pydantic_model(BaseModel): Pydantic Model representing the arguments to the tool.
            tool_function(Callable): Callable function that will be invoked when the agent uses it and will be passed the fields of the pydantic model.

        Returns:
            LlamaCppFunctionTool: The LlamaCppFunctionTool instance.
        """
        pydantic_model = add_run_method_to_dynamic_model(pydantic_model, tool_function)
        return FunctionTool(pydantic_model)

    @staticmethod
    def from_openai_tool(openai_tool_schema: dict, tool_function: Callable):
        """
        Converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
        Args:
            openai_tool_schema(dict): OpenAI tool description dictionary.
            tool_function(Callable): Callable function that will be invoked when the agent uses it.

        Returns:
            LlamaCppFunctionTool: The LlamaCppFunctionTool instance.
        """
        models = create_dynamic_models_from_dictionaries([openai_tool_schema])
        model = add_run_method_to_dynamic_model(models[0], tool_function)
        return FunctionTool(model)

    @staticmethod
    def from_llama_index_tool(llama_index_tool):
        """
        Converts a llama-index tool into a LlamaCppFunctionTool
        Args:
            llama_index_tool(["BaseTool"]): OpenAI tool description dictionary.

        Returns:
            LlamaCppFunctionTool: The LlamaCppFunctionTool instance.
        """
        models = create_dynamic_models_from_dictionaries(
            [llama_index_tool.metadata.to_openai_tool()]
        )
        model = add_run_method_to_dynamic_model(models[0], llama_index_tool.call)
        return FunctionTool(model)

    def to_mistral_tool(self):
        from mistral_common.protocol.instruct.tool_calls import Tool, Function

        root = pydantic_model_to_openai_function_definition(self.model)
        return Tool(
            function=Function(
                name=root["function"]["name"],
                description=root["function"]["description"],
                parameters=root["function"]["parameters"],
            )
        )

    def to_openai_tool(self):
        root = pydantic_model_to_openai_function_definition(self.model)
        return root

    def to_anthropic_tool(self):
        root = pydantic_model_to_openai_function_definition(self.model)
        function = root["function"]

        return {
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"]
        }

    def to_nous_hermes_pro_tool(self):
        root = pydantic_model_to_openai_function_definition(self.model)
        function = root["function"]

        nous_hermes_pro_tool = {
            "type": "function",
            "function": {
                "name": function["name"],
                "description": function["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": function["parameters"].get("required", [])
                }
            }
        }

        for prop_name, prop_info in function["parameters"]["properties"].items():
            nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name] = {
                "type": prop_info.get("type", "string"),
                "description": prop_info.get("description", "")
            }

            # Handle enum types
            if "enum" in prop_info:
                nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name]["enum"] = prop_info["enum"]

            # Handle array types
            if prop_info.get("type") == "array" and "items" in prop_info:
                nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name]["items"] = {
                    "type": prop_info["items"].get("type", "string")
                }
                if "enum" in prop_info["items"]:
                    nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name]["items"]["enum"] = \
                        prop_info["items"]["enum"]

        return nous_hermes_pro_tool

    def execute(self, parameters: dict[str, Any]) -> Any:
        """
        Execute the function tool with the given parameters.

        Args:
            parameters: Input parameters for the function

        Returns:
            Processed result from the function execution
        """
        if self.debug_mode:
            print("Input parameters:")
            print(json.dumps(parameters, indent=4))

        # Apply pre-processors in sequence
        processed_params = parameters
        for processor in self.pre_processors:
            try:
                processed_params = processor(processed_params)
                if self.debug_mode:
                    print(f"After {processor.__class__.__name__}:")
                    print(json.dumps(processed_params, indent=4))
            except Exception as e:
                print(f"Error in {processor.__class__.__name__}: {str(e)}")
                raise Exception(f"Error in {processor.__class__.__name__}: {str(e)}")

        # Execute function
        try:
            instance = self.model(**processed_params)
            result = instance.run(**self.additional_parameters)
        except Exception as e:
            print(f"Error in function execution: {str(e)}")
            return f"Error in function execution: {str(e)}"

        # Apply post-processors in sequence
        processed_result = result
        for processor in self.post_processors:
            try:
                processed_result = processor(processed_result)
                if self.debug_mode:
                    print(f"After {processor.__class__.__name__}:")
                    print(processed_result)
            except Exception as e:
                print(f"Error in {processor.__class__.__name__}: {str(e)}")
                raise Exception(f"Error in {processor.__class__.__name__}: {str(e)}")

        return processed_result

    def add_pre_processor(
            self,
            processor: Union[PreProcessor, Callable],
            position: int = None
    ) -> None:
        """
        Add a new preprocessor to the function tool.

        Args:
            processor: Preprocessor to add (either PreProcessor instance or callable)
            position: Optional position to insert the processor (None = append to end)
        """
        normalized = self._normalize_processors([processor])[0]
        if position is None:
            self.pre_processors.append(normalized)
        else:
            self.pre_processors.insert(position, normalized)

    def add_post_processor(
            self,
            processor: Union[PostProcessor, Callable],
            position: int = None
    ) -> None:
        """
        Add a new postprocessor to the function tool.

        Args:
            processor: Postprocessor to add (either PostProcessor instance or callable)
            position: Optional position to insert the processor (None = append to end)
        """
        normalized = self._normalize_processors([processor])[0]
        if position is None:
            self.post_processors.append(normalized)
        else:
            self.post_processors.insert(position, normalized)

class ToolRegistry:
    def __init__(self, guided_sampling_enabled: bool = False):
        self.tools: Dict[str, FunctionTool] = {}
        self.guided_sampling_enabled = guided_sampling_enabled

    def reset_registry(self):
        self.tools = {}

    def add_tools(self, tools: List[FunctionTool]):
        for tool in tools:
            if tool.model.__name__ not in self.tools:
                self.tools[tool.model.__name__] = tool

    def add_tool(self, tool: FunctionTool):
        if tool.model.__name__ not in self.tools:
            self.tools[tool.model.__name__] = tool


    def remove(self, tool_name: str):
        del self.tools[tool_name]

    def get_tool(self, name: str) -> FunctionTool:
        return self.tools[name]

    def get_tools(self):
        return self.tools.values()

    def get_mistral_tools(self):
        return [tool.to_mistral_tool() for tool in self.tools.values()]

    def get_openai_tools(self):
        return [tool.to_openai_tool() for tool in self.tools.values()]

    def get_anthropic_tools(self):
        return [tool.to_anthropic_tool() for tool in self.tools.values()]

    def get_nous_hermes_pro_tools(self):
        return [tool.to_nous_hermes_pro_tool() for tool in self.tools.values()]

    def get_guided_sampling_grammar(self):
        return generate_gbnf_grammar_from_pydantic_models(models=[tool.model for tool in self.tools.values()],
                                                          outer_object_name="name", outer_object_content="arguments",
                                                          list_of_outputs=True, add_inner_thoughts=False,
                                                          allow_only_inner_thoughts=False, add_request_heartbeat=False)

    def get_guided_sampling_json_schema(self):
        pass
        #return generate_json_schemas(models=[tool.model for tool in self.tools.values()], outer_object_name="name",
        #                             allow_list=True, outer_object_properties_name="arguments")

    def get_tools_documentation(self):
        return generate_text_documentation([tool.model for tool in self.tools.values()], model_prefix="Tool",
                                           fields_prefix="Arguments")

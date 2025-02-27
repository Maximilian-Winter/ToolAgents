import asyncio

import typing
from enum import auto, Enum

from typing import Type, Dict


from abc import ABC, abstractmethod
from typing import Any, List, Union, Callable, Tuple

from pydantic import BaseModel

from ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_from_pydantic_models,
)
from ToolAgents.utilities.json_schema_generator.old_schema_generator import (
    generate_json_schemas,
)

from ToolAgents.utilities.llm_documentation.documentation_generation import (
    generate_text_documentation,
    generate_function_definition,
)
from ToolAgents.utilities.pydantic_utilites import (
    create_dynamic_models_from_dictionaries,
    add_run_method_to_dynamic_model,
    create_dynamic_model_from_function,
    pydantic_model_to_openai_function_definition,
)


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


class ConfirmationState(Enum):
    """Represents the state of a confirmation request."""

    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()


class ConfirmationRequest:
    """Represents a confirmation request for a function execution."""

    def __init__(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        description: typing.Optional[str] = None,
    ):
        self.function_name = function_name
        self.parameters = parameters
        self.description = description
        self.state = ConfirmationState.PENDING
        self._future = asyncio.Future()

    def approve(self):
        """Approve the function execution."""
        self.state = ConfirmationState.APPROVED
        self._future.set_result(True)

    def reject(self):
        """Reject the function execution."""
        self.state = ConfirmationState.REJECTED
        self._future.set_result(False)

    async def wait_for_decision(self) -> bool:
        """Wait for user decision on the confirmation request."""
        return await self._future


class FunctionTool:
    """
    Class representing a function tool for a LLM.

    Args:
        function_tool: The function tool, can be either a pydantic model with a run method,
            a python function or a tuple of a OpenAI tool specification and a function as callback.
        pre_processors: Single preprocessor or list of preprocessors to apply before function execution
        post_processors: Single postprocessor or list of postprocessors to apply after function execution
        debug_mode: Enable debug mode to print parameters
        require_confirmation: Whether to require user confirmation before executing the tool
        confirmation_description: Optional description for the confirmation request
        confirmation_handler: Optional callable that handles confirmation requests
        **additional_parameters: Additional parameters to pass to the call if function is called.
    """

    def __init__(
        self,
        function_tool: Union[BaseModel, Callable, Tuple[Dict[str, Any], Callable]],
        pre_processors=None,
        post_processors=None,
        debug_mode: bool = False,
        require_confirmation: bool = False,
        confirmation_description: typing.Optional[str] = None,
        confirmation_handler: typing.Optional[
            Callable[[ConfirmationRequest], None]
        ] = None,
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
        self.additional_parameters = (
            additional_parameters if additional_parameters else {}
        )

        # Confirmation-related properties
        self.require_confirmation = require_confirmation
        self.confirmation_description = confirmation_description
        self.confirmation_handler = confirmation_handler

    @staticmethod
    def _normalize_processors(
        processors: Union[
            BaseProcessor, List[BaseProcessor], Callable, List[Callable], None
        ],
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
                    type(
                        "CallableProcessor",
                        (BaseProcessor,),
                        {"process": staticmethod(proc)},
                    )()
                )
            else:
                raise ValueError(f"Invalid processor type: {type(proc)}")

        return normalized

    def set_name(self, new_name: str):
        self.model.__name__ = new_name

    def set_keyword_argument(self, key: str, value: Any):
        self.additional_parameters[key] = value

    def get_python_documentation(self):
        return generate_function_definition(
            self.model, self.model.__name__, self.model.__doc__
        )

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
            "input_schema": function["parameters"],
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
                    "required": function["parameters"].get("required", []),
                },
            },
        }

        for prop_name, prop_info in function["parameters"]["properties"].items():
            nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name] = {
                "type": prop_info.get("type", "string"),
                "description": prop_info.get("description", ""),
            }

            # Handle enum types
            if "enum" in prop_info:
                nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name][
                    "enum"
                ] = prop_info["enum"]

            # Handle array types
            if prop_info.get("type") == "array" and "items" in prop_info:
                nous_hermes_pro_tool["function"]["parameters"]["properties"][prop_name][
                    "items"
                ] = {"type": prop_info["items"].get("type", "string")}
                if "enum" in prop_info["items"]:
                    nous_hermes_pro_tool["function"]["parameters"]["properties"][
                        prop_name
                    ]["items"]["enum"] = prop_info["items"]["enum"]

        return nous_hermes_pro_tool

    def set_confirmation_handler(self, handler: Callable[[ConfirmationRequest], None]):
        """Set the handler for confirmation requests."""
        self.confirmation_handler = handler
        return self

    def enable_confirmation(self, description: typing.Optional[str] = None):
        """Enable confirmation for this tool."""
        self.require_confirmation = True
        if description:
            self.confirmation_description = description
        return self

    def disable_confirmation(self):
        """Disable confirmation for this tool."""
        self.require_confirmation = False
        return self

    def execute(self, parameters: Dict[str, Any]) -> Any:
        """
        Execute the function tool with the given parameters.
        If require_confirmation is True, this method will block until confirmation is received.

        Args:
            parameters: Input parameters for the function

        Returns:
            Processed result from the function execution
        """
        if self.require_confirmation:
            if not self.confirmation_handler:
                raise ValueError(
                    "Confirmation handler must be set when require_confirmation is True"
                )

            # Create a confirmation request
            request = ConfirmationRequest(
                function_name=self.model.__name__,
                parameters=parameters,
                description=self.confirmation_description,
            )

            # Call the handler and wait for decision
            self.confirmation_handler(request)

            # Run in asyncio event loop if we're not already in one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context
                    approved = asyncio.run_coroutine_threadsafe(
                        request.wait_for_decision(), loop
                    ).result()
                else:
                    # Start a new event loop
                    approved = asyncio.run(request.wait_for_decision())
            except RuntimeError:
                # No event loop exists, create one
                approved = asyncio.run(request.wait_for_decision())

            if not approved:
                return f"Execution of {self.model.__name__} was rejected by the user."

        # Standard execution path
        if self.debug_mode:
            print("Input parameters:")
            print(parameters)

        # Apply pre-processors in sequence
        processed_params = parameters
        for processor in self.pre_processors:
            try:
                processed_params = processor(processed_params)
            except Exception as e:
                print(f"Error in preprocessor: {str(e)}")
                raise

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
            except Exception as e:
                print(f"Error in postprocessor: {str(e)}")
                raise

        return processed_result

    async def execute_async(self, parameters: Dict[str, Any]) -> Any:
        """
        Asynchronously execute the function tool with the given parameters.
        If require_confirmation is True, this method will await confirmation.

        Args:
            parameters: Input parameters for the function

        Returns:
            Processed result from the function execution
        """
        if self.require_confirmation:
            if not self.confirmation_handler:
                raise ValueError(
                    "Confirmation handler must be set when require_confirmation is True"
                )

            # Create a confirmation request
            request = ConfirmationRequest(
                function_name=self.model.__name__,
                parameters=parameters,
                description=self.confirmation_description,
            )

            # Call the handler and await decision
            self.confirmation_handler(request)
            approved = await request.wait_for_decision()

            if not approved:
                return f"Execution of {self.model.__name__} was rejected by the user."

        # Standard execution path
        if self.debug_mode:
            print("Input parameters:")
            print(parameters)

        # Apply pre-processors in sequence
        processed_params = parameters
        for processor in self.pre_processors:
            try:
                processed_params = processor(processed_params)
            except Exception as e:
                print(f"Error in preprocessor: {str(e)}")
                raise

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
            except Exception as e:
                print(f"Error in postprocessor: {str(e)}")
                raise

        return processed_result

    def add_pre_processor(
        self, processor: Union[PreProcessor, Callable], position: int = None
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
        self, processor: Union[PostProcessor, Callable], position: int = None
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


class AsyncFunctionTool(FunctionTool):

    def __init__(
        self,
        function_tool: Union[BaseModel, Callable, Tuple[Dict[str, Any], Callable]],
        pre_processors=None,
        post_processors=None,
        debug_mode: bool = False,
        require_confirmation: bool = False,
        confirmation_description: typing.Optional[str] = None,
        confirmation_handler: typing.Optional[
            Callable[[ConfirmationRequest], None]
        ] = None,
        **additional_parameters,
    ) -> None:
        super().__init__(
            function_tool,
            pre_processors,
            post_processors,
            debug_mode,
            require_confirmation,
            confirmation_description,
            confirmation_handler,
            **additional_parameters,
        )

    async def execute_async(self, parameters: Dict[str, Any]) -> Any:
        """
        Asynchronously execute the function tool with the given parameters.
        If require_confirmation is True, this method will await confirmation.

        Args:
            parameters: Input parameters for the function

        Returns:
            Processed result from the function execution
        """
        if self.require_confirmation:
            if not self.confirmation_handler:
                raise ValueError(
                    "Confirmation handler must be set when require_confirmation is True"
                )

            # Create a confirmation request
            request = ConfirmationRequest(
                function_name=self.model.__name__,
                parameters=parameters,
                description=self.confirmation_description,
            )

            # Call the handler and await decision
            self.confirmation_handler(request)
            approved = await request.wait_for_decision()

            if not approved:
                return f"Execution of {self.model.__name__} was rejected by the user."

        # Standard execution path
        if self.debug_mode:
            print("Input parameters:")
            print(parameters)

        # Apply pre-processors in sequence
        processed_params = parameters
        for processor in self.pre_processors:
            try:
                processed_params = processor(processed_params)
            except Exception as e:
                print(f"Error in preprocessor: {str(e)}")
                raise

        # Execute function
        try:
            instance = self.model(**processed_params)
            result = await instance.run(**self.additional_parameters)
        except Exception as e:
            print(f"Error in function execution: {str(e)}")
            return f"Error in function execution: {str(e)}"

        # Apply post-processors in sequence
        processed_result = result
        for processor in self.post_processors:
            try:
                processed_result = processor(processed_result)
            except Exception as e:
                print(f"Error in postprocessor: {str(e)}")
                raise

        return processed_result


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
        return generate_gbnf_grammar_from_pydantic_models(
            models=[tool.model for tool in self.tools.values()],
            outer_object_name="tool",
            outer_object_content="parameters",
            list_of_outputs=True,
            add_inner_thoughts=False,
            allow_only_inner_thoughts=False,
            add_request_heartbeat=False,
        )

    def get_guided_sampling_json_schema(self):
        return generate_json_schemas(
            models=[tool.model for tool in self.tools.values()],
            outer_object_name="tool_calls",
            allow_list=True,
            outer_object_properties_name="tool",
        )

    def get_tools_documentation(self):
        return generate_text_documentation(
            [tool.model for tool in self.tools.values()],
            model_prefix="Tool",
            fields_prefix="Arguments",
        )

    def get_tools_information(self) -> Dict[str, Any]:
        result = {}
        for tool in self.tools.values():
            result[tool.model.__name__] = tool.model.model_dump()

        return result


def cli_confirmation_handler(request: ConfirmationRequest):
    """Simple CLI-based confirmation handler."""
    print(f"\n=== Confirmation Required for Tool Use: {request.function_name} ===")
    if request.description:
        print(f"Description: {request.description}")
    print("Parameters:")
    for key, value in request.parameters.items():
        print(f"  - {key}: {value}")

    response = input("\nApprove this execution? (y/n): ").strip().lower()

    if response == "y":
        request.approve()
    else:
        request.reject()

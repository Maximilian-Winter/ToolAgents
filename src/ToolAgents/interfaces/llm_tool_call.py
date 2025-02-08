import abc
import json
import random
import re
import string
from typing import Dict, Any, List, Optional

from ToolAgents.function_tool import ToolRegistry
from ToolAgents.utilities.chat_history import Message


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


class LLMToolCall(abc.ABC):

    @abc.abstractmethod
    def get_tool_call_id(self) -> str:
        pass

    @abc.abstractmethod
    def get_tool_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_tool_call_arguments(self) -> Dict[str, Any]:
        pass


class LLMToolCallHandler(abc.ABC):

    @abc.abstractmethod
    def contains_tool_calls(self, response: str) -> bool:
        pass

    @abc.abstractmethod
    def parse_tool_calls(self, response: str) -> [List[LLMToolCall], bool]:
        pass

    @abc.abstractmethod
    def get_tool_call_messages(self, tool_calls: List[LLMToolCall]) -> Dict[str, Any] | List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_tool_call_result_messages(self, tool_calls: List[LLMToolCall], tool_call_results: List[Any]) -> Dict[
                                                                                                                str, Any] | \
                                                                                                            List[Dict[
                                                                                                                str, Any]]:
        pass

    @abc.abstractmethod
    def execute_tool_calls(self, tool_calls: List[LLMToolCall], tool_registry: ToolRegistry) -> List[Any]:
        pass


class GenericToolCall(LLMToolCall, dict):
    def __init__(self, tool_call_id: str, name: str, arguments: Dict[str, Any]):
        dict.__init__(self, id=tool_call_id, name=name, arguments=arguments)
        self.id = tool_call_id
        self.name = name
        self.arguments = arguments

    def get_tool_call_id(self) -> str:
        return self.id

    def get_tool_name(self) -> str:
        return self.name

    def get_tool_call_arguments(self) -> Dict[str, Any]:
        return self.arguments

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }


class TemplateToolCallHandler(LLMToolCallHandler):
    """
    A customizable tool call handler that can be configured with different patterns and formats
    for tool call detection, parsing, and message formatting.
    """

    def __init__(
            self,
            tool_call_pattern: str = r'\[\s*{\s*"name":\s*"[^"]+"\s*,\s*"arguments":\s*{[^}]+}\s*}(?:\s*,\s*{\s*"name":\s*"[^"]+"\s*,\s*"arguments":\s*{[^}]+}\s*})*\s*\]',
            tool_name_field: str = "name",
            arguments_field: str = "arguments",
            debug_mode: bool = False
    ):
        """
        Initialize the template tool call handler with customizable patterns and field names.

        Args:
            tool_call_pattern: Regex pattern to identify tool calls in the response
            tool_name_field: Field name for the tool/function name
            arguments_field: Field name for the arguments
            debug_mode: Enable debug output
        """
        self.tool_call_pattern = re.compile(tool_call_pattern, re.DOTALL)
        self.tool_name_field = tool_name_field
        self.arguments_field = arguments_field
        self.debug = debug_mode

    def contains_tool_calls(self, response: str) -> bool:
        """Check if the response contains tool calls using the configured pattern."""
        matches = self.tool_call_pattern.findall(response.strip())
        if not matches:
            return False
        return True

    def parse_tool_calls(self, response: str) ->[ List[GenericToolCall], bool]:
        """Parse tool calls from the response using the configured patterns."""
        if self.debug:
            print(f"Parsing response: {response}", flush=True)

        tool_calls = []
        matches = self.tool_call_pattern.findall(response.strip())

        for match in matches:
            try:
                # Parse the JSON content
                parsed = json.loads(match)
                # Handle both single tool call and array formats
                if isinstance(parsed, list):
                    calls = parsed
                else:
                    calls = [parsed]

                # Create GenericToolCall objects for valid calls
                for call in calls:
                    tool_calls.append(
                        GenericToolCall(
                            tool_call_id=generate_id(length=9),
                            name=call[self.tool_name_field],
                            arguments=call[self.arguments_field]
                        )
                    )

            except json.JSONDecodeError as e:
                if self.debug:
                    print(f"Failed to parse tool call: {str(e)}", flush=True)
                return "Failed to parse tool call", False

        return tool_calls, True

    def get_tool_call_messages(
            self,
            tool_calls: List[GenericToolCall]
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Convert tool calls to assistant messages."""
        tool_call_messages = []
        for tool_call in tool_calls:
            tool_call_dict = {
                self.tool_name_field: tool_call.get_tool_name(),
                self.arguments_field: tool_call.get_tool_call_arguments()
            }
            tool_call_messages.append(tool_call_dict)

        return Message(
            role="assistant",
            content=json.dumps(tool_call_messages)
        ).to_dict()

    def get_tool_call_result_messages(
            self,
            tool_calls: List[GenericToolCall],
            tool_call_results: List[Any]
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Convert tool call results to tool messages."""
        return [
            Message(
                role="tool",
                content=str(tool_call_result) if not isinstance(tool_call_result, dict)
                else json.dumps(tool_call_result),
                tool_call_id=tool_call.get_tool_call_id()
            ).to_dict()
            for tool_call, tool_call_result in zip(tool_calls, tool_call_results)
        ]

    def execute_tool_calls(
            self,
            tool_calls: List[GenericToolCall],
            tool_registry: ToolRegistry
    ) -> List[Any]:
        """Execute the tool calls using the provided tool registry."""
        results = []
        for tool_call in tool_calls:
            tool = tool_registry.get_tool(tool_call.get_tool_name())
            call_parameters = tool_call.get_tool_call_arguments()
            output = tool.execute(call_parameters)
            results.append(output)
        return results

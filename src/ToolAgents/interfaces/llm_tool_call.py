import abc
import random
import string
from typing import Dict, Any, List

from ToolAgents.function_tool import ToolRegistry


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
    def parse_tool_calls(self, response: str) -> List[LLMToolCall]:
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


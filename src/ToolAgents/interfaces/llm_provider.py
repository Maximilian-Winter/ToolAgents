import abc
from typing import List, Dict, Optional, Generator, Any, Union

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.messages.chat_message import ChatMessage

class StreamingChatAPIResponse:
    """
    Represents a streaming chat API response.
    """
    def __init__(self, chunk: str, is_tool_call: bool = False, partial_tool_call: Dict[str, Any] = None, finished: bool = False, finished_chat_message: Optional[ChatMessage] = None):
        self.chunk = chunk
        self.is_tool_call = is_tool_call
        self.partial_tool_call = partial_tool_call
        self.finished = finished
        self.finished_chat_message = finished_chat_message

    def get_chunk(self) -> str:
        return self.chunk

    def get_is_tool_call(self) -> bool:
        return self.is_tool_call

    def get_partial_tool_call(self) -> Dict[str, Any]:
        return self.partial_tool_call

    def get_finished(self) -> bool:
        return self.finished

    def get_finished_chat_message(self) -> Union[ChatMessage, None]:
        return self.finished_chat_message

class LLMSamplingSettings(abc.ABC):

    @abc.abstractmethod
    def save_to_file(self, settings_file: str):
        pass

    @abc.abstractmethod
    def load_from_file(self, settings_file: str):
        pass

    @abc.abstractmethod
    def as_dict(self):
        pass

    @abc.abstractmethod
    def set_stop_tokens(self, tokens: List[str], tokenizer):
        pass

    @abc.abstractmethod
    def set_max_new_tokens(self, max_new_tokens: int):
        pass

    @abc.abstractmethod
    def set(self, setting_key: str, setting_value: Any):
        pass

    @abc.abstractmethod
    def neutralize_sampler(self, sampler_name: str):
        pass

    @abc.abstractmethod
    def neutralize_all_samplers(self):
        pass

    @abc.abstractmethod
    def set_response_format(self, response_format: dict[str, Any]):
        pass

    @abc.abstractmethod
    def set_extra_body(self, extra_body: dict[str, Any]):
        pass

    @abc.abstractmethod
    def set_extra_request_kwargs(self, **kwargs):
        pass

    @abc.abstractmethod
    def set_tool_choice(self, tool_choice: str):
        pass

class ChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    def get_default_settings(self):
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings) -> None:
        pass

    @abc.abstractmethod
    def get_response(self, messages: List[Dict[str, Any]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        pass

    @abc.abstractmethod
    def get_streaming_response(self, messages: List[Dict[str, Any]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[StreamingChatAPIResponse, None, None]:
        pass

    @abc.abstractmethod
    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        pass
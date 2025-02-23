import abc
import copy
import json
from typing import List, Dict, Optional, Generator, Any, Union

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.messages.chat_message import ChatMessage
from pydantic import BaseModel
from typing import Dict, Optional, Any, Union

from ToolAgents.messages.chat_message import ChatMessage

class StreamingChatMessage(BaseModel):
    """
    Represents a streaming chat API response.
    """
    chunk: str
    is_tool_call: bool = False
    tool_call: Optional[Dict[str, Any]] = None
    finished: bool = False
    finished_chat_message: Optional[ChatMessage] = None

    def get_chunk(self) -> str:
        return self.chunk

    def get_is_tool_call(self) -> bool:
        return self.is_tool_call

    def get_tool_call(self) -> Dict[str, Any]:
        return self.tool_call

    def get_finished(self) -> bool:
        return self.finished

    def get_finished_chat_message(self) -> Union[ChatMessage, None]:
        return self.finished_chat_message

    class Config:
        arbitrary_types_allowed = True  # To allow ChatMessage custom type

class ProviderSettings(abc.ABC):
    def __init__(self, initial_tool_choice: Union[str, dict], : dict[str, (Any, Any)]):
        self.extra_body = None
        self.response_format = None
        self.request_kwargs = {}
        self.tool_choice = initial_tool_choice
        self.max_tokens = 512
        self.stop_sequences = []
        self.samplers = samplers

    def set(self, setting_key: str, setting_value: str):
        if hasattr(self, setting_key):
            setattr(self, setting_key, setting_value)
        else:
            if self.request_kwargs is None:
                self.request_kwargs = {}
            self.request_kwargs[setting_key] = setting_value

    def save_to_file(self, settings_file: str):
        with open(settings_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, settings_file: str):
        with open(settings_file, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)

    def to_dict(self, include: list[str] = None):
        result = {"max_tokens": self.max_tokens, "stop_sequences": self.stop_sequences}
        if self.extra_body:
            result['extra_body'] = self.extra_body
        if self.response_format:
            result['response_format'] = self.response_format

        if len(self.request_kwargs) > 0:
            for key, value in self.request_kwargs.items():
                result[key] = value

        for key, value in self.samplers.items():
            result[key] = value

        if include is not None:
            for key in filter_out:
                if key in result:
                    result.pop(key)

        return result


    def set_stop_tokens(self, tokens: List[str]):
        self.stop_sequences = tokens

    def set_max_new_tokens(self, max_new_tokens: int):
        self.max_tokens = max_new_tokens

    @abc.abstractmethod
    def neutralize_sampler(self, sampler_name: str):
        pass

    @abc.abstractmethod
    def neutralize_all_samplers(self):
        pass

    def set_response_format(self, response_format: dict[str, Any]):
        self.response_format = response_format

    def set_extra_body(self, extra_body: dict[str, Any]):
        self.extra_body = extra_body

    def set_extra_request_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            self.request_kwargs[key] = value

    def set_tool_choice(self, tool_choice):
        self.tool_choice = tool_choice

class ChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    def get_response(self, messages: List[ChatMessage], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        pass

    @abc.abstractmethod
    def get_streaming_response(self, messages: List[ChatMessage], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatMessage, None, None]:
        pass

    @abc.abstractmethod
    def get_default_settings(self):
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings) -> None:
        pass

    @abc.abstractmethod
    def get_provider_identifier(self) -> str:
        pass

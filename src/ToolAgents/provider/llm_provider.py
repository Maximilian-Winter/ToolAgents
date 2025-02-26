import abc
import copy
import json
from typing import List, Dict, Optional, Generator, Any, Union, AsyncGenerator

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.messages.chat_message import ChatMessage
from pydantic import BaseModel, Field
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


class SamplerSetting(BaseModel):
    name: str = Field(..., title="Sampler name")
    default_value: dict[str, Any] = Field(..., title="Sampler default value")
    neutral_value: dict[str, Any] = Field(..., title="Sampler when neutral (turned off)")
    sampler_value: dict[str, Any] = Field(..., title="Sampler value")
    is_single_value: bool = Field(..., title="Sampler when value is single value")

    @staticmethod
    def create_sampler_setting(name: str, default_value: Any, neutral_value: Any) -> 'SamplerSetting':
        if isinstance(default_value, dict) and isinstance(neutral_value, dict):
            return SamplerSetting(name=name, default_value=default_value, neutral_value=neutral_value, sampler_value=default_value, is_single_value=False)
        elif not isinstance(default_value, dict) and not isinstance(neutral_value, dict):
            return SamplerSetting(name=name, default_value={name: default_value}, neutral_value={name: neutral_value}, sampler_value={name: default_value}, is_single_value=True)
        else:
            raise RuntimeError(f"Wrong default and neutral value types, has to be either both dict or not!\nDefault Value Type:{default_value}\nNeutral Value:{neutral_value}")

    def get_sampler_name(self) -> str:
        return self.name

    def get_default_value(self) -> Any:
        if self.is_single_value:
            return self.default_value[self.name]
        else:
            return self.sampler_default_value

    def get_neutral_value(self) -> Any:
        if self.is_single_value:
            return self.sampler_neutral_value[self.name]
        else:
            return self.sampler_neutral_value

    def get_value(self) -> Any:
        if self.is_single_value:
            return self.sampler_value[self.name]
        else:
            return self.sampler_value

    def set_value(self, value: Any) -> None:
        if self.is_single_value:
            self.sampler_value[self.name] = value
        else:
            self.sampler_value = value

    def reset(self) -> None:
        self.sampler_value = self.default_value

    def neutralize(self) -> None:
        self.sampler_value = self.neutral_value

class ProviderSettings(abc.ABC):
    def __init__(self, initial_tool_choice: Union[str, dict], samplers: List[SamplerSetting]):
        self.extra_body = None
        self.response_format = None
        self.request_kwargs = {}
        self.tool_choice = initial_tool_choice
        self.max_tokens = 4096
        self.stop_sequences = []
        self.samplers = {}
        for sampler in samplers:
            self.samplers[sampler.name] = sampler

    def set(self, setting_key: str, setting_value: Any):
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

    def to_dict(self, include: list[str] = None, filter_out: list[str] = None) -> dict[str, Any]:
        result = {"max_tokens": self.max_tokens, "stop_sequences": self.stop_sequences, "tool_choice": self.tool_choice}
        if self.extra_body:
            result['extra_body'] = self.extra_body
        if self.response_format:
            result['response_format'] = self.response_format

        if len(self.request_kwargs) > 0:
            for key, value in self.request_kwargs.items():
                result[key] = value

        for key, value in self.samplers.items():
            result[key] = value.get_value()

        filtered = {}
        if include is not None:
            for key in include:
                if key in result:
                    filtered[key] = result[key]
            result = filtered
        if filter_out is not None:
            for key in filter_out:
                if key in result:
                    result.pop(key)

        return result


    def set_stop_tokens(self, tokens: List[str]):
        self.stop_sequences = tokens

    def set_max_new_tokens(self, max_new_tokens: int):
        self.max_tokens = max_new_tokens

    def set_sampler_value(self, sampler_name: str, sampler_value: Any):
        self.samplers[sampler_name].set_value(sampler_value)

    def reset_sampler_value(self, sampler_name: str):
        self.samplers[sampler_name].reset()

    def reset_all_samplers(self):
        for sampler in self.samplers.values():
            sampler.reset()

    def neutralize_sampler(self, sampler_name: str):
        self.samplers[sampler_name].neutralize()

    def neutralize_all_samplers(self):
        for sampler in self.samplers.values():
            sampler.neutralize()

    def set_response_format(self, response_format: dict[str, Any]):
        self.response_format = response_format

    def set_extra_body(self, extra_body: dict[str, Any]):
        self.extra_body = extra_body

    def set_extra_request_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            self.request_kwargs[key] = value

    def set_tool_choice(self, tool_choice):
        self.tool_choice = tool_choice

    def __getattr__(self, name):
        if name in ["max_tokens", "stop_sequences", "tool_choice", "extra_body", "response_format", "request_kwargs",
                    "samplers"]:
            # Access the underlying dict to avoid recursion
            return super().__getattribute__(name)
        else:
            if name in self.request_kwargs:
                return self.request_kwargs[name]
            elif name in self.samplers:
                return self.samplers[name].get_value()
            else:
                raise RuntimeError(f"Setting attribute {name} not found.")

    def __setattr__(self, name, value):
        if name in ["max_tokens", "stop_sequences", "tool_choice", "extra_body", "response_format", "request_kwargs",
                    "samplers"]:
            # Use object.__setattr__ to bypass this method for core attributes
            super().__setattr__(name, value)
        else:
            # Make sure core attributes are initialized
            if not hasattr(self, "request_kwargs") or not hasattr(self, "samplers"):
                super().__setattr__(name, value)
                return

            if name in self.request_kwargs:
                self.request_kwargs[name] = value
            elif name in self.samplers:
                self.samplers[name].set_value(value)
            else:
                raise RuntimeError(f"Setting attribute {name} not found.")

class ChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    def get_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        pass

    @abc.abstractmethod
    def get_streaming_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
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


class AsyncChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    async def get_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        pass

    @abc.abstractmethod
    async def get_streaming_response(self, messages: List[ChatMessage], settings: ProviderSettings=None,
                               tools: Optional[List[FunctionTool]] = None) -> AsyncGenerator[
        StreamingChatMessage, None]:
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
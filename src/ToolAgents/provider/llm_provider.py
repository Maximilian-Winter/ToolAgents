import abc
from typing import List, Dict, Optional, Generator, Any, Union

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.messages.chat_message import ChatMessage
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Union

from ToolAgents.messages.chat_message import ChatMessage

class StreamingChatAPIResponse(BaseModel):
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
            return SamplerSetting(name=name, default_value={name: default_value}, neutral_value={name: neutral_value}, sampler_value=default_value, is_single_value=True)
        else:
            raise RuntimeError(f"Wrong default and neutral value types, has to be either both dict or not!\nDefault Value Type:{default_value}\nNeutral Value:{neutral_value}")

    def get_sampler_name(self) -> str:
        return self.name

    def get_default_value(self) -> Any:
        if self.is_single_value:
            return self.default_value[self.sampler_name]
        else:
            return self.sampler_default_value

    def get_neutral_value(self) -> Any:
        if self.is_single_value:
            return self.sampler_neutral_value[self.sampler_name]
        else:
            return self.sampler_neutral_value

    def get_value(self) -> Any:
        if self.is_single_value:
            return self.sampler_value[self.sampler_name]
        else:
            return self.sampler_value

    def set_value(self, value: Any) -> None:
        if self.is_single_value:
            self.sampler_value[self.sampler_name] = value
        else:
            self.sampler_value = value

    def reset(self) -> None:
        self.sampler_value = self.default_value

    def neutralize(self) -> None:
        self.sampler_value = self.neutral_value

class ChatAPISettings(BaseModel):
    general_settings: Dict[str, Any] = Field(..., description="The settings of the LLM provider.")
    samplers: Dict[str, SamplerSetting] = Field(..., description="The settings of the sampler.")

    def get_general_settings(self) -> Dict[str, Any]:
        return self.general_settings

    def get_sampler_settings(self) -> Dict[str, SamplerSetting]:
        return self.samplers

    def set_general_setting(self, settings_name: str, setting_value: Any) -> None:
        self.general_settings[settings_name] = setting_value

    def set_sampler(self, sampler_name: str, sampler_value: Any) -> None:
        self.samplers[sampler_name].set_value(sampler_value)

    def neutralize_all_samplers(self) -> None:
        for name, sampler in self.samplers.items():
            sampler.neutralize()

    def reset_all_samplers(self) -> None:
        for name, sampler in self.samplers.items():
            sampler.reset()

    def neutralize_sampler(self, sampler_name: str) -> None:
        self.samplers[sampler_name].neutralize()

    def reset_sampler(self, sampler_name: str) -> None:
        self.samplers[sampler_name].reset()

    def get_combined_dict(self) -> Dict[str, Any]:
        result = {}
        for name, setting in self.general_settings.items():
            result[name] = setting
        for name, sampler in self.samplers.items():
            result[name] = sampler.get_value()

        return result

    def __getattr__(self, name: str):
        try:
            if name in self.model_fields():
                return super().__getattr__(name)
            else:
                data = self.get_combined_dict()
                return data[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name in self.model_fields():
            super().__setattr__(name, value)
        else:
            if name in self.samplers:
                self.samplers[name].set_value(value)
                return None
            self.set_general_setting(name, value)

class ChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    def get_default_settings(self) -> ChatAPISettings:
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings: ChatAPISettings) -> None:
        pass

    @abc.abstractmethod
    def get_response(self, messages: List[Dict[str, Any]], settings: ChatAPISettings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        pass

    @abc.abstractmethod
    def get_streaming_response(self, messages: List[Dict[str, Any]], settings: ChatAPISettings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[StreamingChatAPIResponse, None, None]:
        pass

    @abc.abstractmethod
    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_provider_identifier(self) -> str:
        pass

import abc
from dataclasses import dataclass
from typing import List, Dict, Optional, Generator, Any

from ToolAgents import ToolRegistry, FunctionTool

@dataclass
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
    def set(self, setting_key: str, setting_value: str):
        pass

    @abc.abstractmethod
    def neutralize_sampler(self, sampler_name: str):
        pass

    @abc.abstractmethod
    def neutralize_all_samplers(self):
        pass


class HostedLLMProvider(abc.ABC):
    @abc.abstractmethod
    def get_default_settings(self) -> LLMSamplingSettings:
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings: LLMSamplingSettings) -> None:
        pass

    @abc.abstractmethod
    def create_completion(self, prompt, settings: LLMSamplingSettings, tool_registry: ToolRegistry = None):
        pass

    @abc.abstractmethod
    def create_chat_completion(self, messages: List[Dict[str, str]], settings: LLMSamplingSettings, tool_registry: ToolRegistry = None):
        pass



class ChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    def get_default_settings(self):
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings) -> None:
        pass

    @abc.abstractmethod
    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> str:
        pass

    @abc.abstractmethod
    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[str, None, None]:
        pass

    @abc.abstractmethod
    def generate_tool_response_message(self, tool_call_id: str, tool_name: str, tool_response: str) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def generate_tool_use_message(self, content: str, tool_call_id: str, tool_name: str, tool_args: str) -> Dict[
        str, Any]:
        pass


import abc
from typing import List, Dict

from ToolAgents import ToolRegistry


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
    def create_completion(self, prompt, settings: LLMSamplingSettings, tool_registry: ToolRegistry = None):
        pass

    @abc.abstractmethod
    def create_chat_completion(self, messages: List[Dict[str, str]], settings: LLMSamplingSettings, tool_registry: ToolRegistry = None):
        pass

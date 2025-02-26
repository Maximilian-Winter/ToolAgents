import abc
import random
import string
from typing import List, Dict, Union, Generator, AsyncGenerator

from ToolAgents.messages import ToolCallContent, TextContent
from ToolAgents.provider import ProviderSettings


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))

class LLMTokenizer(abc.ABC):
    @abc.abstractmethod
    def apply_template(self, messages: List[Dict[str, str]], tools: List) -> str:
        pass

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass

    @abc.abstractmethod
    def get_eos_token_string(self) -> str:
        pass

class CompletionEndpoint(abc.ABC):
    @abc.abstractmethod
    def create_completion(self, prompt, settings: ProviderSettings)-> str:
        pass
    @abc.abstractmethod
    def create_streaming_completion(self, prompt, settings: ProviderSettings) ->  Generator[str, None, None]:
        pass
    @abc.abstractmethod
    def get_default_settings(self):
        pass

class AsyncCompletionEndpoint(abc.ABC):
    @abc.abstractmethod
    async def create_completion(self, prompt, settings: ProviderSettings)-> str:
        pass
    @abc.abstractmethod
    async def create_streaming_completion(self, prompt, settings: ProviderSettings) -> AsyncGenerator[str, None]:
        pass
    @abc.abstractmethod
    def get_default_settings(self):
        pass


class LLMToolCallHandler(abc.ABC):

    @abc.abstractmethod
    def contains_partial_tool_calls(self, response: str) -> bool:
        pass

    @abc.abstractmethod
    def contains_tool_calls(self, response: str) -> bool:
        pass

    @abc.abstractmethod
    def parse_tool_calls(self, response: str) -> List[Union[ToolCallContent, TextContent]]:
        pass

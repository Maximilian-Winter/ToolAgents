# message_converter.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Optional, AsyncGenerator

from ToolAgents import FunctionTool
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider.llm_provider import StreamingChatMessage, ProviderSettings


class BaseMessageConverter(ABC):
    @abstractmethod
    def prepare_request(self, model: str, messages: List[ChatMessage], settings: ProviderSettings = None,
                         tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        """
        Prepare the request for the provider
        Returns:
            Request arguments as dictionary
        """
        pass

    @abstractmethod
    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert a list of universal ChatMessages to the provider-specific message format.
        """
        pass

class BaseResponseConverter(ABC):

    @abstractmethod
    def from_provider_response(self, response_data: Any) -> ChatMessage:
        """
        Convert provider-specific response data into a universal ChatMessage.
        """
        pass

    @abstractmethod
    def yield_from_provider(self, stream_generator: Any) -> Generator[StreamingChatMessage, None, None]:
        """
        Yield a universal StreamingChatAPIResponse from the provider-specific response.
        """
        pass

    @abstractmethod
    async def async_yield_from_provider(self, stream_generator: Any) -> AsyncGenerator[StreamingChatMessage, None]:
        pass
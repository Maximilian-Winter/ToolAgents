# message_converter.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider.llm_provider import StreamingChatMessage


class BaseMessageConverter(ABC):
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
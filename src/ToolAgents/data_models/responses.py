from enum import Enum
from types import NoneType
from typing import List, Any, Union

from pydantic import BaseModel, Field

from ToolAgents.data_models.messages import ChatMessage


class ChatResponse(BaseModel):
    """
    Represents an agent chat response.
    """

    messages: List[ChatMessage] = Field(
        default_factory=list, description="List of chat messages."
    )
    response: str = Field(
        default_factory=str, description="Final response from the agent."
    )


class ChatResponseChunk(BaseModel):
    """
    Represents an agent chat response chunk.
    """

    chunk: str = Field(
        default_factory=str, description="Response chunk from the agent."
    )

    has_tool_call: bool = Field(
        default_factory=bool, description="Whether the chunk has a tool call."
    )
    tool_call: dict[str, Any] | None = Field(
        default_factory=NoneType, description="Tool call data."
    )

    has_tool_call_result: bool = Field(
        default_factory=bool, description="Whether the chunk has a tool call result."
    )
    tool_call_result: dict[str, Any] = Field(
        default_factory=NoneType, description="The result of the tool call."
    )

    finished: bool = Field(
        default_factory=bool, description="Whether the response has been completed."
    )
    finished_response: ChatResponse = Field(
        default_factory=ChatResponse,
        description="Finished response object from the agent.",
    )

    def get_tool_name(self) -> Union[str, None]:
        if self.has_tool_call:
            if "tool_call_name" in self.tool_call:
                return self.tool_call["tool_call_name"]
            return None
        else:
            return None

    def get_tool_arguments(self) -> Union[dict[str, Any], None]:
        if self.has_tool_call:
            if "tool_call_arguments" in self.tool_call:
                return self.tool_call["tool_call_arguments"]
            return None
        else:
            return None

    def get_tool_results(self) -> Union[str, None]:
        if self.has_tool_call_result:
            if "tool_call_result" in self.tool_call_result:
                return self.tool_call_result["tool_call_result"]
            return None
        else:
            return None
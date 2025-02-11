from enum import Enum
from typing import List, Union, Dict, Any

from pydantic import BaseModel, Field


class ChatMessageRole(str, Enum):
    """
    Enum representing the role of a chat message sender.
    """
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"


class ContentType(str, Enum):
    """
    Enum representing the type of content in a message.
    """
    Text = "text"
    Binary = "binary"
    ToolCall = "tool_call"
    ToolCallResult = "tool_call_result"


class BinaryStorageType(str, Enum):
    """
    Enum representing the storage method used for binary content.
    """
    file = "file"
    base64 = "base64"
    external = "external"


class TextContent(BaseModel):
    """
    Model for text-based content.

    Attributes:
        content_type: Always 'text' for text content.
        content: The actual text content.
    """
    content_type: ContentType = Field(
        ContentType.Text,
        description="The content type, always 'text' for TextContent."
    )
    content: str = Field(..., description="The text content.")


class BinaryContent(BaseModel):
    """
    Model for binary content which is stored using different storage methods.

    Attributes:
        content_type: Always 'binary' for binary content.
        storage_type: Indicates how the binary content is stored (e.g., file, base64, external).
        mime_type: The MIME type of the binary content.
        content: The actual binary data (typically encoded as base64 if storage_type is 'base64').
        additional_information: Extra metadata or information related to the binary content.
    """
    content_type: ContentType = Field(
        ContentType.Binary,
        description="The content type, always 'binary' for BinaryContent."
    )
    storage_type: BinaryStorageType = Field(
        ...,
        description="The storage type for the binary content (e.g., file, base64, external)."
    )
    mime_type: str = Field(..., description="The MIME type of the binary content.")
    content: str = Field(..., description="The binary content, typically base64 encoded if storage_type is 'base64'.")
    additional_information: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata or information regarding the binary content."
    )


class ToolCallContent(BaseModel):
    """
    Model for representing a tool call within a chat message.

    Attributes:
        content_type: Always 'tool_call' for a tool call.
        tool_call_id: Unique identifier for the tool call.
        tool_call_name: The name of the tool to be invoked.
        tool_call_arguments: A dictionary of arguments to be passed to the tool.
    """
    content_type: ContentType = Field(
        ContentType.ToolCall,
        description="The content type, always 'tool_call' for ToolCallContent."
    )
    tool_call_id: str = Field(..., description="Unique identifier for the tool call.")
    tool_call_name: str = Field(..., description="The name of the tool to be called.")
    tool_call_arguments: Dict[str, Any] = Field(
        ...,
        description="Arguments for the tool call."
    )


class ToolCallResult(BaseModel):
    """
    Model for representing the result of a tool call.

    Attributes:
        content_type: Always 'tool_call_result' for a tool call result.
        tool_call_result_id: Unique identifier for the tool call result.
        tool_call_id: Unique identifier for the corresponding tool call.
        tool_call_name: The name of the tool that produced the result.
        tool_call_result: The result data from the tool call.
    """
    content_type: ContentType = Field(
        ContentType.ToolCallResult,
        description="The content type, always 'tool_call_result' for ToolCallResult."
    )
    tool_call_result_id: str = Field(..., description="Unique identifier for the tool call result.")
    tool_call_id: str = Field(..., description="Unique identifier for the corresponding tool call.")
    tool_call_name: str = Field(..., description="The name of the tool that produced the result.")
    tool_call_result: Dict[str, Any] = Field(
        ...,
        description="The result data from the tool call."
    )



class ChatMessage(BaseModel):
    """
    Model representing a chat message in the messaging protocol.

    Attributes:
        id: Unique identifier for the chat message.
        role: The role of the message sender (e.g., system, user, assistant, tool).
        content: A list of message content objects that make up the chat message.
    """
    id: str = Field(
        ...,
        description="Unique identifier for the chat message."
    )
    role: ChatMessageRole = Field(
        ...,
        description="The role of the message sender (e.g., system, user, assistant, tool)."
    )
    content: List[Union[TextContent, BinaryContent, ToolCallContent, ToolCallResult]] = Field(
        ...,
        description="A list of content objects that comprise the chat message."
    )

example = {
    "id": "1",
    "role": "assistant",
    "content": [
      {
        "content_type": "text",
        "content": "I'll help you perform all these tasks. Let me break this down into multiple function calls:\n\n1. First, let's get the weather for all three locations in celsius:"
      },
      {
        "content_type": "tool_call",
        "tool_call_id": "toolu_01WCpS9wxURWdbtUwU3UPvqR",
        "tool_call_name": "get_current_weather",
        "tool_call_arguments": {
          "location": "London, Great Britain",
          "unit": "celsius"
        }
      }
    ]
}

chat_message = ChatMessage(**example)
print(chat_message.model_dump_json(indent=4))
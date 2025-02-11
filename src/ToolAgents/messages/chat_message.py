from enum import Enum
from typing import List, Union, Dict, Any

from pydantic import BaseModel, Field


class ChatMessageRole(str, Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"

class ContentType(str, Enum):
    Text = "text"
    Binary = "binary"
    ToolCall = "tool_call"
    ToolCallResult = "tool_call_result"

class BinaryStorageType(str, Enum):
    file = "file"
    base64 = "base64"
    external = "external"

class TextContent(BaseModel):
    content_type: ContentType = Field(ContentType.Text)
    content: str = Field(...)

class BinaryContent(BaseModel):
    content_type: ContentType = Field(ContentType.Binary)
    storage_type: BinaryStorageType = Field(...)
    mime_type: str = Field(...)
    content: str = Field(...)
    additional_information: Dict[str, str] = Field(...)

class ToolCallContent(BaseModel):
    content_type: ContentType = Field(ContentType.ToolCall)
    tool_call_id: str = Field(...)
    tool_name: str = Field(...)
    tool_call_arguments: Dict[str, Any] = Field(...)

class ToolCallResult(BaseModel):
    content_type: ContentType = Field(ContentType.ToolCallResult)
    tool_call_result_id: str = Field(...)
    tool_call_name: str = Field(...)
    tool_call_result: Dict[str, Any] = Field(...)

class MessageContent(BaseModel):
    content_type: ContentType = Field(..., description="Content Type")
    content: Union[TextContent, BinaryContent, ToolCallContent, ToolCallResult] = Field("", description="Content")

class ChatMessage(BaseModel):
    chat_message_id: str = Field(..., description="Chat message ID")
    role: ChatMessageRole = Field(..., description="The role of the message sender")
    content: List[MessageContent] = Field(..., description="Content")
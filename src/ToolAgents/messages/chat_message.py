import base64
import datetime
import pathlib
import sys
import uuid
from enum import Enum
from os import PathLike
from typing import List, Union, Dict, Any

import httpx
from pydantic import BaseModel, Field


class ChatMessageRole(str, Enum):
    """
    Enum representing the role of a chat message sender.
    """
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"
    Custom = "custom"

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
    Enum representing the storage method used for binary content. Url for links and Base64 for raw file data.
    """
    Url = "url"
    Base64 = "base64"

class TextContent(BaseModel):
    """
    Model for text-based content.

    Attributes:
        type: Always 'text' for text content.
        content: The actual text content.
    """
    type: ContentType = Field(
        default=ContentType.Text,
        description="The content type, always 'text' for TextContent."
    )
    content: str = Field(..., description="The text content.")


class BinaryContent(BaseModel):
    """
    Model for binary content which is stored using different storage methods.

    Attributes:
        type: Always 'binary' for binary content.
        storage_type: The binary content, base64 encoded if storage_type is 'base64'. Or url string if storage_type is 'url'.
        mime_type: The MIME type of the binary content.
        content: The actual binary data (typically encoded as base64 if storage_type is 'base64').
        additional_information: Extra metadata or information related to the binary content.
    """
    type: ContentType = Field(
        default=ContentType.Binary,
        description="The content type, always 'binary' for BinaryContent."
    )
    storage_type: BinaryStorageType = Field(
        ...,
        description="The storage type for the binary content (e.g., url, base64)."
    )
    mime_type: str = Field(..., description="The MIME type of the binary content.")
    content: str = Field(..., description="The binary content, base64 encoded if storage_type is 'base64'. Or url string if storage_type is 'url'.")
    additional_information: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata or information regarding the binary content."
    )


class ToolCallContent(BaseModel):
    """
    Model for representing a tool call within a chat message.

    Attributes:
        type: Always 'tool_call' for a tool call.
        tool_call_id: Unique identifier for the tool call.
        tool_call_name: The name of the tool to be invoked.
        tool_call_arguments: A dictionary of arguments to be passed to the tool.
    """
    type: ContentType = Field(
        default=ContentType.ToolCall,
        description="The content type, always 'tool_call' for ToolCallContent."
    )
    tool_call_id: str = Field(..., description="Unique identifier for the tool call.")
    tool_call_name: str = Field(..., description="The name of the tool to be called.")
    tool_call_arguments: Dict[str, Any] = Field(
        ...,
        description="Arguments for the tool call."
    )


class ToolCallResultContent(BaseModel):
    """
    Model for representing the result of a tool call.

    Attributes:
        type: Always 'tool_call_result' for a tool call result.
        tool_call_result_id: Unique identifier for the tool call result.
        tool_call_id: Unique identifier for the corresponding tool call.
        tool_call_name: The name of the tool that produced the result.
        tool_call_result: The result data from the tool call.
    """
    type: ContentType = Field(
        default=ContentType.ToolCallResult,
        description="The content type, always 'tool_call_result' for ToolCallResult."
    )
    tool_call_result_id: str = Field(..., description="Unique identifier for the tool call result.")
    tool_call_id: str = Field(..., description="Unique identifier for the corresponding tool call.")
    tool_call_name: str = Field(..., description="The name of the tool that produced the result.")
    tool_call_result: str = Field(
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
        created_at: The date and time the message was created.
        updated_at: The date and time the message was last updated.
        additional_information: Extra metadata or information related to the chat message.
    """
    id: str = Field(
        ...,
        description="Unique identifier for the chat message."
    )
    role: ChatMessageRole = Field(
        ...,
        description="The role of the message sender (e.g., system, user, assistant, tool)."
    )
    content: List[Union[TextContent, BinaryContent, ToolCallContent, ToolCallResultContent]] = Field(
        ...,
        description="A list of content objects that comprise the chat message."
    )
    created_at: datetime.datetime = Field(..., description="The creation date of the chat message.")

    updated_at: datetime.datetime = Field(..., description="The last update date of the chat message.")

    additional_fields: Dict[str, Any] = Field(default_factory=dict,description="Additional fields for the chat message. For provider specific features, like caching")

    additional_information: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata or information related to the chat message."
    )

    @staticmethod
    def create_system_message(message: str):
        date = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.System, content=[TextContent(content=message)], created_at=date, updated_at=date, additional_information={})

    @staticmethod
    def create_user_message(message: str):
        date = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.User, content=[TextContent(content=message)], created_at=date, updated_at=date,
                           additional_information={})

    @staticmethod
    def create_assistant_message(message: str):
        date = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=[TextContent(content=message)], created_at=date, updated_at=date,
                           additional_information={})

    @staticmethod
    def create_empty_system_message():
        date = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.System, content=[], created_at=date, updated_at=date,
                           additional_information={})

    @staticmethod
    def create_empty_user_message():
        date = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.User, content=[], created_at=date, updated_at=date,
                           additional_information={})

    @staticmethod
    def create_empty_assistant_message():
        date = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=[], created_at=date, updated_at=date,
                           additional_information={})

    @staticmethod
    def from_dictionaries(messages: List[Dict[str, str]]) -> List['ChatMessage']:
        """
        To convert a list of dictionaries into a list of ChatMessage objects. Only works with simple text messages, that have a role string and content string.
        :param messages: The messages to convert.
        :return: Returns a list of ChatMessage objects.
        """
        converted_messages = []
        for message in messages:
            date = datetime.datetime.now()
            converted_messages.append(ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole(message['role']), content=[TextContent(content=message['content'])], created_at=date, updated_at=date))

        return converted_messages

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ChatMessage':
        """
        To convert a dictionary into a ChatMessage object.
        :param data: The data to convert.
        :return: A ChatMessage object.
        """
        return ChatMessage(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts a ChatMessage object to a dictionary.
        :return: The data as a dictionary.
        """
        return self.model_dump()

    def contains_tool_call(self) -> bool:
        """
        Returns True if the chat message contains a tool call.
        :return: If the chat message contains a tool call.
        """
        for content in self.content:
            if content.type == ContentType.ToolCall:
                return True
        return False

    def get_tool_calls(self) -> List[ToolCallContent]:
        """
        Returns a list of ToolCallContent objects.
        :return: Returns a list of ToolCallContent objects.
        """
        result: List[ToolCallContent] = []
        for content in self.content:
            if isinstance(content, ToolCallContent):
                result.append(content)
        return result

    def get_text_content(self) -> str:
        result = []
        for content in self.content:
            if isinstance(content, TextContent):
                result.append(content.content)

        return '\n'.join(result)

    def set_custom_role(self, custom_role_name: str) -> None:
        self.role = ChatMessageRole.Custom
        self.additional_information["custom_role_name"] = custom_role_name

    def set_additional_fields(self, additional_fields: Dict[str, Any]) -> None:
        self.additional_fields = additional_fields

    def set_additional_field(self, field: str, value: Any) -> None:
        self.additional_fields[field] = value

    def add_text(self, content: str) -> None:
        self.content.append(TextContent(content=content))

    def add_text_file_data(self, file: PathLike, content_prefix: str = "", content_suffix: str = "") -> None:
        with open(file, "r") as f:
            text = f.read()
        self.content.append(TextContent(content=content_prefix + text + content_suffix))

    def add_image_file_data(self, file: PathLike, image_format: str) -> None:
        self.add_binary_file_data(file, f"image/{image_format}")

    def add_binary_file_data(self, file: PathLike, mime_type: str) -> None:
        with open(file, "rb") as f:
            binary_data = f.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            base64_string = base_64_encoded_data.decode('utf-8')
        self.add_base64_data(base64_string, mime_type)

    def add_image_url(self, url: str, image_format: str) -> None:
        self.content.append(BinaryContent(storage_type=BinaryStorageType.Url, mime_type=f"image/{image_format}", content=url ))

    def download_and_add_image_data(self, url: str, image_format: str) -> None:
        base64_string = base64.b64encode(httpx.get(url).content).decode("utf-8")
        self.add_base64_data(base64_string, f"image/{image_format}")

    def add_base64_data(self, base64_string: str, mime_type: str) -> None:
        self.content.append(BinaryContent(storage_type=BinaryStorageType.Base64, mime_type=mime_type, content=base64_string ))




if __name__ == "__main__":
    example = {
        "id": "1",
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "content": "I'll help you perform all these tasks. Let me break this down into multiple function calls:\n\n1. First, let's get the weather for all three locations in celsius:"
          },
          {
            "type": "tool_call",
            "tool_call_id": "toolu_01WCpS9wxURWdbtUwU3UPvqR",
            "tool_call_name": "get_current_weather",
            "tool_call_arguments": {
              "location": { "lat": 50, "long": 120},
              "unit": "celsius"
            }
          }
        ],
        "created_at": datetime.datetime.now(),
        "updated_at": datetime.datetime.now()
    }

    chat_message = ChatMessage(**example)
    print(chat_message.model_dump_json(indent=4))
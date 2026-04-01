import base64
import datetime
import json
import uuid
from enum import Enum
from os import PathLike
from typing import List, Union, Dict, Any, Optional

import httpx
from pydantic import BaseModel, Field


class ChatMessageRole(str, Enum):
    System    = "system"
    User      = "user"
    Assistant = "assistant"
    Tool      = "tool"
    Custom    = "custom"


class ContentType(str, Enum):
    Text           = "text"
    Binary         = "binary"
    ToolCall       = "tool_call"
    ToolCallResult = "tool_call_result"
    Reasoning      = "reasoning"      # thinking/reasoning blocks from reasoning models


class BinaryStorageType(str, Enum):
    Url    = "url"
    Base64 = "base64"


class ContentBase(BaseModel):
    type:              ContentType      = Field(..., description="The type of content.")
    additional_fields: Dict[str, Any]  = Field(default_factory=dict)


class TextContent(ContentBase):
    type:    ContentType = Field(default=ContentType.Text)
    content: str         = Field(..., description="The text content.")


class BinaryContent(ContentBase):
    type:         ContentType       = Field(default=ContentType.Binary)
    storage_type: BinaryStorageType = Field(...)
    mime_type:    str               = Field(...)
    content:      str               = Field(...)


class ToolCallContent(ContentBase):
    type:                ContentType                    = Field(default=ContentType.ToolCall)
    tool_call_id:        str                            = Field(...)
    tool_call_name:      str                            = Field(...)
    tool_call_arguments: Union[Dict[str, Any], None, str] = Field(...)

    def get_as_text(self) -> str:
        return f"Tool Use: {self.tool_call_name}\nTool Arguments: {json.dumps(self.tool_call_arguments)}"


class ToolCallResultContent(ContentBase):
    type:                ContentType = Field(default=ContentType.ToolCallResult)
    tool_call_result_id: str         = Field(...)
    tool_call_id:        str         = Field(...)
    tool_call_name:      str         = Field(...)
    tool_call_result:    str         = Field(...)

    def get_as_text(self) -> str:
        return f"Tool Use: {self.tool_call_name}\nTool Result: {self.tool_call_result}"


class ReasoningContent(ContentBase):
    """
    A thinking/reasoning block produced by a reasoning model.

    For Anthropic models:
      - thinking:  the plaintext reasoning (may be None for redacted blocks)
      - signature: Anthropic's verification signature (must be passed back
                   unchanged in multi-turn requests)
      - is_redacted: True when Anthropic's safety system redacted the block;
                     in this case redacted_data holds the opaque payload to
                     pass back to the API

    For OpenAI-compatible models (via OpenRouter etc.):
      - thinking:  the plaintext reasoning string from message.reasoning
      - signature / redacted_data: None (not used by these providers)

    Usage:
        # Check for reasoning in a response
        for block in message.content:
            if isinstance(block, ReasoningContent):
                if block.is_redacted:
                    print("(redacted reasoning)")
                else:
                    print("Reasoning:", block.thinking)

        # Round-trip to Anthropic API format
        anthropic_block = reasoning_block.to_anthropic_block()
    """

    type:          ContentType      = Field(default=ContentType.Reasoning)
    thinking:      Optional[str]    = Field(default=None, description="Plaintext reasoning content.")
    signature:     Optional[str]    = Field(default=None, description="Anthropic verification signature.")
    is_redacted:   bool             = Field(default=False, description="True if the block was safety-redacted.")
    redacted_data: Optional[str]    = Field(default=None, description="Opaque payload for redacted blocks.")

    def to_anthropic_block(self) -> Dict[str, Any]:
        """Serialise back to the Anthropic API content block format for multi-turn."""
        if self.is_redacted:
            return {"type": "redacted_thinking", "data": self.redacted_data or ""}
        block: Dict[str, Any] = {"type": "thinking", "thinking": self.thinking or ""}
        if self.signature:
            block["signature"] = self.signature
        return block

    def get_as_text(self) -> str:
        if self.is_redacted:
            return "[redacted reasoning]"
        return f"[reasoning]\n{self.thinking or ''}"


# ─── Token usage ──────────────────────────────────────────────────────────────

class TokenUsage(BaseModel):
    """
    Normalised token usage from any provider.

    details may contain provider-specific extras such as:
      - cache_creation_input_tokens  (Anthropic prompt caching)
      - cache_read_input_tokens      (Anthropic prompt caching)
      - reasoning_tokens             (OpenAI o-series, OpenRouter)
    """
    input_tokens:  int              = Field(default=0)
    output_tokens: int              = Field(default=0)
    total_tokens:  int              = Field(default=0)
    details:       Dict[str, Any]   = Field(default_factory=dict)


# ─── ChatMessage ──────────────────────────────────────────────────────────────

# The Union must list ReasoningContent before the catch-all TextContent so
# that Pydantic's discriminated parsing resolves correctly on the `type` field.
_ContentUnion = Union[
    ReasoningContent,
    TextContent,
    BinaryContent,
    ToolCallContent,
    ToolCallResultContent,
]


class ChatMessage(BaseModel):
    id:                     str                  = Field(...)
    role:                   ChatMessageRole      = Field(...)
    content:                List[_ContentUnion]  = Field(...)
    created_at:             datetime.datetime    = Field(...)
    updated_at:             datetime.datetime    = Field(...)
    additional_fields:      Dict[str, Any]       = Field(default_factory=dict)
    additional_information: Dict[str, Any]       = Field(default_factory=dict)
    token_usage:            Optional[TokenUsage] = Field(default=None)

    # ── Factory helpers ───────────────────────────────────────────────────────

    @staticmethod
    def create_system_message(message: str) -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.System,
                           content=[TextContent(content=message)], created_at=d, updated_at=d)

    @staticmethod
    def create_user_message(message: str) -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.User,
                           content=[TextContent(content=message)], created_at=d, updated_at=d)

    @staticmethod
    def create_assistant_message(message: str) -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant,
                           content=[TextContent(content=message)], created_at=d, updated_at=d)

    @staticmethod
    def create_custom_role_message(message: str, custom_role: str) -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Custom,
                           content=[TextContent(content=message)], created_at=d, updated_at=d,
                           additional_fields={"custom_role": custom_role})

    @staticmethod
    def create_empty_system_message() -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.System,
                           content=[], created_at=d, updated_at=d)

    @staticmethod
    def create_empty_user_message() -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.User,
                           content=[], created_at=d, updated_at=d)

    @staticmethod
    def create_empty_assistant_message() -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant,
                           content=[], created_at=d, updated_at=d)

    @staticmethod
    def create_empty_custom_role_message(custom_role: str) -> "ChatMessage":
        d = datetime.datetime.now()
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Custom,
                           content=[], created_at=d, updated_at=d,
                           additional_fields={"custom_role": custom_role})

    @staticmethod
    def from_dictionaries(messages: List[Dict[str, str]]) -> List["ChatMessage"]:
        converted = []
        for msg in messages:
            d = datetime.datetime.now()
            converted.append(ChatMessage(
                id=str(uuid.uuid4()), role=ChatMessageRole(msg["role"]),
                content=[TextContent(content=msg["content"])], created_at=d, updated_at=d,
            ))
        return converted

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ChatMessage":
        return ChatMessage(**data)

    # ── Instance helpers ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def contains_tool_call(self) -> bool:
        return any(c.type == ContentType.ToolCall for c in self.content)

    def contains_tool_call_results(self) -> bool:
        return any(c.type == ContentType.ToolCallResult for c in self.content)

    def contains_reasoning(self) -> bool:
        return any(c.type == ContentType.Reasoning for c in self.content)

    def get_tool_calls(self) -> List[ToolCallContent]:
        return [c for c in self.content if isinstance(c, ToolCallContent)]

    def get_tool_call_results(self) -> List[ToolCallResultContent]:
        return [c for c in self.content if isinstance(c, ToolCallResultContent)]

    def get_reasoning_blocks(self) -> List[ReasoningContent]:
        return [c for c in self.content if isinstance(c, ReasoningContent)]

    def get_reasoning_text(self) -> Optional[str]:
        """Return concatenated plaintext from all (non-redacted) reasoning blocks."""
        parts = [c.thinking for c in self.get_reasoning_blocks()
                 if not c.is_redacted and c.thinking]
        return "\n\n".join(parts) if parts else None

    def get_as_text(self) -> str:
        parts = []
        for c in self.content:
            if isinstance(c, (TextContent, ToolCallContent, ToolCallResultContent, ReasoningContent)):
                parts.append(c.get_as_text())
            elif isinstance(c, BinaryContent):
                parts.append(f"Binary Content\nMime type: {c.mime_type}")
        return "\n".join(parts)

    def get_role(self) -> str:
        if self.role is ChatMessageRole.Custom:
            return self.additional_fields.get("custom_role") or "Custom"
        return self.role

    def set_custom_role(self, custom_role: str) -> None:
        self.role = ChatMessageRole.Custom
        self.additional_information["custom_role"] = custom_role

    def set_additional_fields(self, additional_fields: Dict[str, Any]) -> None:
        self.additional_fields = additional_fields

    def set_additional_field(self, field: str, value: Any) -> None:
        self.additional_fields[field] = value

    def add_text(self, content: str) -> None:
        self.content.append(TextContent(content=content))

    def add_text_file_data(self, file: PathLike,
                           content_prefix: str = "", content_suffix: str = "") -> None:
        with open(file, "r") as f:
            text = f.read()
        self.content.append(TextContent(content=content_prefix + text + content_suffix))

    def add_image_file_data(self, file: PathLike, image_format: str) -> None:
        self.add_binary_file_data(file, f"image/{image_format}")

    def add_binary_file_data(self, file: PathLike, mime_type: str) -> None:
        with open(file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        self.add_base64_data(b64, mime_type)

    def add_image_url(self, url: str, image_format: str) -> None:
        self.content.append(BinaryContent(
            storage_type=BinaryStorageType.Url, mime_type=f"image/{image_format}", content=url))

    def download_and_add_image_data(self, url: str, image_format: str) -> None:
        b64 = base64.b64encode(httpx.get(url).content).decode("utf-8")
        self.add_base64_data(b64, f"image/{image_format}")

    def add_base64_data(self, base64_string: str, mime_type: str) -> None:
        self.content.append(BinaryContent(
            storage_type=BinaryStorageType.Base64, mime_type=mime_type, content=base64_string))


# ─── StreamingChatMessage ─────────────────────────────────────────────────────

class StreamingChatMessage(BaseModel):
    chunk:                str                        = Field(...)
    is_tool_call:         bool                       = Field(default=False)
    tool_call:            Optional[Dict[str, Any]]   = Field(default=None)
    finished:             bool                       = Field(default=False)
    finished_chat_message:Optional[ChatMessage]      = Field(default=None)

    def get_chunk(self)                  -> str:                      return self.chunk
    def get_is_tool_call(self)           -> bool:                     return self.is_tool_call
    def get_tool_call(self)              -> Dict[str, Any]:           return self.tool_call
    def get_finished(self)               -> bool:                     return self.finished
    def get_finished_chat_message(self)  -> Optional[ChatMessage]:    return self.finished_chat_message

    class Config:
        arbitrary_types_allowed = True

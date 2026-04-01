# anthropic_message_converter.py  (reasoning-aware version)
#
# Changes from original:
#   - AnthropicMessageConverter.prepare_request: accepts optional `thinking`
#     dict in settings extra_body and passes it through to the request
#   - AnthropicResponseConverter.from_provider_response: captures thinking
#     blocks and stores them as ReasoningContent on the ChatMessage
#   - Streaming: handles content_block_start/delta for type="thinking"
#   - to_provider_format: round-trips thinking blocks back for multi-turn
#
# Anthropic thinking block format in responses:
#   {"type": "thinking", "thinking": "...", "signature": "..."}
#   {"type": "redacted_thinking", "data": "..."}  (safety-filtered)
#
# To enable: pass thinking={"type": "enabled", "budget_tokens": 8000}
# as an extra setting, or set it directly on the request via extra_body.

import base64
import uuid
import datetime
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Generator, Optional, AsyncGenerator

import httpx

from .message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    TextContent,
    ToolCallContent,
    BinaryContent,
    BinaryStorageType,
    ToolCallResultContent,
    TokenUsage,
    ReasoningContent,
)
from ToolAgents.provider.llm_provider import StreamingChatMessage, ProviderSettings
from ToolAgents import FunctionTool



def prepare_messages(messages: List[Dict[str, str]]) -> tuple:
    system_message = []
    other_messages = []
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            other_messages.append({
                "role":    message["role"],
                "content": message["content"],
            })
    return system_message, other_messages


# ─── Anthropic message converter ──────────────────────────────────────────────

class AnthropicMessageConverter(BaseMessageConverter):

    def prepare_request(
        self,
        model: str,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Dict[str, Any]:
        system, other_messages = prepare_messages(self.to_provider_format(messages))
        anthropic_tools = (
            [tool.to_anthropic_tool() for tool in tools] if tools else None
        )

        request_kwargs = settings.to_dict()["REQUEST_SETTINGS"]

        # Pull out thinking config if present (not a native Anthropic param name
        # in settings, so we handle it separately)
        thinking = request_kwargs.pop("thinking", None)

        request_kwargs["model"]    = model
        request_kwargs["system"]   = system
        request_kwargs["messages"] = other_messages

        if thinking:
            request_kwargs["thinking"] = thinking

        if anthropic_tools and len(anthropic_tools) > 0:
            request_kwargs["tools"] = anthropic_tools
        else:
            request_kwargs.pop("tool_choice", None)

        return request_kwargs

    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in messages:
            role = message.role.value
            if role == ChatMessageRole.Tool:
                role = ChatMessageRole.User
            new_content = []
            for content in message.content:
                try:
                    current_content = {}
                    if isinstance(content, ReasoningContent):
                        # Round-trip thinking blocks for multi-turn
                        current_content = content.to_anthropic_block()
                    elif isinstance(content, TextContent):
                        current_content.update({"type": "text", "text": content.content})
                    elif isinstance(content, BinaryContent):
                        if "image" in content.mime_type and content.storage_type == BinaryStorageType.Base64:
                            current_content.update({"type": "image", "source": {"type": "base64", "media_type": content.mime_type, "data": content.content}})
                        elif "image" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                            response = httpx.get(content.content)
                            b64 = base64.b64encode(response.content).decode("utf-8")
                            current_content.update({"type": "image", "source": {"type": "base64", "media_type": content.mime_type, "data": b64}})
                        elif "pdf" in content.mime_type and content.storage_type == BinaryStorageType.Base64:
                            current_content.update({"type": "document", "source": {"type": "base64", "media_type": content.mime_type, "data": content.content}})
                        elif "pdf" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                            response = httpx.get(content.content)
                            b64 = base64.b64encode(response.content).decode("utf-8")
                            current_content.update({"type": "document", "source": {"type": "base64", "media_type": content.mime_type, "data": b64}})
                    elif isinstance(content, ToolCallContent):
                        current_content.update({"type": "tool_use", "id": content.tool_call_id, "name": content.tool_call_name, "input": content.tool_call_arguments})
                    elif isinstance(content, ToolCallResultContent):
                        current_content.update({"type": "tool_result", "tool_use_id": content.tool_call_id, "content": content.tool_call_result})
                    if hasattr(content, 'additional_fields') and len(content.additional_fields) > 0:
                        if "cache_control" in content.additional_fields:
                            current_content["cache_control"] = content.additional_fields["cache_control"]
                    if current_content:
                        new_content.append(current_content)
                except Exception as e:
                    print(f"Failed to convert message {message}: {e}")
            if len(new_content) > 0:
                converted_messages.append({"role": role, "content": new_content})
        return converted_messages


# ─── Anthropic response converter ─────────────────────────────────────────────

class AnthropicResponseConverter(BaseResponseConverter):

    def from_provider_response(self, response_data: Any) -> ChatMessage:
        contents = []
        for block in response_data.content:
            if not hasattr(block, "type"):
                continue
            if block.type == "thinking":
                contents.append(ReasoningContent(
                    thinking=getattr(block, "thinking", None),
                    signature=getattr(block, "signature", None),
                ))
            elif block.type == "redacted_thinking":
                contents.append(ReasoningContent(
                    redacted_data=getattr(block, "data", None),
                ))
            elif block.type == "tool_use":
                contents.append(ToolCallContent(
                    tool_call_id=block.id,
                    tool_call_name=block.name,
                    tool_call_arguments=block.input,
                ))
            elif block.type == "text":
                contents.append(TextContent(content=block.text))

        token_usage = None
        if hasattr(response_data, "usage") and response_data.usage is not None:
            usage = response_data.usage
            input_tokens  = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            details = {}
            if getattr(usage, "cache_creation_input_tokens", None):
                details["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
            if getattr(usage, "cache_read_input_tokens", None):
                details["cache_read_input_tokens"] = usage.cache_read_input_tokens
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                details=details,
            )

        return ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=contents,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
            token_usage=token_usage,
        )

    def yield_from_provider(
        self, stream_generator: Any
    ) -> Generator[StreamingChatMessage, None, None]:
        current_tool_call     = None
        current_thinking_text = ""
        current_thinking_sig  = ""
        current_redacted_data = ""
        in_thinking_block     = False
        in_redacted_block     = False
        contents              = []
        content               = None
        has_tool_call         = False
        input_tokens          = 0
        output_tokens         = 0
        usage_details         = {}

        for chunk in stream_generator:
            if chunk.type == "message_start" and hasattr(chunk, "message"):
                msg = chunk.message
                if hasattr(msg, "usage") and msg.usage is not None:
                    input_tokens  = getattr(msg.usage, "input_tokens", 0)
                    output_tokens = getattr(msg.usage, "output_tokens", 0)
                    if getattr(msg.usage, "cache_creation_input_tokens", None):
                        usage_details["cache_creation_input_tokens"] = msg.usage.cache_creation_input_tokens
                    if getattr(msg.usage, "cache_read_input_tokens", None):
                        usage_details["cache_read_input_tokens"] = msg.usage.cache_read_input_tokens
                continue

            if chunk.type == "message_delta":
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    output_tokens = getattr(chunk.usage, "output_tokens", output_tokens)
                continue

            if chunk.type == "content_block_start":
                block_type = chunk.content_block.type

                if block_type == "thinking":
                    in_thinking_block     = True
                    current_thinking_text = getattr(chunk.content_block, "thinking", "") or ""
                    current_thinking_sig  = getattr(chunk.content_block, "signature", "") or ""
                    # Emit a streaming hint (empty chunk) so callers know thinking started
                    yield StreamingChatMessage(chunk="", is_tool_call=False)

                elif block_type == "redacted_thinking":
                    in_redacted_block     = True
                    current_redacted_data = getattr(chunk.content_block, "data", "") or ""

                elif block_type == "tool_use":
                    current_tool_call = {"function": {"id": chunk.content_block.id, "name": chunk.content_block.name, "arguments": ""}}
                    yield StreamingChatMessage(chunk="", is_tool_call=True, tool_call=ToolCallContent(
                        tool_call_id=current_tool_call["function"]["id"],
                        tool_call_name=current_tool_call["function"]["name"],
                        tool_call_arguments=None,
                    ).model_dump(exclude_none=True))

                elif block_type == "text":
                    content = getattr(chunk.content_block, "text", "") or ""
                    yield StreamingChatMessage(chunk=content)

            elif chunk.type == "content_block_delta":
                delta_type = chunk.delta.type

                if delta_type == "thinking_delta":
                    current_thinking_text += chunk.delta.thinking
                    # Optionally stream thinking text (callers can filter on is_thinking)
                    yield StreamingChatMessage(chunk=chunk.delta.thinking, is_tool_call=False)

                elif delta_type == "signature_delta":
                    current_thinking_sig += chunk.delta.signature

                elif delta_type == "text_delta":
                    if content is not None:
                        content += chunk.delta.text
                    yield StreamingChatMessage(chunk=chunk.delta.text)

                elif delta_type == "input_json_delta":
                    if current_tool_call:
                        current_tool_call["function"]["arguments"] += chunk.delta.partial_json

            elif chunk.type == "content_block_stop":
                if in_thinking_block:
                    contents.append(ReasoningContent(
                        thinking=current_thinking_text,
                        signature=current_thinking_sig or None,
                    ))
                    in_thinking_block     = False
                    current_thinking_text = ""
                    current_thinking_sig  = ""

                elif in_redacted_block:
                    contents.append(ReasoningContent(redacted_data=current_redacted_data))
                    in_redacted_block     = False
                    current_redacted_data = ""

                elif content is not None:
                    contents.append(TextContent(content=content))
                    content = None

                elif current_tool_call:
                    has_tool_call = True
                    try:
                        arguments = json.loads(current_tool_call["function"]["arguments"])
                    except JSONDecodeError as e:
                        arguments = {"parsing_error": f"Exception during JSON decoding: {e}"}
                    contents.append(ToolCallContent(
                        tool_call_id=current_tool_call["function"]["id"],
                        tool_call_name=current_tool_call["function"]["name"],
                        tool_call_arguments=arguments,
                    ))
                    yield StreamingChatMessage(chunk="", is_tool_call=True, finished=False, tool_call=contents[-1].model_dump())
                    current_tool_call = None

        token_usage = None
        if input_tokens > 0 or output_tokens > 0:
            token_usage = TokenUsage(
                input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens, details=usage_details,
            )
        yield StreamingChatMessage(
            chunk="", is_tool_call=False, finished=True,
            finished_chat_message=ChatMessage(
                id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=contents,
                created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(),
                token_usage=token_usage,
            ),
        )

    async def async_yield_from_provider(
        self, stream_generator: Any
    ) -> AsyncGenerator[StreamingChatMessage, None]:
        # Mirrors yield_from_provider exactly, just async for
        current_tool_call     = None
        current_thinking_text = ""
        current_thinking_sig  = ""
        current_redacted_data = ""
        in_thinking_block     = False
        in_redacted_block     = False
        contents              = []
        content               = None
        has_tool_call         = False
        input_tokens          = 0
        output_tokens         = 0
        usage_details         = {}

        async for chunk in await stream_generator:
            if chunk.type == "message_start" and hasattr(chunk, "message"):
                msg = chunk.message
                if hasattr(msg, "usage") and msg.usage is not None:
                    input_tokens  = getattr(msg.usage, "input_tokens", 0)
                    output_tokens = getattr(msg.usage, "output_tokens", 0)
                    if getattr(msg.usage, "cache_creation_input_tokens", None):
                        usage_details["cache_creation_input_tokens"] = msg.usage.cache_creation_input_tokens
                    if getattr(msg.usage, "cache_read_input_tokens", None):
                        usage_details["cache_read_input_tokens"] = msg.usage.cache_read_input_tokens
                continue

            if chunk.type == "message_delta":
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    output_tokens = getattr(chunk.usage, "output_tokens", output_tokens)
                continue

            if chunk.type == "content_block_start":
                block_type = chunk.content_block.type
                if block_type == "thinking":
                    in_thinking_block     = True
                    current_thinking_text = getattr(chunk.content_block, "thinking", "") or ""
                    current_thinking_sig  = getattr(chunk.content_block, "signature", "") or ""
                    yield StreamingChatMessage(chunk="", is_tool_call=False)
                elif block_type == "redacted_thinking":
                    in_redacted_block     = True
                    current_redacted_data = getattr(chunk.content_block, "data", "") or ""
                elif block_type == "tool_use":
                    current_tool_call = {"function": {"id": chunk.content_block.id, "name": chunk.content_block.name, "arguments": ""}}
                    yield StreamingChatMessage(chunk="", is_tool_call=True, tool_call=ToolCallContent(
                        tool_call_id=current_tool_call["function"]["id"],
                        tool_call_name=current_tool_call["function"]["name"],
                        tool_call_arguments=None,
                    ).model_dump(exclude_none=True))
                elif block_type == "text":
                    content = getattr(chunk.content_block, "text", "") or ""
                    yield StreamingChatMessage(chunk=content)

            elif chunk.type == "content_block_delta":
                delta_type = chunk.delta.type
                if delta_type == "thinking_delta":
                    current_thinking_text += chunk.delta.thinking
                    yield StreamingChatMessage(chunk=chunk.delta.thinking, is_tool_call=False)
                elif delta_type == "signature_delta":
                    current_thinking_sig += chunk.delta.signature
                elif delta_type == "text_delta":
                    if content is not None:
                        content += chunk.delta.text
                    yield StreamingChatMessage(chunk=chunk.delta.text)
                elif delta_type == "input_json_delta":
                    if current_tool_call:
                        current_tool_call["function"]["arguments"] += chunk.delta.partial_json

            elif chunk.type == "content_block_stop":
                if in_thinking_block:
                    contents.append(ReasoningContent(thinking=current_thinking_text, signature=current_thinking_sig or None))
                    in_thinking_block = False; current_thinking_text = ""; current_thinking_sig = ""
                elif in_redacted_block:
                    contents.append(ReasoningContent(redacted_data=current_redacted_data))
                    in_redacted_block = False; current_redacted_data = ""
                elif content is not None:
                    contents.append(TextContent(content=content)); content = None
                elif current_tool_call:
                    has_tool_call = True
                    try:
                        arguments = json.loads(current_tool_call["function"]["arguments"])
                    except JSONDecodeError as e:
                        arguments = {"parsing_error": str(e)}
                    contents.append(ToolCallContent(
                        tool_call_id=current_tool_call["function"]["id"],
                        tool_call_name=current_tool_call["function"]["name"],
                        tool_call_arguments=arguments,
                    ))
                    yield StreamingChatMessage(chunk="", is_tool_call=True, finished=False, tool_call=contents[-1].model_dump())
                    current_tool_call = None

        token_usage = None
        if input_tokens > 0 or output_tokens > 0:
            token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens, details=usage_details)
        yield StreamingChatMessage(
            chunk="", is_tool_call=False, finished=True,
            finished_chat_message=ChatMessage(
                id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=contents,
                created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(),
                token_usage=token_usage,
            ),
        )

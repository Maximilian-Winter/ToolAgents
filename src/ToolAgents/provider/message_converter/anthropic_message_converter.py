# anthropic_message_converter.py
import base64
import uuid
import datetime
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Generator, Optional, AsyncGenerator

import httpx

from .message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent, ToolCallContent, BinaryContent, \
    BinaryStorageType, ToolCallResultContent
from ToolAgents.provider.llm_provider import StreamingChatMessage, ProviderSettings
from ToolAgents import FunctionTool


def prepare_messages(messages: List[Dict[str, str]]) -> tuple:
    system_message = []
    other_messages = []
    cleaned_messages = messages
    for i, message in enumerate(cleaned_messages):
        if message['role'] == 'system':
            system_message = message['content']
        else:
            msg = {
                'role': message['role'],
                'content': message["content"],
            }

            other_messages.append(msg)
    return system_message, other_messages


class AnthropicMessageConverter(BaseMessageConverter):

    def prepare_request(self, model: str, messages: List[ChatMessage], settings: ProviderSettings = None,
                        tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        system, other_messages = prepare_messages(self.to_provider_format(messages))
        anthropic_tools = [tool.to_anthropic_tool() for tool in tools] if tools else None

        request_kwargs = settings.to_dict(
            include=["temperature", "top_p", "top_k", "max_tokens", "stop_sequences", "tool_choice"])
        request_kwargs["model"] = model
        request_kwargs['system'] = system
        request_kwargs['messages'] = other_messages
        if anthropic_tools and len(anthropic_tools) > 0:
            request_kwargs['tools'] = anthropic_tools
        else:
            request_kwargs.pop('tool_choice')
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
                    if isinstance(content, TextContent):
                        current_content.update({"type": "text", "text": content.content})
                    elif isinstance(content, BinaryContent):
                        if "image" in content.mime_type and content.storage_type == BinaryStorageType.Base64:
                            current_content.update({"type": "image",
                                                "source": {"type": "base64", "media_type": content.mime_type,
                                                           "data": content.content}})
                        elif "image" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                            response = httpx.get(content.content)
                            base64_string = base64.b64encode(response.content).decode("utf-8")
                            current_content.update({"type": "image",
                                                "source": {"type": "base64", "media_type": content.mime_type,
                                                           "data": base64_string}})
                        elif "pdf" in content.mime_type and content.storage_type == BinaryStorageType.Base64:
                            current_content.update({"type": "document",
                                                "source": {"type": "base64", "media_type": content.mime_type,
                                                           "data": content.content}})
                        elif "pdf" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                            response = httpx.get(content.content)
                            base64_string = base64.b64encode(response.content).decode("utf-8")
                            current_content.update({"type": "document",
                                                "source": {"type": "base64", "media_type": content.mime_type,
                                                           "data": base64_string}})
                    elif isinstance(content, ToolCallContent):
                        current_content.update({
                            "type": "tool_use",
                            "id": content.tool_call_id,
                            "name": content.tool_call_name,
                            "input": content.tool_call_arguments
                        })
                    elif isinstance(content, ToolCallResultContent):
                        current_content.update({
                            "type": "tool_result",
                            "tool_use_id": content.tool_call_id,
                            "content": content.tool_call_result
                        })
                    if len(content.additional_fields) > 0:
                        if "cache_control" in content.additional_fields:
                            current_content["cache_control"] = content.additional_fields["cache_control"]
                    new_content.append(current_content)
                except Exception as e:
                    print(f"Failed to convert message {message}: {e}")
            if len(new_content) > 0:
                converted_messages.append({"role": role, "content": new_content})

        return converted_messages

class AnthropicResponseConverter(BaseResponseConverter):
    def from_provider_response(self, response_data: Any) -> ChatMessage:
        # Anthropic's response is expected to have a 'content' attribute which is a list of blocks.
        contents = []
        for block in response_data.content:
            # Here we assume that each block has a 'type' attribute.
            if hasattr(block, "type"):
                if block.type == "tool_use":
                    contents.append(ToolCallContent(
                        tool_call_id=block.id,
                        tool_call_name=block.name,
                        tool_call_arguments=block.input
                    ))
                elif block.type == "text":
                    contents.append(TextContent(content=block.text))
        return ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=contents,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

    def yield_from_provider(self, stream_generator: Any) -> Generator[StreamingChatMessage, None, None]:
        current_tool_call = None
        contents = []
        content = None
        has_tool_call = False

        for chunk in stream_generator:
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    current_tool_call = {
                        "function": {
                            "id": chunk.content_block.id,
                            "name": chunk.content_block.name,
                            "arguments": ""
                        }
                    }
                    yield StreamingChatMessage(chunk="", is_tool_call=True, tool_call=ToolCallContent(tool_call_id=current_tool_call["function"]["id"],
                                                        tool_call_name=current_tool_call["function"]["name"],
                                                    tool_call_arguments=None).model_dump(exclude_none=True))
                if chunk.content_block.type == "text":
                    content = chunk.content_block.text
                    yield StreamingChatMessage(chunk=chunk.content_block.text)
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    content += chunk.delta.text
                    yield StreamingChatMessage(chunk=chunk.delta.text)
                elif chunk.delta.type == "input_json_delta":
                    if current_tool_call:
                        current_tool_call["function"]["arguments"] += chunk.delta.partial_json
            elif chunk.type == "content_block_stop":
                if content:
                    contents.append(TextContent(content=content))
                    content = None
                if current_tool_call:
                    has_tool_call = True
                    try:
                        arguments = json.loads(
                            current_tool_call["function"]["arguments"])
                    except JSONDecodeError as e:
                        arguments = f"Exception during JSON decoding of arguments: {e}"
                    contents.append(ToolCallContent(tool_call_id=current_tool_call["function"]["id"],
                                                        tool_call_name=current_tool_call["function"]["name"],
                                                        tool_call_arguments=arguments))
                    yield StreamingChatMessage(chunk="", is_tool_call=has_tool_call, finished=False,
                                               tool_call=contents[-1].model_dump())
                    current_tool_call = None
        yield StreamingChatMessage(chunk="", is_tool_call=False, finished=True,
                                   finished_chat_message=ChatMessage(id=str(uuid.uuid4()),
                                                                         role=ChatMessageRole.Assistant,
                                                                         content=contents,
                                                                         created_at=datetime.datetime.now(),
                                                                         updated_at=datetime.datetime.now()))

    async def async_yield_from_provider(self, stream_generator: Any) -> AsyncGenerator[StreamingChatMessage, None]:
        current_tool_call = None
        content = None
        has_tool_call = False
        contents = []
        async for chunk in await stream_generator:
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    current_tool_call = {
                        "function": {
                            "id": chunk.content_block.id,
                            "name": chunk.content_block.name,
                            "arguments": ""
                        }
                    }
                    yield StreamingChatMessage(chunk="", is_tool_call=True, tool_call=ToolCallContent(tool_call_id=current_tool_call["function"]["id"],
                                                        tool_call_name=current_tool_call["function"]["name"],
                                                    tool_call_arguments=None).model_dump(exclude_none=True))
                if chunk.content_block.type == "text":
                    content = chunk.content_block.text
                    yield StreamingChatMessage(chunk=chunk.content_block.text)
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    content += chunk.delta.text
                    yield StreamingChatMessage(chunk=chunk.delta.text)
                elif chunk.delta.type == "input_json_delta":
                    if current_tool_call:
                        current_tool_call["function"]["arguments"] += chunk.delta.partial_json
            elif chunk.type == "content_block_stop":
                if content:
                    contents.append(TextContent(content=content))
                    content = None

                if current_tool_call:
                    has_tool_call = True
                    try:
                        arguments = json.loads(
                            current_tool_call["function"]["arguments"])
                    except JSONDecodeError as e:
                        arguments = f"Exception during JSON decoding of arguments: {e}"
                    contents.append(ToolCallContent(tool_call_id=current_tool_call["function"]["id"],
                                                        tool_call_name=current_tool_call["function"]["name"],
                                                        tool_call_arguments=arguments))
                    yield StreamingChatMessage(chunk="", is_tool_call=has_tool_call, finished=False,
                                               tool_call=contents[-1].model_dump())
                    current_tool_call = None
        yield StreamingChatMessage(chunk="", is_tool_call=False, finished=True,
                                   tool_call=None,
                                   finished_chat_message=ChatMessage(id=str(uuid.uuid4()),
                                                                     role=ChatMessageRole.Assistant,
                                                                     content=contents,
                                                                     created_at=datetime.datetime.now(),
                                                                     updated_at=datetime.datetime.now()))

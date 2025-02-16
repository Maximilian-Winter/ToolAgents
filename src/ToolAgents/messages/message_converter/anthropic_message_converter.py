# anthropic_message_converter.py
import uuid
import datetime
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Generator
from .message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent, ToolCallContent, BinaryContent, \
    BinaryStorageType, ToolCallResultContent
from ...provider.llm_provider import StreamingChatAPIResponse


class AnthropicMessageConverter(BaseMessageConverter):

    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in messages:
            role = message.role.value
            if role == ChatMessageRole.Tool:
                role = ChatMessageRole.User
            new_content = []
            for content in message.content:
                if isinstance(content, TextContent):
                    new_content.append({"type": "text", "text": content.content})
                elif isinstance(content, BinaryContent):
                    if "image" in content.mime_type and content.storage_type == BinaryStorageType.Base64:
                        new_content.append({"type": "image",
                                            "source": {"type": "base64", "media_type": content.mime_type,
                                                       "data": content.content}})
                    elif "pdf" in content.mime_type and content.storage_type == BinaryStorageType.Base64:
                        new_content.append({"type": "document",
                                            "source": {"type": "base64", "media_type": content.mime_type,
                                                       "data": content.content}})
                elif isinstance(content, ToolCallContent):
                    new_content.append({
                        "type": "tool_use",
                        "id": content.tool_call_id,
                        "name": content.tool_call_name,
                        "input": content.tool_call_arguments
                    })
                elif isinstance(content, ToolCallResultContent):
                    new_content.append({
                        "type": "tool_result",
                        "tool_use_id": content.tool_call_id,
                        "content": content.tool_call_result
                    })
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

    def yield_from_provider(self, stream_generator: Any) -> Generator[StreamingChatAPIResponse, None, None]:
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
                    yield StreamingChatAPIResponse(chunk="", is_tool_call=True, tool_call=current_tool_call)
                if chunk.content_block.type == "text":
                    content = chunk.content_block.text
                    yield StreamingChatAPIResponse(chunk=chunk.content_block.text)
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    content += chunk.delta.text
                    yield StreamingChatAPIResponse(chunk=chunk.delta.text)
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
                        data = json.loads(
                            current_tool_call["function"]["arguments"])
                        contents.append(ToolCallContent(tool_call_id=current_tool_call["function"]["id"],
                                                        tool_call_name=current_tool_call["function"]["name"],
                                                        tool_call_arguments=data))
                        current_tool_call = None
                    except JSONDecodeError as e:
                        contents.append(TextContent(
                            content=f"Error Parsing Tool Use JSON Data: {str(e)}\n\n\nJSON Data:\n{current_tool_call["function"]["arguments"]}"))
        yield StreamingChatAPIResponse(chunk="", is_tool_call=has_tool_call, finished=True, tool_call=current_tool_call,
                                       finished_chat_message=ChatMessage(id=str(uuid.uuid4()),
                                                                         role=ChatMessageRole.Assistant,
                                                                         content=contents,
                                                                         created_at=datetime.datetime.now(),
                                                                         updated_at=datetime.datetime.now()))

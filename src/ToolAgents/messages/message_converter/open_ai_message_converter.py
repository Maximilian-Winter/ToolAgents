# openai_message_converter.py
import uuid
import datetime
import json
from typing import List, Dict, Any, Generator

from ToolAgents.messages.message_converter.message_converter import BaseResponseConverter
from ToolAgents.provider.llm_provider import StreamingChatAPIResponse
from .message_converter import BaseMessageConverter
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent, ToolCallContent, BinaryContent, \
    BinaryStorageType, ToolCallResultContent


class OpenAIMessageConverter(BaseMessageConverter):
    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in messages:
            role = message.role.value
            if role == ChatMessageRole.Custom.value:
                role = message.additional_information["custom_role_name"]
            new_content = []
            tool_calls = []
            for content in message.content:
                if isinstance(content, TextContent):
                    if len(content.content) > 0:
                        new_content.append({"type": "text", "text": content.content})
                elif isinstance(content, BinaryContent):
                    if "image" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                        new_content.append({"type": "image_url", "image_url": {
                            "url": content.content,
                        }})
                elif isinstance(content, ToolCallContent):
                    tool_calls.append({
                        "id": content.tool_call_id,
                        "function": {
                            "name": content.tool_call_name,
                            "arguments": json.dumps(content.tool_call_arguments)
                        },
                        "type": "function"
                    })
                elif isinstance(content, ToolCallResultContent):
                    converted_messages.append({
                        "tool_call_id": content.tool_call_id,
                        "role": "tool",
                        "name": content.tool_call_name,
                        "content": content.tool_call_result,
                    })
            if len(new_content) > 0:
                if len(tool_calls) > 0:
                    converted_messages.append({"role": role, "content": new_content, "tool_calls": tool_calls})
                else:
                    if len(new_content) == 1 and new_content[0]["type"] == "text":
                        converted_messages.append({"role": role, "content": new_content[0]["text"]})
                    else:
                        converted_messages.append({"role": role, "content": new_content})
            elif len(tool_calls) > 0:
                converted_messages.append({"role": role, "content": "", "tool_calls": tool_calls})
        return converted_messages

class OpenAIResponseConverter(BaseResponseConverter):

    def from_provider_response(self, response_data: Any) -> ChatMessage:
        # OpenAI's response: get the first choice's message.
        if hasattr(response_data, 'model_extra') and "error" in response_data.model_extra:
            raise Exception(json.dumps(response_data.model_extra["error"]))

        if response_data.choices[0].message.content is not None:
            content = [TextContent(content=response_data.choices[0].message.content)]
        else:
            content = []

        tool_calls = response_data.choices[0].message.tool_calls

        additional_information = response_data.model_dump()
        additional_information.pop("choices")

        if tool_calls:
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    arguments = "Exception during JSON decoding of arguments: {}".format(e)
                content.append(
                    ToolCallContent(
                        tool_call_id=tool_call.id,
                        tool_call_name=tool_call.function.name,
                        tool_call_arguments=arguments,
                    )
                )
        return ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=content,
                           created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(),
                           additional_information=additional_information)

    def yield_from_provider(self, stream_generator: Any) -> Generator[StreamingChatAPIResponse, None, None]:
        current_content = ""
        current_tool_calls = []
        alt_index = 0

        for chunk in stream_generator:
            delta = chunk.choices[0].delta

            if delta.content:
                current_content += delta.content
                yield StreamingChatAPIResponse(
                    chunk=delta.content,
                    is_tool_call=False,
                    finished=False,
                    finished_chat_message=None
                )

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not hasattr(tool_call, "index") or tool_call.index is None:
                        tool_call.index = alt_index
                        alt_index += 1
                    if len(current_tool_calls) <= tool_call.index:
                        current_tool_calls.append({
                            "function": {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": ""
                            }
                        })

                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                if "yielded" not in current_tool_calls[-1]:
                    yield StreamingChatAPIResponse(
                        chunk="",
                        is_tool_call=True,
                        tool_call=current_tool_calls[-1],
                        finished=False,
                        finished_chat_message=None
                    )
                    current_tool_calls[-1]["yielded"] = True

            if chunk.choices[0].finish_reason is not None:
                contents = [TextContent(content=current_content)]
                has_tool_call = False
                if len(current_tool_calls) > 0:
                    has_tool_call = True
                    for tc in current_tool_calls:
                        try:
                            arguments = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError as e:
                            arguments = f"Exception during JSON decoding of arguments: {e}"
                        contents.append(
                            ToolCallContent(
                                tool_call_id=tc["function"]["id"],
                                tool_call_name=tc["function"]["name"],
                                tool_call_arguments=arguments
                            )
                        )
                additional_data = chunk.__dict__
                additional_data.pop("choices")
                additional_data['usage'] = chunk.usage.model_dump()
                finished_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    role=ChatMessageRole.Assistant,
                    content=contents,
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now(),
                    additional_information=additional_data
                )
                yield StreamingChatAPIResponse(
                    chunk="",
                    is_tool_call=has_tool_call,
                    tool_call=current_tool_calls[-1] if has_tool_call else None,
                    finished=True,
                    finished_chat_message=finished_message
                )
                break

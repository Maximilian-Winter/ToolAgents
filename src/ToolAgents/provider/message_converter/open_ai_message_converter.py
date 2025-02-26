# openai_message_converter.py
import uuid
import datetime
import json
from typing import List, Dict, Any, Generator, Optional, AsyncGenerator

from .message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent, ToolCallContent, BinaryContent, \
    BinaryStorageType, ToolCallResultContent
from ToolAgents.provider.llm_provider import StreamingChatMessage, ProviderSettings
from ToolAgents import FunctionTool



class OpenAIMessageConverter(BaseMessageConverter):
    def prepare_request(self, model: str, messages: List[ChatMessage], settings: ProviderSettings = None,
                        tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        other_messages = self.to_provider_format(messages)
        open_ai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        request_kwargs = settings.to_dict(
            include=["temperature", "top_p", "max_tokens", "tool_choice", "extra_body", "response_format"])
        request_kwargs["model"] = model
        request_kwargs['messages'] = other_messages
        if open_ai_tools and len(open_ai_tools) > 0:
            request_kwargs['tools'] = open_ai_tools
        else:
            request_kwargs.pop('tool_choice')
        return request_kwargs
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

    def yield_from_provider(self, stream_generator: Any) -> Generator[StreamingChatMessage, None, None]:
        current_content = ""
        current_tool_calls = []
        alt_index = 0

        for chunk in stream_generator:
            delta = chunk.choices[0].delta

            if delta.content:
                current_content += delta.content
                yield StreamingChatMessage(
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
                        if len(current_tool_calls) > 0:
                            yield StreamingChatMessage(
                                chunk="",
                                is_tool_call=True,
                                tool_call=ToolCallContent(
                                    tool_call_id=current_tool_calls[-1]["function"]["id"],
                                    tool_call_name=current_tool_calls[-1]["function"]["name"],
                                    tool_call_arguments=json.loads(current_tool_calls[-1]["function"]["arguments"])
                                ).model_dump(exclude_none=True),
                                finished=False,
                                finished_chat_message=None
                            )
                        current_tool_calls.append({
                            "function": {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": ""
                            }
                        })
                        yield StreamingChatMessage(
                            chunk="",
                            is_tool_call=True,
                            tool_call=ToolCallContent(
                                tool_call_id=current_tool_calls[-1]["function"]["id"],
                                tool_call_name=current_tool_calls[-1]["function"]["name"],
                                tool_call_arguments=None
                            ).model_dump(exclude_none=True),
                            finished=False,
                            finished_chat_message=None
                        )

                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

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
                if additional_data["usage"] is not None:
                    additional_data['usage'] = chunk.usage.model_dump()
                else:
                    additional_data.pop("usage")
                finished_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    role=ChatMessageRole.Assistant,
                    content=contents,
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now(),
                    additional_information=additional_data
                )
                yield StreamingChatMessage(
                    chunk="",
                    is_tool_call=has_tool_call,
                    tool_call=contents[-1].model_dump() if has_tool_call else None,
                    finished=True,
                    finished_chat_message=finished_message
                )
    async def async_yield_from_provider(self, stream_generator: Any) -> AsyncGenerator[StreamingChatMessage, None]:
        current_content = ""
        current_tool_calls = []
        alt_index = 0

        async for chunk in await stream_generator:
            delta = chunk.choices[0].delta

            if delta.content:
                current_content += delta.content
                yield StreamingChatMessage(
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
                        if len(current_tool_calls) > 0:
                            yield StreamingChatMessage(
                                chunk="",
                                is_tool_call=True,
                                tool_call=ToolCallContent(
                                    tool_call_id=current_tool_calls[-1]["function"]["id"],
                                    tool_call_name=current_tool_calls[-1]["function"]["name"],
                                    tool_call_arguments=json.loads(current_tool_calls[-1]["function"]["arguments"])
                                ).model_dump(exclude_none=True),
                                finished=False,
                                finished_chat_message=None
                            )
                        current_tool_calls.append({
                            "function": {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": ""
                            }
                        })
                        yield StreamingChatMessage(
                            chunk="",
                            is_tool_call=True,
                            tool_call=ToolCallContent(
                                tool_call_id=current_tool_calls[-1]["function"]["id"],
                                tool_call_name=current_tool_calls[-1]["function"]["name"],
                                tool_call_arguments=None
                            ).model_dump(exclude_none=True),
                            finished=False,
                            finished_chat_message=None
                        )
                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
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
                if additional_data["usage"] is not None:
                    additional_data['usage'] = chunk.usage.model_dump()
                else:
                    additional_data.pop("usage")
                finished_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    role=ChatMessageRole.Assistant,
                    content=contents,
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now(),
                    additional_information=additional_data
                )
                yield StreamingChatMessage(
                    chunk="",
                    is_tool_call=has_tool_call,
                    tool_call=contents[-1].model_dump(exclude_none=True) if has_tool_call else None,
                    finished=True,
                    finished_chat_message=finished_message
                )

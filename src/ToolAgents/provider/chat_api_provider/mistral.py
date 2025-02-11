import copy
import dataclasses
import datetime
import json
import uuid
from typing import List, Dict, Optional, Any, Generator

from mistralai import ChatCompletionRequest
from mistralai import UserMessage
from mistralai import Tool, Function

from mistralai import Mistral

from ToolAgents import FunctionTool
from ToolAgents.interfaces import LLMTokenizer, LLMSamplingSettings
from ToolAgents.interfaces.llm_provider import ChatAPIProvider, StreamingChatAPIResponse
from ToolAgents.messages.chat_message import ChatMessage, TextContent, ToolCallContent, ChatMessageRole, \
    ToolCallResultContent, BinaryStorageType, BinaryContent
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages


class MistralSettings(LLMSamplingSettings):
    def __init__(self):
        self.temperature = 0.4
        self.top_p = 1
        self.max_tokens = 8192

    def save_to_file(self, settings_file: str):
        with open(settings_file, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

    def load_from_file(self, settings_file: str):
        with open(settings_file, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)

    def as_dict(self):
        return copy.copy(self.__dict__)

    def set_stop_tokens(self, tokens: List[str], tokenizer: LLMTokenizer = None):
        pass

    def set_max_new_tokens(self, max_new_tokens: int):
        self.max_tokens = max_new_tokens

    def set(self, setting_key: str, setting_value: str):
        if hasattr(self, setting_key):
            setattr(self, setting_key, setting_value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{setting_key}'")

    def neutralize_sampler(self, sampler_name: str):
        if sampler_name == "temperature":
            self.temperature = 1.0
        elif sampler_name == "top_p":
            self.top_p = 1.0

        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_p = 1.0

class MistralChatAPI(ChatAPIProvider):


    def __init__(self, api_key: str, model: str):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.settings = MistralSettings()

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None
        if openai_tools is None:
            response = self.client.chat.complete(
                model=self.model,
                messages=clean_history_messages(messages),
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature if settings is None else settings.temperature,
                top_p=self.settings.top_p if settings is None else settings.top_p
            )
        else:
            response = self.client.chat.complete(
                model=self.model,
                messages=clean_history_messages(messages),
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature if settings is None else settings.temperature,
                top_p=self.settings.top_p if settings is None else settings.top_p,
                tools=openai_tools,
                tool_choice="auto"
            )
        content = [TextContent(content=response.choices[0].message.content)]
        tool_calls = response.choices[0].message.tool_calls

        additional_information = response.model_dump()
        additional_information.pop("choices")
        if tools and tool_calls:
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

    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[StreamingChatAPIResponse, None, None]:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        if openai_tools is None:
            stream = self.client.chat.stream(
                model=self.model,
                messages=clean_history_messages(messages),
                max_tokens=self.settings.max_tokens,
                stream=True,
                temperature=self.settings.temperature if settings is None else settings.temperature,
                top_p=self.settings.top_p if settings is None else settings.top_p
            )
        else:
            stream = self.client.chat.stream(
                model=self.model,
                messages=clean_history_messages(messages),
                max_tokens=self.settings.max_tokens,
                stream=True,
                temperature=self.settings.temperature if settings is None else settings.temperature,
                top_p=self.settings.top_p if settings is None else settings.top_p,
                tools=openai_tools,
                tool_choice="auto"
            )
        current_content = ""
        current_tool_calls = []
        alt_index = 0
        for chunk in stream:
            delta = chunk.data.choices[0].delta

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
                        partial_tool_call=current_tool_calls[-1],
                        finished=False,
                        finished_chat_message=None
                    )
                    current_tool_calls[-1]["yielded"] = True
            if chunk.data.choices[0].finish_reason is not None:
                contents = [TextContent(content=current_content)]
                has_tool_call = False
                if tools and len(current_tool_calls) > 0:
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
                additional_data = chunk.data.__dict__
                additional_data.pop("choices")
                finished_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    role=ChatMessageRole.Assistant,
                    content=contents,
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now(),
                    additional_information=additional_data
                )

                yield StreamingChatAPIResponse(
                    chunk=current_content,
                    is_tool_call=has_tool_call,
                    finished=True,
                    finished_chat_message=finished_message,

                )
                break

        # if current_content:
        #     yield current_content

    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in messages:
            role = message.role.value
            new_content = []
            tool_calls = []
            for content in message.content:
                if isinstance(content, TextContent):
                    new_content.append({"type": "text", "text": content.content})
                elif isinstance(content, BinaryContent):
                    if "image" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                        new_content.append({"type": "image_url", "image_url": {
                            "url": content.content,
                        }})
                    else:
                        new_content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{content.content}",
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
                    converted_messages.append({"role": role, "content": new_content})
            elif len(tool_calls) > 0:
                converted_messages.append({"role": role, "content": "", "tool_calls": tool_calls})
        return converted_messages

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings
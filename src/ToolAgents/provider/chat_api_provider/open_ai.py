import dataclasses
import datetime
import json
import uuid
from copy import copy
from typing import List, Dict, Optional, Any, Generator

from openai import OpenAI

from ToolAgents import FunctionTool
from ToolAgents.interfaces import LLMTokenizer, SamplingSettings
from ToolAgents.interfaces.llm_provider import ChatAPIProvider, StreamingChatAPIResponse
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent, ToolCallContent, ToolCallResultContent, \
    BinaryContent, BinaryStorageType
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages


class OpenAISettings(SamplingSettings):
    def __init__(self):
        self.temperature = 0.4
        self.top_p = 1
        self.max_tokens = 4096
        self.response_format = None
        self.request_kwargs = None
        self.extra_body = None
        self.tool_choice = "auto"
        self.debug_mode = False

    def save_to_file(self, settings_file: str):
        with open(settings_file, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

    def load_from_file(self, settings_file: str):
        with open(settings_file, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)

    def as_dict(self):
        return copy(self.__dict__)

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

    def set_response_format(self, response_format: dict[str, Any]):
        self.response_format = response_format

    def set_extra_body(self, extra_body: dict[str, Any]):
        self.extra_body = extra_body

    def set_extra_request_kwargs(self, **kwargs):
        self.request_kwargs = kwargs

    def set_tool_choice(self, tool_choice: str):
        self.tool_choice = tool_choice

class OpenAIChatAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str , base_url: str = "https://api.openai.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.settings = OpenAISettings()

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        # Prepare base request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": clean_history_messages(messages),
            "max_tokens": self.settings.max_tokens if settings is None else settings.max_tokens,
            "temperature": self.settings.temperature if settings is None else settings.temperature,
            "top_p": self.settings.top_p if settings is None else settings.top_p,
        }

        # Add tools if present
        if openai_tools:
            request_kwargs.update({
                "tools": openai_tools,
                "tool_choice": "auto"
            })

        # Add response format if present
        if (settings is None and self.settings.response_format) or (settings and settings.response_format):
            response_format = self.settings.response_format if settings is None else settings.response_format
            request_kwargs["response_format"] = response_format

        # Add extra_body if present
        if (settings is None and self.settings.extra_body) or (settings and settings.extra_body):
            extra_body = self.settings.extra_body if settings is None else settings.extra_body
            request_kwargs["extra_body"] = extra_body

        # Add extra request kwargs if present
        if (settings is None and self.settings.request_kwargs) or (settings and settings.request_kwargs):
            extra_kwargs = self.settings.request_kwargs if settings is None else settings.request_kwargs
            request_kwargs.update(extra_kwargs)

        response = self.client.chat.completions.create(**request_kwargs)

        if response.choices[0].message.content is not None:
            content = [TextContent(content=response.choices[0].message.content)]
        else:
            content = []

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
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatAPIResponse, None, None]:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        # Prepare base request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": clean_history_messages(messages),
            "max_tokens": self.settings.max_tokens if settings is None else settings.max_tokens,
            "stream": True,
            "temperature": self.settings.temperature if settings is None else settings.temperature,
            "top_p": self.settings.top_p if settings is None else settings.top_p,
        }

        # Add tools if present
        if openai_tools:
            request_kwargs.update({
                "tools": openai_tools,
                "tool_choice": self.settings.tool_choice if settings is None else settings.tool_choice,
            })

        # Add response format if present
        if (settings is None and self.settings.response_format) or (settings and settings.response_format):
            response_format = self.settings.response_format if settings is None else settings.response_format
            request_kwargs["response_format"] = response_format

        # Add extra_body if present
        if (settings is None and self.settings.extra_body) or (settings and settings.extra_body):
            extra_body = self.settings.extra_body if settings is None else settings.extra_body
            request_kwargs["extra_body"] = extra_body

        # Add extra request kwargs if present
        if (settings is None and self.settings.request_kwargs) or (settings and settings.request_kwargs):
            extra_kwargs = self.settings.request_kwargs if settings is None else settings.request_kwargs
            request_kwargs.update(extra_kwargs)

        stream = self.client.chat.completions.create(**request_kwargs)

        current_content = ""
        current_tool_calls = []
        alt_index = 0

        for chunk in stream:
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
                        partial_tool_call=current_tool_calls[-1],
                        finished=False,
                        finished_chat_message=None
                    )
                    current_tool_calls[-1]["yielded"] = True

            if chunk.choices[0].finish_reason is not None:
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
                additional_data = chunk.__dict__
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
                    chunk="",
                    is_tool_call=has_tool_call,
                    finished=True,
                    finished_chat_message=finished_message
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
                    new_content.append({"type": "text","text": content.content})
                elif isinstance(content, BinaryContent):
                    if "image" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                        new_content.append({"type": "image_url","image_url": {
                        "url": content.content,
                    }})
                    else:
                        new_content.append({"type": "image_url", "image_url":{
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
                    if len(new_content) == 1 and new_content[0]["type"] == "text":
                        converted_messages.append({"role": role, "content": new_content[0]["text"]})
                    else:
                        converted_messages.append({"role": role, "content": new_content})
            elif len(tool_calls) > 0:
                converted_messages.append({"role": role, "content": "", "tool_calls": tool_calls})
        return converted_messages

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings
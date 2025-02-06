import dataclasses
import json
from typing import List, Dict, Optional, Any, Generator

from mistralai import ChatCompletionRequest
from mistralai import UserMessage
from mistralai import Tool, Function

from mistralai import Mistral

from ToolAgents import FunctionTool
from ToolAgents.interfaces import LLMTokenizer, LLMSamplingSettings
from ToolAgents.interfaces.llm_provider import ChatAPIProvider
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
        return dataclasses.asdict(self)

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
                     tools: Optional[List[FunctionTool]] = None) -> str:
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
        if tools and response.choices[0].message.tool_calls:
            return json.dumps({
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "function": {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        }
                    } for tool_call in response.choices[0].message.tool_calls
                ]
            })
        return response.choices[0].message.content

    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[str, None, None]:
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
                yield delta.content

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

            if chunk.data.choices[0].finish_reason == "tool_calls":
                yield json.dumps({
                    "content": current_content,
                    "tool_calls": current_tool_calls
                })
                current_content = ""
                current_tool_calls = []

        # if current_content:
        #     yield current_content

    def generate_tool_use_message(self, content: str, tool_call_id: str, tool_name: str, tool_args: str) -> Dict[
        str, Any]:
        return {
            "id": tool_call_id,
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_args)
            },
            "type": "function"
        }

    def generate_tool_response_message(self, tool_call_id: str, tool_name: str, tool_response: str) -> Dict[str, Any]:
        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": tool_name,
            "content": tool_response,
        }

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings
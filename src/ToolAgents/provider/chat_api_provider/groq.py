import dataclasses
import json
from typing import List, Optional, Dict, Generator, Any

from groq import Groq

from ToolAgents import FunctionTool
from ToolAgents.interfaces import LLMSamplingSettings, LLMTokenizer
from ToolAgents.interfaces.llm_provider import ChatAPIProvider
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages


class GroqSettings(LLMSamplingSettings):
    def __init__(self):
        self.temperature = 0.4
        self.top_p = 1.0
        self.max_tokens = 4096

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

class GroqChatAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.settings = GroqSettings()

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> str:
        groq_tools = [tool.to_openai_tool() for tool in tools] if tools else None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=clean_history_messages(messages),
            max_tokens=self.settings.max_tokens if settings is None else settings.max_tokens,
            temperature=self.settings.temperature if settings is None else settings.temperature,
            top_p=self.settings.top_p if settings is None else settings.top_p,
            tools=groq_tools,
            tool_choice="auto" if tools else None
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            return json.dumps({
                "content": response_message.content,
                "tool_calls": [
                    {
                        "function": {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        }
                    } for tool_call in tool_calls
                ]
            })
        return response_message.content

    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[str, None, None]:
        groq_tools = [tool.to_openai_tool() for tool in tools] if tools else None
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=clean_history_messages(messages),
            max_tokens=self.settings.max_tokens if settings is None else settings.max_tokens,
            temperature=self.settings.temperature if settings is None else settings.temperature,
            top_p=self.settings.top_p if settings is None else settings.top_p,
            tools=groq_tools,
            tool_choice="auto" if tools else None,
            stream=True
        )

        current_content = ""
        current_tool_calls = []

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                current_content += delta.content
                yield delta.content

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
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

            if chunk.choices[0].finish_reason == "tool_calls":
                yield json.dumps({
                    "content": current_content,
                    "tool_calls": current_tool_calls
                })
                current_content = ""
                current_tool_calls = []

    def generate_tool_use_message(self, content: str, tool_call_id: str, tool_name: str, tool_args: str) -> Dict[
        str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                    },
                    "type": "function"
                }
            ]
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
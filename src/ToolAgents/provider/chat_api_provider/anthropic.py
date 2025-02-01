import dataclasses
import json
from typing import List, Dict, Optional, Any, Generator

from anthropic import Anthropic

from ToolAgents import FunctionTool
from ToolAgents.interfaces import LLMTokenizer
from ToolAgents.interfaces.llm_provider import ChatAPIProvider, LLMSamplingSettings
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages


@dataclasses.dataclass
class AnthropicSettings(LLMSamplingSettings):
    def __init__(self):
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 0
        self.max_tokens = 1024
        self.stop_sequences = []
        self.cache_system_prompt = False
        self.cache_user_messages = False
        self.cache_recent_messages = 4

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
        elif sampler_name == "top_k":
            self.top_k = 0
        elif sampler_name == "top_p":
            self.top_p = 1.0

        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_k = 0
        self.top_p = 1.0

class AnthropicChatAPI(ChatAPIProvider):

    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.settings = AnthropicSettings()

    def prepare_messages(self, settings: AnthropicSettings, messages: List[Dict[str, str]]) -> tuple:
        system_message = None
        other_messages = []
        cleaned_messages = clean_history_messages(messages)
        for i, message in enumerate(cleaned_messages):
            if message['role'] == 'system':
                system_message = [
                    {"type": "text", "text": message['content']}
                ]
                if settings.cache_system_prompt:
                    system_message[0]["cache_control"] = {"type": "ephemeral"}
            else:
                msg = {
                    'role': message['role'],
                    'content': message["content"],
                }
                if settings.cache_user_messages:
                    if i >= len(cleaned_messages) - settings.cache_recent_messages:
                        msg["content"][0]["cache_control"] = {"type": "ephemeral"}

                other_messages.append(msg)
        return system_message, other_messages

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> str:
        system, other_messages = self.prepare_messages(self.settings if settings is None else settings, messages)
        anthropic_tools = [tool.to_anthropic_tool() for tool in tools] if tools else None
        response = self.client.messages.create(
            model=self.model,
            system=system if system else [],
            messages=other_messages,
            temperature=self.settings.temperature if settings is None else settings.temperature,
            top_p=self.settings.top_p if settings is None else settings.top_p,
            top_k=self.settings.top_k if settings is None else settings.top_k,
            max_tokens=self.settings.max_tokens if settings is None else settings.max_tokens,
            stop_sequences=self.settings.stop_sequences if settings is None else settings.stop_sequences,
            tools=anthropic_tools if anthropic_tools else []
        )
        if tools and (response.content[0].type == 'tool_use' or (
                len(response.content) > 1 and response.content[1].type == 'tool_use')):
            if response.content[0].type == 'tool_use':
                return json.dumps({
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "id": response.content[0].id,
                            "name": response.content[0].name,
                            "arguments": response.content[0].input
                        }
                    }]
                })
            elif response.content[1].type == 'tool_use':
                return json.dumps({
                    "content": response.content[0].text,
                    "tool_calls": [{
                        "function": {
                            "id": response.content[1].id,
                            "name": response.content[1].name,
                            "arguments": response.content[1].input
                        }
                    }]
                })
        return response.content[0].text

    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[str, None, None]:
        system, other_messages = self.prepare_messages(self.settings if settings is None else settings, messages)
        anthropic_tools = [tool.to_anthropic_tool() for tool in tools] if tools else None
        stream = self.client.messages.create(
            model=self.model,
            system=system if system else [],
            messages=other_messages,
            stream=True,
            temperature=self.settings.temperature if settings is None else settings.temperature,
            top_p=self.settings.top_p if settings is None else settings.top_p,
            max_tokens=self.settings.max_tokens if settings is None else settings.max_tokens,
            tools=anthropic_tools if anthropic_tools else []
        )
        current_tool_call = None
        content = ""
        for chunk in stream:
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    current_tool_call = {
                        "function": {
                            "id": chunk.content_block.id,
                            "name": chunk.content_block.name,
                            "arguments": ""
                        }
                    }
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    content += chunk.delta.text
                    yield chunk.delta.text
                elif chunk.delta.type == "input_json_delta":
                    if current_tool_call:
                        current_tool_call["function"]["arguments"] += chunk.delta.partial_json

            elif chunk.type == "content_block_stop":
                if current_tool_call:
                    yield json.dumps({
                        "content": content if len(content) > 0 else None,
                        "tool_calls": [current_tool_call]
                    })
                    current_tool_call = None

    def generate_tool_use_message(self, content: str, tool_call_id: str, tool_name: str, tool_args: str) -> Dict[
        str, Any]:
        if content is None or len(content) == 0:
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": tool_name,
                        "input": json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    }
                ]
            }
        else:
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    },
                    {
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": tool_name,
                        "input": json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    }
                ]
            }

    def generate_tool_response_message(self, tool_call_id: str, tool_name: str, tool_response: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_response
                }
            ]
        }

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings
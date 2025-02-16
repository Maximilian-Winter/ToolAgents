import copy
import datetime
import json
import uuid
from dataclasses import dataclass
from typing import Any, List, Union, Optional, Dict, Generator

import requests

from ToolAgents import FunctionTool
from ToolAgents.messages import ChatMessage, ChatMessageRole, TextContent
from ToolAgents.provider.llm_provider import SamplingSettings, ChatAPIProvider, StreamingChatAPIResponse
from .default_implementations import TemplateToolCallHandler, MistralMessageConverterLlamaCpp, MistralTokenizer
from .generation_interfaces import LLMTokenizer, LLMToolCallHandler
from ...messages.message_converter.message_converter import BaseMessageConverter


@dataclass
class LlamaCppSamplingSettings(SamplingSettings):
    def set_response_format(self, response_format: dict[str, Any]):
        pass

    def set_extra_body(self, extra_body: dict[str, Any]):
        pass

    def set_extra_request_kwargs(self, **kwargs):
        pass

    def set_tool_choice(self, tool_choice: str):
        pass

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    n_predict: int = -1
    n_keep: int = 0
    stream: bool = True
    stop: List[str] = None
    tfs_z: float = 1.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = -1
    penalize_nl: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_prompt: Union[None, str, List[int]] = None
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    cache_prompt: bool = True
    seed: int = -1
    ignore_eos: bool = False
    samplers: List[str] = None

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

    def set_stop_tokens(self, tokens: List[str]):
        self.stop = tokens

    def set_max_new_tokens(self, max_new_tokens: int):
        self.n_predict = max_new_tokens

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
        elif sampler_name == "min_p":
            self.min_p = 0.0
        elif sampler_name == "tfs_z":
            self.tfs_z = 1.0
        elif sampler_name == "typical_p":
            self.typical_p = 1.0
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_k = 0
        self.top_p = 1.0
        self.min_p = 0.0
        self.tfs_z = 1.0
        self.typical_p = 1.0


class LlamaCppServerProvider(ChatAPIProvider):


    def __init__(self, server_address: str, tokenizer: LLMTokenizer = MistralTokenizer(), message_converter: BaseMessageConverter = MistralMessageConverterLlamaCpp(), tool_call_handler: LLMToolCallHandler = TemplateToolCallHandler(), api_key: str = None):
        self.tokenizer = tokenizer
        self.message_converter = message_converter
        self.server_address = server_address
        self.tool_call_handler = tool_call_handler
        self.server_completion_endpoint = f"{self.server_address}/completion"
        self.server_chat_completion_endpoint = f"{self.server_address}/v1/chat/completions"
        self.server_tokenize_endpoint = f"{self.server_address}/tokenize"
        self.api_key = api_key
        self.default_settings = LlamaCppSamplingSettings()

    def get_default_settings(self) -> LlamaCppSamplingSettings:
        return self.default_settings

    def set_default_settings(self, settings: LlamaCppSamplingSettings):
        self.default_settings = settings

    def create_completion(self, prompt: str, settings: LlamaCppSamplingSettings):
        settings = copy.deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_completion_endpoint)

        response = requests.post(self.server_completion_endpoint, headers=headers, json=data)
        data = response.json()
        return data["content"]


    def get_response(self, messages: List[Dict[str, Any]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> ChatMessage:
        msg = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=[], created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        prompt = self.tokenizer.apply_template(messages=messages, tools=[tool.to_openai_tool() for tool in tools])
        if settings is None:
            settings = self.get_default_settings()
        settings.stream = False
        result = self.create_completion(prompt, settings)
        if self.tool_call_handler.contains_tool_calls(result):
            tool_calls = self.tool_call_handler.parse_tool_calls(result)
            msg.content.extend(tool_calls)
        else:
            msg.add_text(result.replace(self.tokenizer.get_eos_token_string(), ""))
        return msg

    def get_streaming_response(self, messages: List[Dict[str, Any]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[
        StreamingChatAPIResponse, None, None]:

        prompt = self.tokenizer.apply_template(messages=messages, tools=[tool.to_openai_tool() for tool in tools])
        if settings is None:
            settings = self.get_default_settings()
        settings.stream = False
        result = self.create_completion(prompt, settings)
        complete_response = ""
        is_in_tool_call = False
        eos_token = self.tokenizer.get_eos_token_string()
        for token in result:
            token = token.replace(eos_token, "")
            chunk = StreamingChatAPIResponse(chunk=token)
            complete_response += token
            if self.tool_call_handler.contains_partial_tool_calls(complete_response):
                chunk.is_tool_call = True
                is_in_tool_call = True
                chunk.tool_call = {}
            if not is_in_tool_call:
                yield chunk

        final_chunk = StreamingChatAPIResponse(chunk="", finished=True)
        if self.tool_call_handler.contains_tool_calls(complete_response):
            final_chunk.is_tool_call = True
            final_chunk.tool_call = {}
            msg = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=self.tool_call_handler.parse_tool_calls(complete_response),
                              created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        else:
            msg = ChatMessage(id=str(uuid.uuid4()), role=ChatMessageRole.Assistant,
                              content=[TextContent(content=complete_response.replace(self.tokenizer.get_eos_token_string(), ""))],
                              created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        final_chunk.finished_chat_message = msg

        yield final_chunk



    def convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        return self.message_converter.to_provider_format(messages)

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_data(self, settings, **kwargs):
        data = copy.deepcopy(settings)
        data.update(kwargs)

        # Adjust some key names to match the API expectations
        if 'mirostat_mode' in data:
            data['mirostat'] = data.pop('mirostat_mode')
        if 'additional_stop_sequences' in data:
            data['stop'] = data.pop('additional_stop_sequences')
        if 'max_tokens' in data:
            data['n_predict'] = data.pop('max_tokens')

        # Set default samplers if not provided
        if 'samplers' not in data or data['samplers'] is None:
            data['samplers'] = ["top_k", "tfs_z", "typical_p", "top_p", "min_p", "temperature"]

        return data

    def _get_response_stream(self, headers, data, endpoint_address):
        response = requests.post(endpoint_address, headers=headers, json=data, stream=True)
        response.raise_for_status()

        def generate_text_chunks():
            decoded_chunk = ""
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk += chunk.decode("utf-8")
                    if decoded_chunk.strip().startswith("error:"):
                        raise RuntimeError(decoded_chunk)
                    new_data = json.loads(decoded_chunk.replace("data:", ""))
                    returned_data = new_data["content"]
                    yield returned_data
                    decoded_chunk = ""

        return generate_text_chunks()




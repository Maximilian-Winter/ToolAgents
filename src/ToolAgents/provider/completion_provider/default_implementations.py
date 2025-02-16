import copy
import re
from dataclasses import dataclass
from enum import Enum

from typing import List, Dict, Union, Any
import json

import requests
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MistralTokenizerOfficial
from transformers import AutoTokenizer

from ToolAgents.messages import ToolCallContent, TextContent, ChatMessage, BinaryContent
from ToolAgents.messages.chat_message import BinaryStorageType, ToolCallResultContent
from ToolAgents.messages.message_converter.message_converter import BaseMessageConverter
from ToolAgents.provider.llm_provider import SamplingSettings
from ToolAgents.provider.completion_provider.completion_interfaces import LLMTokenizer, LLMToolCallHandler, generate_id, \
    CompletionEndpoint


class HuggingFaceTokenizer(LLMTokenizer):
    def __init__(self, huggingface_tokenizer_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_tokenizer_model)

    def apply_template(self, messages: List[Dict[str, str]], tools: List) -> str:
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools if tools and len(tools) > 0 else None,
            tokenize=False,
            add_generation_prompt=True
        )

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_eos_token_string(self) -> str:
        return self.tokenizer.decode(self.tokenizer.eos_token_id)



class MistralTokenizerVersion(Enum):
    v1 = 0
    v2 = 1
    v3 = 2
    v7 = 3


class MistralTokenizer(LLMTokenizer):


    def __init__(self, tokenizer_file: str = None,
                 tokenizer_version: MistralTokenizerVersion = MistralTokenizerVersion.v7):
        if tokenizer_file is not None:
            self.tokenizer = MistralTokenizerOfficial.from_file(tokenizer_filename=tokenizer_file)
        else:
            if tokenizer_version == MistralTokenizerVersion.v1:
                self.tokenizer = MistralTokenizerOfficial.v1()
            elif tokenizer_version == MistralTokenizerVersion.v2:
                self.tokenizer = MistralTokenizerOfficial.v2()
            elif tokenizer_version == MistralTokenizerVersion.v3:
                self.tokenizer = MistralTokenizerOfficial.v3()
            elif tokenizer_version == MistralTokenizerVersion.v7:
                self.tokenizer = MistralTokenizerOfficial.v7()

    def apply_template(self, messages: List[Dict[str, str]], tools: List) -> str:
        request = ChatCompletionRequest(
            tools=tools,
            messages=messages
        )
        tokenized = self.tokenizer.encode_chat_completion(request)
        text = tokenized.text
        text = text.replace("▁", " ")[3:]
        text = text.replace("<0x0A>", "\n")
        return text

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.instruct_tokenizer.tokenizer.encode(text, False, False)

    def get_eos_token_string(self) -> str:
        return "</s>"

class TemplateToolCallHandler(LLMToolCallHandler):
    """
    A customizable tool call handler that can be configured with different patterns and formats
    for tool call detection, parsing, and message formatting.
    """

    def __init__(
            self,
            tool_call_start: str = "[TOOL_CALLS]",
            tool_call_pattern: str = r'\[\s*{\s*"name":\s*"[^"]+"\s*,\s*"arguments":\s*{[^}]*}\s*}(?:\s*,\s*{\s*"name":\s*"[^"]+"\s*,\s*"arguments":\s*{[^}]*}\s*})*\s*\]',
            tool_name_field: str = "name",
            arguments_field: str = "arguments",
            debug_mode: bool = False
    ):
        """
        Initialize the template tool call handler with customizable patterns and field names.

        Args:
            tool_call_pattern: Regex pattern to identify tool calls in the response
            tool_name_field: Field name for the tool/function name
            arguments_field: Field name for the arguments
            debug_mode: Enable debug output
        """
        self.tool_call_start = tool_call_start
        self.tool_call_pattern = re.compile(tool_call_pattern, re.DOTALL)
        self.tool_name_field = tool_name_field
        self.arguments_field = arguments_field
        self.debug = debug_mode

    def contains_partial_tool_calls(self, response: str) -> bool:
        if self.tool_call_start in response:
            return True
        return False

    def contains_tool_calls(self, response: str) -> bool:
        """Check if the response contains tool calls using the configured pattern."""
        response = response.replace('\n', '').replace('\t', '').replace('\r', '')
        matches = self.tool_call_pattern.findall(response.strip())
        if not matches:
            return False
        return True

    def parse_tool_calls(self, response: str) -> List[Union[ToolCallContent, TextContent]]:
        """Parse tool calls from the response using the configured patterns."""
        if self.debug:
            print(f"Parsing response: {response}", flush=True)

        tool_calls = []
        matches = self.tool_call_pattern.findall(response.strip())

        for match in matches:
            try:
                # Parse the JSON content
                parsed = json.loads(match)
                # Handle both single tool call and array formats
                if isinstance(parsed, list):
                    calls = parsed
                else:
                    calls = [parsed]

                # Create GenericToolCall objects for valid calls
                for call in calls:
                    tool_calls.append(ToolCallContent(tool_call_id=generate_id(length=9), tool_call_name=call[self.tool_name_field], tool_call_arguments=call[self.arguments_field]))

            except json.JSONDecodeError as e:
                if self.debug:
                    print(f"Failed to parse tool call: {str(e)}", flush=True)
                return [TextContent(content=f"Failed to parse tool call: {str(e)}\n\n{response.strip()}")]


        return tool_calls

class MistralMessageConverterLlamaCpp(BaseMessageConverter):

    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
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
                new_content = '\n'.join([con["text"] if con["type"] == "text" else "" for con in new_content])
                if len(tool_calls) > 0:
                    converted_messages.append({"role": role, "tool_calls": tool_calls})
                else:
                    converted_messages.append({"role": role, "content": new_content})
            elif len(tool_calls) > 0:
                converted_messages.append({"role": role,  "tool_calls": tool_calls})
        return converted_messages


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


class LlamaCppServer(CompletionEndpoint):


    def __init__(self, server_address: str, api_key: str = None):
        super().__init__()
        self.server_address = server_address
        self.api_key = api_key
        self.server_completion_endpoint = f"{server_address}/completion"

    def create_completion(self, prompt: str, settings: LlamaCppSamplingSettings):
        settings = copy.deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_completion_endpoint)

        response = requests.post(self.server_completion_endpoint, headers=headers, json=data)
        data = response.json()
        return data["content"]

    def get_default_settings(self):
        return LlamaCppSamplingSettings()

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

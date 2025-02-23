import copy
import re
from enum import Enum

from typing import List, Dict, Union, Any, Optional, Generator
import json

import requests
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MistralTokenizerOfficial


from ToolAgents.messages import ToolCallContent, TextContent, ChatMessage, BinaryContent
from ToolAgents.messages.chat_message import BinaryStorageType, ToolCallResultContent, ChatMessageRole
from ToolAgents.provider.message_converter.message_converter import BaseMessageConverter
from ToolAgents.provider.llm_provider import ProviderSettings, SamplerSetting
from ToolAgents.provider.completion_provider.completion_interfaces import LLMTokenizer, LLMToolCallHandler, generate_id, \
    CompletionEndpoint, AsyncCompletionEndpoint

from ToolAgents import FunctionTool

class HuggingFaceTokenizer(LLMTokenizer):
    def __init__(self, huggingface_tokenizer_model: str):
        from transformers import AutoTokenizer
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
    def prepare_request(self, model: str, messages: List[ChatMessage], settings: ProviderSettings = None,
                        tools: Optional[List[FunctionTool]] = None) -> Dict[str, Any]:
        other_messages = self.to_provider_format(messages)
        open_ai_tools = [tool.to_openai_tool() for tool in tools] if tools else None

        request_kwargs = settings.to_dict(
            include=["temperature", "top_p", "max_tokens"])
        request_kwargs["model"] = model
        request_kwargs['messages'] = other_messages
        if open_ai_tools and len(open_ai_tools) > 0:
            request_kwargs['tools'] = open_ai_tools
        else:
            if "tool_choice" in request_kwargs:
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


from typing import Any
import json

class LlamaCppProviderSettings(ProviderSettings):
    def __init__(self):
        # Define sampler settings with their default and neutral values
        samplers = [
            SamplerSetting.create_sampler_setting("temperature", 1.0, 1.0),
            SamplerSetting.create_sampler_setting("top_k", 0, 0),
            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0),
            SamplerSetting.create_sampler_setting("min_p", 0.0, 0.0),
            SamplerSetting.create_sampler_setting("tfs_z", 1.0, 1.0),
            SamplerSetting.create_sampler_setting("typical_p", 1.0, 1.0)
        ]

        # Initialize base class with empty tool choice and samplers
        super().__init__(initial_tool_choice="", samplers=samplers)

        # Initialize other default settings
        self.set_extra_request_kwargs(
            n_keep=0,
            repeat_penalty=1.1,
            repeat_last_n=256,
            penalize_nl=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            penalty_prompt=None,
            mirostat=0,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
            cache_prompt=True,
            seed=-1,
            ignore_eos=False
        )

    def __getattr__(self, name):
        super().__getattr__(name)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

    def to_dict(self, include: list[str] = None, filter_out: list[str] = None) -> dict[str, Any]:
        """Override to handle the specific requirements of llama.cpp"""
        # Get base dictionary from parent class
        result = super().to_dict(include, filter_out)

        # Rename max_tokens to n_predict if present
        if 'max_tokens' in result:
            result['n_predict'] = result.pop('max_tokens')

        # Rename stop_sequences to stop if present
        if 'stop_sequences' in result:
            result['stop'] = result.pop('stop_sequences')

        return result

    def set_extra_body(self, extra_body: dict[str, Any]):
        pass

class LlamaCppServer(CompletionEndpoint):

    def __init__(self, server_address: str, api_key: str = None):
        super().__init__()
        self.server_address = server_address
        self.api_key = api_key
        self.server_completion_endpoint = f"{server_address}/completion"

    def create_completion(self, prompt: str, settings: LlamaCppProviderSettings):
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)
        data["stream"] = False
        response = requests.post(self.server_completion_endpoint, headers=headers, json=data)
        data = response.json()
        return data["content"]


    def create_streaming_completion(self, prompt, settings: ProviderSettings) -> Union[str, Generator[str, None, None]]:
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)
        data["stream"] = True
        return self._get_response_stream(headers, data, self.server_completion_endpoint)

    def get_default_settings(self):
        return LlamaCppProviderSettings()

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_data(self, settings, **kwargs):
        data = settings.to_dict()
        data.update(kwargs)

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


import json
import aiohttp
import requests
from typing import AsyncGenerator, Dict, Any


class AsyncLlamaCppServer(AsyncCompletionEndpoint):
    def __init__(self, server_address: str, api_key: str = None):
        super().__init__()
        self.server_address = server_address
        self.api_key = api_key
        self.server_completion_endpoint = f"{server_address}/completion"

    async def create_completion(self, prompt: str, settings: LlamaCppProviderSettings) -> str:
        """
        Async completion implementation
        """
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)
        data["stream"] = False

        async with aiohttp.ClientSession() as session:
            async with session.post(self.server_completion_endpoint, headers=headers, json=data) as response:
                response.raise_for_status()
                data = await response.json()
                return data["content"]

    async def create_streaming_completion(
            self, prompt: str, settings: ProviderSettings
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming completion implementation
        """
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)
        data["stream"] = True

        async for chunk in self._get_async_response_stream(headers, data, self.server_completion_endpoint):
            yield chunk

    def get_default_settings(self) -> LlamaCppProviderSettings:
        return LlamaCppProviderSettings()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_data(self, settings: ProviderSettings, **kwargs) -> Dict[str, Any]:
        data = settings.to_dict()
        data.update(kwargs)

        # Set default samplers if not provided
        if 'samplers' not in data or data['samplers'] is None:
            data['samplers'] = ["top_k", "tfs_z", "typical_p", "top_p", "min_p", "temperature"]

        return data

    async def _get_async_response_stream(
            self, headers: Dict[str, str], data: Dict[str, Any], endpoint_address: str
    ) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_address, headers=headers, json=data) as response:
                response.raise_for_status()

                # Buffer for incomplete chunks
                buffer = ""

                async for chunk in response.content.iter_chunks():
                    if chunk[0]:  # Only process non-empty chunks
                        buffer += chunk[0].decode("utf-8")

                        # Process complete lines in buffer
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                if line.strip().startswith("error:"):
                                    raise RuntimeError(line)

                                if line.startswith("data:"):
                                    line = line.replace("data:", "").strip()
                                    try:
                                        data = json.loads(line)
                                        yield data["content"]
                                    except json.JSONDecodeError as e:
                                        raise RuntimeError(f"Failed to parse JSON: {line}") from e

                # Process any remaining data in buffer
                if buffer.strip():
                    if buffer.strip().startswith("error:"):
                        raise RuntimeError(buffer)
                    if buffer.startswith("data:"):
                        buffer = buffer.replace("data:", "").strip()
                        try:
                            data = json.loads(buffer)
                            yield data["content"]
                        except json.JSONDecodeError as e:
                            raise RuntimeError(f"Failed to parse JSON: {buffer}") from e
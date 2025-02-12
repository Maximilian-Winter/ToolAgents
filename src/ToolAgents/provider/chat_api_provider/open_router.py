import dataclasses
import json
from types import NoneType
from typing import List, Dict, Optional, Any, Generator, Union

import requests

from ToolAgents import FunctionTool
from ToolAgents.interfaces import SamplingSettings, LLMTokenizer
from ToolAgents.provider.llm_provider import ChatAPIProvider
from ToolAgents.provider.chat_api_provider.utilities import clean_history_messages


class OpenRouterSettings(SamplingSettings):
    def __init__(self):
        self.temperature = 0.1
        self.top_p = 1.0
        self.top_k = 0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.repetition_penalty = 1.0
        self.max_tokens = None
        self.stop = None
        self.seed = None
        self.provider_ = ["Lepton"]
        self.allow_fallback_ = False

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


class OpenRouterAPI(ChatAPIProvider):
    def __init__(self, api_key: str, model: str, tool_return_value_role: str = "ipython"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.settings = OpenRouterSettings()
        self.tool_return_value_role = tool_return_value_role

    def _prepare_request_body(self, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                              stream: bool = False,
                              settings: Optional[OpenRouterSettings] = None,
                              tools: Optional[List[FunctionTool]] = None,
                              tool_choice: Optional[Union[str, Dict]] = None) -> Dict:
        body = {
            "model": self.model,
            "messages": clean_history_messages(messages),
            "stream": stream,
            "provider": {
                "order": self.settings.provider_,
                "allow_fallbacks": self.settings.allow_fallback_
            },
        }

        settings = settings or self.settings
        for key, value in settings.__dict__.items():
            if value is not None and not key.endswith("_"):
                body[key] = value

        if tools:
            body["tools"] = []
            tools = [tool.to_openai_tool() for tool in tools]
            for tool in tools:
                tool["type"] = "function"
                body["tools"].append(tool)
            body["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        return body

    def _parse_function_call(self, response: str):
        raw_output = response
        if response.strip().startswith("<|python_tag|>"):
            response = response.replace("<|python_tag|>", "")
            response = json.loads(response)
            return json.dumps({
                "raw_output": raw_output,
                "content": '',
                "tool_calls": [response]
            })
        elif response.strip().startswith("<function="):
            # Parse the function call format used by some models
            function_name = response.split("=")[1].split(">")[0]
            param_str = response.split(">")[1].split("</function")[0]
            parameters = json.loads(param_str)
            return json.dumps({
                "raw_output": raw_output,
                "tool_calls": [{
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(parameters)
                    }
                }]
            })
        elif response.strip().startswith("{") and response.strip().endswith("}"):
            if "function" in response and ("arguments" in response or "parameters" in response):
                function_call = json.loads(response.strip())
                if isinstance(function_call["function"], str):
                    function_name = function_call["function"]
                    args_name = "arguments" if "arguments" in function_call else "parameters"
                    return json.dumps({
                        "raw_output": raw_output,
                        "tool_calls": [{
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(function_call[args_name]) if isinstance(
                                    function_call[args_name], dict) else function_call[args_name]
                            }
                        }]
                    })
                else:
                    if isinstance(function_call["function"], dict):
                        function_name = function_call["function"]["name"]
                        args_name = "arguments" if "arguments" in function_call["function"] else "parameters"
                        return json.dumps({
                            "raw_output": raw_output,
                            "tool_calls": [{
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_call["function"][args_name]) if isinstance(
                                        function_call[args_name], dict) else function_call[args_name]
                                }
                            }]
                        })
            else:
                return response
        else:
            return response

    def get_response(self, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                     settings: Optional[OpenRouterSettings] = None,
                     tools: Optional[List[Dict]] = None,
                     tool_choice: Optional[Union[str, Dict]] = None) -> str:
        response = requests.post(
            url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._prepare_request_body(messages, settings=settings, tools=tools, tool_choice=tool_choice)
        )
        response_json = response.json()

        if 'choices' in response_json and len(response_json['choices']) > 0:
            message = response_json['choices'][0]['message']
            if isinstance(message, str):
                return self._parse_function_call(message)
            elif isinstance(message, dict):
                if 'tool_calls' in message:
                    return json.dumps({
                        "content": message.get('content', ''),
                        "tool_calls": message['tool_calls']
                    })
                elif message.get('content', '').startswith("<|python_tag|>"):
                    return self._parse_function_call(message['content'])
                else:
                    return message.get('content', '')
        return ''

    def get_streaming_response(self, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                               settings: Optional[OpenRouterSettings] = None,
                               tools: Optional[List[Dict]] = None,
                               tool_choice: Optional[Union[str, Dict]] = None) -> Generator[str, None, None]:
        response = requests.post(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            json=self._prepare_request_body(messages, stream=True, settings=settings, tools=tools,
                                            tool_choice=tool_choice),
            stream=True
        )

        current_content = ""
        current_tool_calls = []
        alt_index = 1
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data != '[DONE]':
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content'] is not None:
                                    content = delta['content']
                                    if not isinstance(content, NoneType):
                                        current_content += content
                                        yield content
                                elif 'tool_calls' in delta:
                                    for tool_call in delta['tool_calls']:
                                        if "index" not in tool_call:
                                            tool_call["index"] = alt_index
                                            alt_index += 1
                                        if len(current_tool_calls) <= tool_call['index'] - 1:
                                            current_tool_calls.append({
                                                "type": tool_call.get('type', ''),
                                                "function": {
                                                    "id": tool_call.get('id', ''),
                                                    "name": tool_call['function'].get('name', ''),
                                                    "arguments": tool_call['function'].get('arguments', '')
                                                }
                                            })
                                        else:
                                            current_tool_calls[tool_call['index'] - 1]['function']['arguments'] += \
                                                tool_call['function'].get('arguments', '')

                            if "choices" in chunk and chunk['choices'][0].get('finish_reason') in ['tool_calls',
                                                                                                   'function_call']:
                                yield json.dumps({
                                    "content": current_content,
                                    "tool_calls": current_tool_calls
                                })
                                current_content = ""
                                current_tool_calls = []
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON: {data}")

        if current_content:
            parsed_content = self._parse_function_call(current_content)
            try:
                parsed_json = json.loads(parsed_content)
                yield json.dumps({
                    "raw_output": current_content,
                    "content": parsed_json.get("content", ""),
                    "tool_calls": parsed_json.get("tool_calls", [])
                })
            except json.JSONDecodeError:
                yield ""

    def generate_tool_use_message(self, content: str, tool_call_id: str, tool_name: str, tool_args: str) -> Dict[
        str, Any]:
        return {
            "role": "assistant",
            "content": content if len(content) > 0 else None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args,
                    },
                    "type": "function"
                }
            ]
        }

    def generate_tool_response_message(self, tool_call_id: str, tool_name: str, tool_response: str) -> Dict[str, Any]:
        return {
            "tool_call_id": tool_call_id,
            "role": self.tool_return_value_role,
            "name": tool_name,
            "content": tool_response,
        }

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings

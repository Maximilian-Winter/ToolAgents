from abc import ABC, abstractmethod
from types import NoneType
from typing import List, Dict, Generator, Optional, Any, Union

import requests
from groq import Groq

from ToolAgents.function_tool import FunctionTool
import json
from openai import OpenAI
from anthropic import Anthropic


def clean_history_messages(history_messages: List[dict]) -> List[dict]:
    clean_messages = []
    for msg in history_messages:
        if "id" in msg:
            msg.pop("id")
        clean_messages.append(msg)

    return clean_messages


class ChatAPIProvider(ABC):
    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> str:
        pass

    @abstractmethod
    def get_streaming_response(self, messages: List[Dict[str, str]], settings=None,
                               tools: Optional[List[FunctionTool]] = None) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def generate_tool_response_message(self, tool_call_id: str, tool_name: str, tool_response: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_tool_use_message(self, content: str, tool_call_id: str, tool_name: str, tool_args: str) -> Dict[
        str, Any]:
        pass

    @abstractmethod
    def get_default_settings(self):
        pass


class OpenAISettings:
    def __init__(self):
        self.temperature = 0.4
        self.top_p = 1
        self.max_tokens = 8192


class OpenAIChatAPI(ChatAPIProvider):

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.settings = OpenAISettings()

    def get_response(self, messages: List[Dict[str, str]], settings=None,
                     tools: Optional[List[FunctionTool]] = None) -> str:
        openai_tools = [tool.to_openai_tool() for tool in tools] if tools else None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=clean_history_messages(messages),
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature if settings is None else settings.temperature,
            top_p=self.settings.top_p if settings is None else settings.top_p,
            tools=openai_tools,
            tool_choice="auto" if tools else None
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
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=clean_history_messages(messages),
                max_tokens=self.settings.max_tokens,
                stream=True,
                temperature=self.settings.temperature if settings is None else settings.temperature,
                top_p=self.settings.top_p if settings is None else settings.top_p
            )
        else:
            stream = self.client.chat.completions.create(
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
            delta = chunk.choices[0].delta

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

            if chunk.choices[0].finish_reason == "tool_calls":
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
        return OpenAISettings()


class AnthropicSettings:
    def __init__(self):
        self.temperature = 0.7
        self.top_p = 1.0
        self.top_k = 0
        self.max_tokens = 1024
        self.stop_sequences = []
        self.cache_system_prompt = False
        self.cache_user_messages = False
        self.cache_recent_messages = 10


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
            extra_headers={
                "anthropic-beta": "prompt-caching-2024-07-31"
            } if self.settings.cache_system_prompt or (settings is not None and settings.cache_system_prompt) else None,
            model=self.model,
            system=system if system else [],
            messages=other_messages,
            temperature=self.settings.temperature if settings is None else settings.temperature,
            top_p=self.settings.top_p if settings is None else settings.top_p,
            top_k=self.settings.top_k if settings is None else settings.top_k,
            max_tokens=self.settings.max_tokens if settings is None else settings.max_tokens,
            stop_sequences=self.settings.stop_sequences if settings is None else settings.stop_sequences,
            tools=anthropic_tools
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
            extra_headers={
                "anthropic-beta": "prompt-caching-2024-07-31"
            } if self.settings.cache_system_prompt or (settings is not None and settings.cache_system_prompt) else None,
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
        return AnthropicSettings()


class OpenRouterSettings:
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
        return OpenRouterSettings()


class GroqSettings:
    def __init__(self):
        self.temperature = 0.4
        self.top_p = 1.0
        self.max_tokens = 4096


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
        return GroqSettings()

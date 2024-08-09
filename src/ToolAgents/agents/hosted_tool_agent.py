from typing import Any

from ToolAgents import FunctionTool
from ToolAgents.function_tool import ToolRegistry

from ToolAgents.interfaces import HostedLLMProvider
from ToolAgents.interfaces import LLMTokenizer
from ToolAgents.interfaces import LLMToolCallHandler


class HostedToolAgent:
    def __init__(
            self,
            provider: HostedLLMProvider,
            tokenizer: LLMTokenizer,
            tool_call_handler: LLMToolCallHandler,
            debug_output: bool = False,
    ):

        self.provider = provider
        self.tokenizer = tokenizer
        self.tool_call_handler = tool_call_handler
        self.debug_output = debug_output

        self.tool_registry = ToolRegistry()
        self.last_messages_buffer = []

    def step(
            self,
            messages: list[dict[str, Any]],
            tools: list[FunctionTool] = None,
            sampling_settings=None,
            reset_last_messages_buffer: bool = True,
    ):
        if tools is None:
            tools = []

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        current_messages = messages

        self.tool_registry.register_tools(tools)

        text = self.tokenizer.apply_template(current_messages, self.tool_registry)

        if self.debug_output:
            print(text, flush=True)

        if sampling_settings is None:
            sampling_settings = self.provider.get_default_settings()

        sampling_settings.stream = False

        result = self.provider.create_completion(
            prompt=text,
            settings=sampling_settings,
        )["choices"][0]["text"]
        if self.tool_call_handler.contains_tool_calls(result):

            return self.tool_call_handler.parse_tool_calls(result), True
        else:
            return result.strip(), False

    def stream_step(
            self,
            messages: list[dict[str, Any]],
            tools: list[FunctionTool] = None,
            sampling_settings=None,
            reset_last_messages_buffer: bool = True,
    ):
        if tools is None:
            tools = []

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        current_messages = messages

        self.tool_registry.register_tools(tools)

        text = self.tokenizer.apply_template(current_messages, self.tool_registry)

        if self.debug_output:
            print(text, flush=True)

        if sampling_settings is None:
            sampling_settings = self.provider.get_default_settings()

        sampling_settings.stream = True
        result = ""
        for chunk in self.provider.create_completion(
                prompt=text,
                settings=sampling_settings,
        ):
            ch = chunk["choices"][0]["text"]
            result += ch
            yield ch, False

        if self.tool_call_handler.contains_tool_calls(result):
            yield self.tool_call_handler.parse_tool_calls(result), True

    def get_response(
            self,
            messages: list[dict[str, Any]],
            tools: list[FunctionTool] = None,
            sampling_settings=None,
            reset_last_messages_buffer: bool = True,
    ):
        result, contains_tool_call = self.step(messages, tools, sampling_settings, reset_last_messages_buffer)
        if contains_tool_call:
            self.handle_function_calling_response(result, messages)
            return self.get_response(sampling_settings=sampling_settings, tools=tools, messages=messages,
                                     reset_last_messages_buffer=False)
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": result.strip()})
            return result.strip()

    def get_streaming_response(
            self,
            messages: list[dict[str, Any]],
            tools: list[FunctionTool] = None,
            sampling_settings=None,
            reset_last_messages_buffer: bool = True,
    ):

        result = ""
        tool_calls = None
        for chunk, contains_tool_call in self.stream_step(
                messages=messages,
                tools=tools,
                sampling_settings=sampling_settings,
                reset_last_messages_buffer=reset_last_messages_buffer
        ):
            if contains_tool_call:
                tool_calls = chunk
            else:
                ch = chunk
                result += ch
                yield ch

        if tool_calls is not None:
            self.handle_function_calling_response(tool_calls, messages)
            yield "\n"
            yield from self.get_streaming_response(sampling_settings=sampling_settings, tools=tools,
                                                   messages=messages, reset_last_messages_buffer=False)
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": result.strip()})

    def handle_function_calling_response(self, tool_calls, current_messages):
        assistant_tool_call_message = self.tool_call_handler.get_tool_call_messages(tool_calls=tool_calls)

        tool_call_results = self.tool_call_handler.execute_tool_calls(tool_calls=tool_calls,
                                                                      tool_registry=self.tool_registry)
        tool_messages = self.tool_call_handler.get_tool_call_result_messages(tool_calls=tool_calls,
                                                                             tool_call_results=tool_call_results)

        if isinstance(assistant_tool_call_message, dict):
            current_messages.append(assistant_tool_call_message)
            self.last_messages_buffer.append(assistant_tool_call_message)
        elif isinstance(assistant_tool_call_message, list):
            current_messages.extend(assistant_tool_call_message)
            self.last_messages_buffer.extend(assistant_tool_call_message)

        if isinstance(tool_messages, dict):
            current_messages.append(tool_messages)
            self.last_messages_buffer.append(tool_messages)
        elif isinstance(tool_messages, list):
            current_messages.extend(tool_messages)
            self.last_messages_buffer.extend(tool_messages)

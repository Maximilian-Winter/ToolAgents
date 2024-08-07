import json
import random
import string
from typing import Any

from mistral_common.protocol.instruct.messages import (
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import ToolCall, FunctionCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from ToolAgents import FunctionTool
from ToolAgents.function_tool import ToolRegistry


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


class MistralAgent:
    def __init__(
            self,
            llm_provider,
            tokenizer_file: str = None,
            debug_output: bool = False,
    ):

        self.provider = llm_provider

        self.debug_output = debug_output

        self.tool_registry = ToolRegistry()

        if tokenizer_file is not None:
            self.tokenizer = MistralTokenizer.from_file(tokenizer_filename=tokenizer_file)
        else:
            self.tokenizer = MistralTokenizer.v3()

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

        mistral_tools = self.tool_registry.get_mistral_tools()

        text = self.prepare_prompt_text(current_messages, mistral_tools)

        if self.debug_output:
            print(text, flush=True)

        if sampling_settings is None:
            sampling_settings = self.provider.get_default_settings()

        sampling_settings.stream = False

        result = self.provider.create_completion(
            prompt=text,
            settings=sampling_settings,
        )["choices"][0]["text"]
        if self.is_function_call(result):
            if self.debug_output:
                print(result, flush=True)
            result = result.replace("[TOOL_CALLS]", "")
            function_calls = json.loads(result.strip())
            return function_calls, True
        else:
            return result.strip, False

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

        mistral_tools = self.tool_registry.get_mistral_tools()

        text = self.prepare_prompt_text(current_messages, mistral_tools)

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

        if self.is_function_call(result):
            if self.debug_output:
                print(result, flush=True)
            result = result.replace("[TOOL_CALLS]", "")
            function_calls = json.loads(result.strip())
            yield function_calls, True

    def get_response(
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

        mistral_tools = self.tool_registry.get_mistral_tools()

        text = self.prepare_prompt_text(current_messages, mistral_tools)

        if self.debug_output:
            print(text, flush=True)

        if sampling_settings is None:
            sampling_settings = self.provider.get_default_settings()

        sampling_settings.stream = False

        result = self.provider.create_completion(
            prompt=text,
            settings=sampling_settings,
        )["choices"][0]["text"]
        if self.is_function_call(result):
            self.handle_function_calling_response(result, current_messages)
            return self.get_response(sampling_settings=sampling_settings, tools=tools, messages=current_messages,
                                     reset_last_messages_buffer=False)
        else:
            self.last_messages_buffer.append(AssistantMessage(content=result.strip()).model_dump())
            return result.strip()

    def get_streaming_response(
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

        mistral_tools = self.tool_registry.get_mistral_tools()

        text = self.prepare_prompt_text(current_messages, mistral_tools)

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
            yield ch

        if self.is_function_call(result):
            self.handle_function_calling_response(result, current_messages)
            yield "\n"
            yield from self.get_streaming_response(sampling_settings=sampling_settings, tools=tools,
                                                   messages=current_messages, reset_last_messages_buffer=False)
        else:
            self.last_messages_buffer.append(AssistantMessage(content=result.strip()).model_dump())

    def get_assistant_tool_call_message(self, tool_call_messages):
        return AssistantMessage(content=None, tool_calls=tool_call_messages)

    def get_tool_call_message(self, function_call, tool_call_id):
        return ToolCall(
            function=FunctionCall(
                name=function_call["name"],
                arguments=json.dumps(function_call["arguments"]),
            ),
            id=tool_call_id,
        )

    def get_tool_message(self, tool_output, tool_call_id):
        return ToolMessage(content=str(tool_output) if not isinstance(tool_output, dict) else json.dumps(tool_output), tool_call_id=tool_call_id)

    def is_function_call(self, result):
        return (result.strip().startswith("[TOOL_CALLS]")) or (result.strip().startswith("[{") and result.strip().endswith(
                "}]") and "name" in result and "arguments" in result)

    def execute_tool(self, function_call):
        tool = self.tool_registry.get_tool(function_call["name"])
        call_parameters = function_call["arguments"]
        output = tool.execute(call_parameters)
        tool_call_id = generate_id(length=9)
        return tool_call_id, output

    def handle_function_calling_response(self, result, current_messages):
        tool_calls = []
        if self.debug_output:
            print(result, flush=True)
        result = result.replace("[TOOL_CALLS]", "")
        function_calls = json.loads(result.strip())
        tool_messages = []
        for function_call in function_calls:
            tool_call_id, output = self.execute_tool(function_call)
            tool_calls.append(self.get_tool_call_message(function_call, tool_call_id))
            tool_messages.append(
                self.get_tool_message(output, tool_call_id).model_dump()
            )
        assistant_tool_call_message = self.get_assistant_tool_call_message(tool_calls)
        current_messages.append(assistant_tool_call_message.model_dump())
        current_messages.extend(tool_messages)
        self.last_messages_buffer.append(assistant_tool_call_message.model_dump())
        self.last_messages_buffer.extend(tool_messages)

    def prepare_prompt_text(self, current_messages, mistral_tools):
        request = ChatCompletionRequest(
            tools=mistral_tools,
            messages=current_messages
        )
        tokenized = self.tokenizer.encode_chat_completion(request)
        tokens, text = tokenized.tokens, tokenized.text
        text = text.replace("▁", " ")[3:]
        text = text.replace("<0x0A>", "\n")
        return text


import json
import random
import string

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


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


class MistralAgent:
    def __init__(
            self,
            llm_provider,
            system_prompt: str = None,
            debug_output: bool = False,
    ):
        self.messages: list[
            SystemMessage | UserMessage | AssistantMessage | ToolMessage
            ] = []

        self.provider = llm_provider

        self.debug_output = debug_output
        self.system_prompt = system_prompt
        if system_prompt is not None:
            self.messages.append(SystemMessage(content=system_prompt))
        self.tokenizer_v3 = MistralTokenizer.v3()

    def get_response(
            self,
            message=None,
            tools: list[FunctionTool] = None,
            sampling_settings=None,
            messages: list = None,
            override_system_prompt: str = None,
    ):
        if tools is None:
            tools = []

        # Use provided messages if available, otherwise use internal messages
        current_messages = messages if messages is not None else self.messages.copy()

        # Override system prompt if provided
        if override_system_prompt is not None:
            current_messages = [msg for msg in current_messages if not isinstance(msg, SystemMessage)]
            current_messages.insert(0, SystemMessage(content=override_system_prompt))
        elif self.system_prompt is not None and not any(isinstance(msg, SystemMessage) for msg in current_messages):
            current_messages.insert(0, SystemMessage(content=self.system_prompt))

        if message is not None:
            msg = UserMessage(content=message)
            current_messages.append(msg)

        mistral_tools = []
        mistral_tool_mapping = {}
        for tool in tools:
            mistral_tools.append(tool.to_mistral_tool())
            mistral_tool_mapping[tool.model.__name__] = tool
        request = ChatCompletionRequest(
            tools=mistral_tools,
            messages=current_messages
        )
        tokenized = self.tokenizer_v3.encode_chat_completion(request)
        tokens, text = tokenized.tokens, tokenized.text
        text = text.replace("▁", " ")[3:]
        text = text.replace("<0x0A>", "\n")
        if self.debug_output:
            print(text, flush=True)

        if sampling_settings is None:
            sampling_settings = self.provider.get_default_settings()

        sampling_settings.stream = False

        result = self.provider.create_completion(
            prompt=text,
            settings=sampling_settings,
        )["choices"][0]["text"]
        if result.strip().startswith("[TOOL_CALLS]") or (result.strip().startswith("[{") and result.strip().endswith(
                "}]") and "name" in result and "arguments" in result):
            tool_calls = []
            if self.debug_output:
                print(result, flush=True)
            result = result.replace("[TOOL_CALLS]", "")
            function_calls = json.loads(result.strip())
            tool_messages = []
            for function_call in function_calls:
                tool = mistral_tool_mapping[function_call["name"]]
                cls = tool.model
                call_parameters = function_call["arguments"]
                call = cls(**call_parameters)
                output = call.run(**tool.additional_parameters)
                tool_call_id = generate_id(length=9)
                tool_calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(call_parameters),
                        ),
                        id=tool_call_id,
                    )
                )
                tool_messages.append(
                    ToolMessage(content=str(output), tool_call_id=tool_call_id)
                )
            current_messages.append(AssistantMessage(content=None, tool_calls=tool_calls))
            current_messages.extend(tool_messages)
            return self.get_response(sampling_settings=sampling_settings, tools=tools, messages=current_messages)
        else:
            current_messages.append(AssistantMessage(content=result.strip()))
            return result.strip()

    def get_streaming_response(
            self,
            message=None,
            tools: list[FunctionTool] = None,
            sampling_settings=None,
            messages: list = None,
            override_system_prompt: str = None,
    ):
        if tools is None:
            tools = []

        current_messages = messages if messages is not None else self.messages.copy()

        if override_system_prompt is not None:
            current_messages = [msg for msg in current_messages if not isinstance(msg, SystemMessage)]
            current_messages.insert(0, SystemMessage(content=override_system_prompt))
        elif self.system_prompt is not None and not any(isinstance(msg, SystemMessage) for msg in current_messages):
            current_messages.insert(0, SystemMessage(content=self.system_prompt))

        if message is not None:
            msg = UserMessage(content=message)
            current_messages.append(msg)

        mistral_tools = []
        mistral_tool_mapping = {}
        for tool in tools:
            mistral_tools.append(tool.to_mistral_tool())
            mistral_tool_mapping[tool.model.__name__] = tool

        request = ChatCompletionRequest(
            tools=mistral_tools,
            messages=current_messages
        )
        tokenized = self.tokenizer_v3.encode_chat_completion(request)
        tokens, text = tokenized.tokens, tokenized.text
        text = text.replace("▁", " ")[3:]
        text = text.replace("<0x0A>", "\n")

        if self.debug_output:
            print(text, flush=True)

        if sampling_settings is None:
            sampling_settings = self.provider.get_default_settings()

        sampling_settings.stream = True
        result = ""
        for chunk in self.provider.create_completion(
                prompt=text,
                settings=sampling_settings ,
        ):
            result += chunk["choices"][0]["text"]
            yield chunk["choices"][0]["text"]

        if result.strip().startswith("[TOOL_CALLS]") or (result.strip().startswith("[{") and result.strip().endswith(
                "}]") and "name" in result and "arguments" in result):
            tool_calls = []
            if self.debug_output:
                print(result, flush=True)
            result = result.replace("[TOOL_CALLS]", "")
            function_calls = json.loads(result.strip())
            tool_messages = []
            for function_call in function_calls:
                tool = mistral_tool_mapping[function_call["name"]]
                cls = tool.model
                call_parameters = function_call["arguments"]
                call = cls(**call_parameters)
                output = call.run(**tool.additional_parameters)
                tool_call_id = generate_id(length=9)
                tool_calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(call_parameters),
                        ),
                        id=tool_call_id,
                    )
                )
                tool_messages.append(
                    ToolMessage(content=str(output), tool_call_id=tool_call_id)
                )
            current_messages.append(AssistantMessage(content=None, tool_calls=tool_calls))
            current_messages.extend(tool_messages)
            yield "\n"
            yield from self.get_streaming_response(sampling_settings=sampling_settings, tools=tools,
                                                   messages=current_messages)
        else:
            current_messages.append(AssistantMessage(content=result.strip()))

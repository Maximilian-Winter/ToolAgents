import json
import random
import re
import string

from ToolAgents import FunctionTool
from ToolAgents.utilities.chat_history import Message, AdvancedChatFormatter


def parse_function_call(function_call):
    pattern = r'<function=(\w+)>(.*?)</function>'
    match = re.match(pattern, function_call)

    if not match:
        raise ValueError("Invalid function call format")

    function_name = match.group(1)
    parameters_json = match.group(2)

    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in function parameters")

    return function_name, parameters


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


system_prompt_part2 = """If you choose to call one or multiple functions, ONLY reply in the following format:

[{"function": "{function_name}", "arguments": arguments}, ...(additional function calls)]

where function_name is the name of the function, arguments is the arguments of the function as a dictionary.

Here is an example,
[{"function": "{example_function}", "arguments": {"example_parameter": 10, "example_parameter_object": { "a": 5, "b": 10}}}]

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Put the entire function call reply on one line

When you receive a tool call response, use the output to format an answer to the original user question."""


class LlamaAgent:
    def __init__(
            self,
            llm_provider,
            system_prompt: str = None,
            debug_output: bool = False,
    ):
        self.messages: list[Message] = []

        self.provider = llm_provider

        self.debug_output = debug_output
        self.system_prompt = system_prompt
        if system_prompt is not None:
            self.messages.append(Message(role="system", content=system_prompt))

        llama_31_system_template = "<|start_header_id|>system<|end_header_id|>\n\n{content}\n<|eot_id|>"
        llama_31_assistant_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{content}\n<|eot_id|>"
        llama_31_user_template = "<|start_header_id|>user<|end_header_id|>\n\n{content}\n<|eot_id|>"
        llama_31_tool_template = "<|start_header_id|>ipython<|end_header_id|>\n\n{content}\n<|eot_id|>"
        self.llama_31_chat_formatter = AdvancedChatFormatter({
            "system": llama_31_system_template,
            "user": llama_31_user_template,
            "assistant": llama_31_assistant_template,
            "tool": llama_31_tool_template,
        })

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

        if message is not None:
            msg = Message(role="user", content=message)
            self.messages.append(msg)

        # Use provided messages if available, otherwise use internal messages
        current_messages = messages if messages is not None else self.messages.copy()

        # Override system prompt if provided
        if override_system_prompt is not None:
            current_messages = [msg for msg in current_messages if msg.role != "system"]
            current_messages.insert(0, Message(role="system", content=override_system_prompt))

        converted_messages = [msg.to_dict() for msg in current_messages]

        tools_llama = []
        tool_mapping = {}
        for tool in tools:
            tools_llama.append(tool.to_openai_tool())
            tool_mapping[tool.model.__name__] = tool

        if len(tools_llama) > 0:
            converted_messages[0]["content"] += f"""\n\nYou have access to the following functions:

{json.dumps(tools_llama, indent=2)}

""" + system_prompt_part2

        text = self.llama_31_chat_formatter.format_messages(converted_messages)
        text += "<|start_header_id|>assistant<|end_header_id|>"
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
                "}]") and "function" in result and "arguments" in result):

            if self.debug_output:
                print(result, flush=True)

            function_calls = json.loads(result.strip())
            tool_messages = []
            for function_call in function_calls:
                tool = tool_mapping[function_call["function"]]
                cls = tool.model
                call_parameters = function_call["arguments"]
                call = cls(**call_parameters)
                output = call.run(**tool.additional_parameters)
                tool_messages.append(Message(role="tool", content=f'Function: "{function_call['function']}"\n\nArguments: "{json.dumps(function_call['arguments'])}"\n\nOutput:\n\n' + str(output) if output is not isinstance(output, dict) else json.dumps(output, indent=2)))

            current_messages.append(Message(role="assistant", content=result))
            current_messages.extend(tool_messages)
            return self.get_response(sampling_settings=sampling_settings, tools=tools, messages=current_messages)
        else:
            current_messages.append(Message(role="assistant", content=result.strip()))
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

        if message is not None:
            msg = Message(role="user", content=message)
            self.messages.append(msg)

        # Use provided messages if available, otherwise use internal messages
        current_messages = messages if messages is not None else self.messages.copy()

        # Override system prompt if provided
        if override_system_prompt is not None:
            current_messages = [msg for msg in current_messages if msg.role != "system"]
            current_messages.insert(0, Message(role="system", content=override_system_prompt))

        converted_messages = [msg.to_dict() for msg in current_messages]

        tools_llama = []
        tool_mapping = {}
        for tool in tools:
            tools_llama.append(tool.to_openai_tool())
            tool_mapping[tool.model.__name__] = tool

        if len(tools_llama) > 0:
            converted_messages[0]["content"] += f"""\n\nYou have access to the following functions:

{json.dumps(tools_llama, indent=2)}

""" + system_prompt_part2

        text = self.llama_31_chat_formatter.format_messages(converted_messages)
        text += "<|start_header_id|>assistant<|end_header_id|>"

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

        if result.strip().startswith("[TOOL_CALLS]") or (result.strip().startswith("[{") and result.strip().endswith(
                    "}]") and "function" in result and "arguments" in result):

            if self.debug_output:
                print(result, flush=True)

            function_calls = json.loads(result.strip())
            tool_messages = []
            for function_call in function_calls:
                tool = tool_mapping[function_call["function"]]
                cls = tool.model
                call_parameters = function_call["arguments"]
                call = cls(**call_parameters)
                output = call.run(**tool.additional_parameters)
                tool_messages.append(Message(role="tool", content=f'Function: "{function_call['function']}"\n\nArguments: "{json.dumps(function_call['arguments'])}"\n\nOutput:\n\n' + str(output) if output is not isinstance(output, dict) else json.dumps(output, indent=2)))

            current_messages.append(Message(role="assistant", content=result))
            current_messages.extend(tool_messages)
            yield "\n"
            yield from self.get_streaming_response(sampling_settings=sampling_settings, tools=tools,
                                                   messages=current_messages)
        else:
            current_messages.append(Message(role="assistant", content=result.strip()))

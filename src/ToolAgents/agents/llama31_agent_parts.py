import json
from typing import List, Dict, Any

from ToolAgents.function_tool import ToolRegistry
from ToolAgents.interfaces.llm_tool_call import LLMToolCallHandler, GenericToolCall, generate_id
from ToolAgents.interfaces.llm_tokenizer import HuggingFaceTokenizer
from ToolAgents.utilities.chat_history import Message

jinja2_template = """{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{%- if tools is not none %}
    {{- "You have access to the following functions:" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
If you choose to call one or multiple functions, ONLY reply in the following format:

[{"function": "{function_name}", "arguments": arguments}, ...(additional function calls)]

where function_name is the name of the function, arguments is the arguments of the function as a dictionary.

Here is an example,
[{"function": "{example_function}", "arguments": {"example_parameter": 10, "example_parameter_object": { "a": 5, "b": 10}}}]

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Put the entire function call reply on one line
- The ipython messages are only visible to you, not the user!

When you receive a tool call response, use the output to format an answer to the original user question.
{{- "\n\n" }}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {{- message.content }}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


class Llama31ToolCallHandler(LLMToolCallHandler):
    def __init__(self, debug_mode=False):
        self.debug = debug_mode

    def contains_tool_calls(self, response: str) -> bool:
        try:
            # Split the response by semicolons and strip whitespace
            json_data = json.loads(response)

            result = all(
                isinstance(item, dict) and
                "function" in item and
                "arguments" in item
                for item in json_data
            )
        except json.JSONDecodeError:
            result = False

        if self.debug:
            print("\nResponse is tool call" if result else "\nResponse is not tool call", flush=True)
        return result

    def parse_tool_calls(self, response: str) -> List[GenericToolCall]:
        if self.debug:
            print(response, flush=True)

        function_calls = json.loads(response)

        results = [GenericToolCall(tool_call_id=generate_id(length=9),
                                   name=tool_call["function"],
                                   arguments=tool_call["arguments"])
                   for tool_call in function_calls]
        return results

    def get_tool_call_messages(self, tool_calls: List[GenericToolCall]) -> Dict[str, Any] | List[Dict[str, Any]]:
        tool_call_messages = []
        for tool_call in tool_calls:
            tool_call_dict = tool_call.to_dict()
            del tool_call_dict["id"]
            tool_call_messages.append(tool_call_dict)
        return Message(role="assistant", content=json.dumps(tool_call_messages)).to_dict()

    def get_tool_call_result_messages(self, tool_calls: List[GenericToolCall], tool_call_results: List[Any]) -> Dict[str, Any] | List[Dict[str, Any]]:
        return [Message(role="tool",
                        content=str(tool_call_result) if not isinstance(tool_call_result, dict) else json.dumps(
                            tool_call_result),
                        tool_call_id=tool_call.get_tool_call_id()).to_dict()
                for tool_call, tool_call_result in zip(tool_calls, tool_call_results)]

    def execute_tool_calls(self, tool_calls: List[GenericToolCall], tool_registry: ToolRegistry) -> List[Any]:
        results = []
        for tool_call in tool_calls:
            tool = tool_registry.get_tool(tool_call.get_tool_name())
            call_parameters = tool_call.get_tool_call_arguments()
            output = tool.execute(call_parameters)
            results.append(output)
        return results


class Llama31Tokenizer(HuggingFaceTokenizer):
    def __init__(self):
        super().__init__("meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.tokenizer.chat_template = jinja2_template

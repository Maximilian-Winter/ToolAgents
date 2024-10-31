import json
from typing import List, Dict, Any

from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MistralTokenizerOfficial
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from ToolAgents.function_tool import ToolRegistry
from ToolAgents.interfaces.llm_tool_call import LLMToolCallHandler, LLMToolCall, GenericToolCall, generate_id
from ToolAgents.interfaces.llm_tokenizer import LLMTokenizer


class MistralToolCallHandler(LLMToolCallHandler):
    def __init__(self, debug_mode=False):
        self.debug = debug_mode

    def contains_tool_calls(self, response: str) -> bool:
        result = (response.strip().startswith("[TOOL_CALLS]")) or (response.strip().startswith("[{")
                                                                   and response.strip().endswith("}]")
                                                                   and "name" in response and "arguments" in response
                                                                   )
        if self.debug:
            print("\nResponse is tool call" if result else "\nResponse is not tool call", flush=True)
        return result

    def parse_tool_calls(self, response: str) -> List[LLMToolCall]:
        if self.debug:
            print(response, flush=True)
        result = response.replace("[TOOL_CALLS]", "")
        function_calls = json.loads(result.strip())
        results = [GenericToolCall(tool_call_id=generate_id(length=9), name=tool_call["name"],
                                   arguments=tool_call["arguments"]) for tool_call in
                   function_calls]
        return results

    def get_tool_call_messages(self, tool_calls: List[LLMToolCall]) -> Dict[str, Any] | List[Dict[str, Any]]:
        return AssistantMessage(content=None, tool_calls=[ToolCall(
            function=FunctionCall(
                name=function_call.get_tool_name(),
                arguments=json.dumps(function_call.get_tool_call_arguments()),
            ),
            id=function_call.get_tool_call_id(),
        ) for function_call in tool_calls]).model_dump()

    def get_tool_call_result_messages(self, tool_calls: List[LLMToolCall], tool_call_results: List[Any]) -> Dict[
                                                                                                                str, Any] | \
                                                                                                            List[Dict[
                                                                                                                str, Any]]:

        return [ToolMessage(content=str(tool_call_result) if not isinstance(tool_call_result, dict) else json.dumps(tool_call_result), tool_call_id=tool_call.get_tool_call_id(), name=tool_call.get_tool_name()).model_dump() for tool_call, tool_call_result in zip(tool_calls, tool_call_results)]

    def execute_tool_calls(self, tool_calls: List[LLMToolCall], tool_registry: ToolRegistry) -> List[Any]:
        results = []
        for tool_call in tool_calls:
            tool = tool_registry.get_tool(tool_call.get_tool_name())
            call_parameters = tool_call.get_tool_call_arguments()
            output = tool.execute(call_parameters)
            results.append(output)
        return results


class MistralTokenizer(LLMTokenizer):
    def __init__(self, tokenizer_file: str = None):
        if tokenizer_file is not None:
            self.tokenizer = MistralTokenizerOfficial.from_file(tokenizer_filename=tokenizer_file)
        else:
            self.tokenizer = MistralTokenizerOfficial.v3()

    def apply_template(self, messages: List[Dict[str, str]], tools: ToolRegistry) -> str:
        request = ChatCompletionRequest(
            tools=tools.get_mistral_tools(),
            messages=messages
        )
        tokenized = self.tokenizer.encode_chat_completion(request)
        text = tokenized.text
        text = text.replace("▁", " ")[3:]
        text = text.replace("<0x0A>", "\n")
        return text

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.instruct_tokenizer.tokenizer.encode(text, False, False)



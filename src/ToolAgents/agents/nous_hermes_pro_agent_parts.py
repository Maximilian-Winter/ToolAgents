import json
import re
from typing import List, Dict, Any

from ToolAgents.interfaces import LLMToolCallHandler
from ToolAgents.interfaces.llm_tokenizer import HuggingFaceTokenizer


class NousHermesProToolCallHandler(LLMToolCallHandler):
    def __init__(self, debug_mode=False):
        self.debug = debug_mode

    def contains_tool_calls(self, response: str) -> bool:
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_calls = re.findall(tool_call_pattern, response, re.DOTALL)

        result = len(tool_calls) > 0

        if self.debug:
            print("Response contains tool calls" if result else "Response does not contain tool calls", flush=True)

        return result

    def parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        if self.debug:
            print(response, flush=True)

        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_calls = re.findall(tool_call_pattern, response, re.DOTALL)

        parsed_tool_calls = []
        for tool_call in tool_calls:
            try:
                parsed_call = json.loads(tool_call)
                if "name" in parsed_call and "arguments" in parsed_call:
                    parsed_tool_calls.append(parsed_call)
                else:
                    if self.debug:
                        print(f"Invalid tool call format: {tool_call}", flush=True)
            except json.JSONDecodeError:
                if self.debug:
                    print(f"Failed to parse tool call: {tool_call}", flush=True)

        return parsed_tool_calls

    def get_tool_call_messages(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["arguments"])
                    }
                }
                for tool_call in tool_calls
            ]
        }

    def get_tool_call_result_messages(self, tool_calls: List[Dict[str, Any]], tool_call_results: List[Any]) -> List[
        Dict[str, Any]]:
        return [
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": json.dumps(tool_call_result) if isinstance(tool_call_result, dict) else str(tool_call_result)
            }
            for tool_call, tool_call_result in zip(tool_calls, tool_call_results)
        ]

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]], tool_registry: Any) -> List[Any]:
        results = []
        for tool_call in tool_calls:
            tool = tool_registry.get_tool(tool_call["name"])
            call_parameters = tool_call["arguments"]
            output = tool.execute(call_parameters)
            results.append(output)
        return results


class NousHermesProTokenizer(HuggingFaceTokenizer):
    def __init__(self):
        super().__init__("NousResearch/Hermes-2-Pro-Llama-3-8B")

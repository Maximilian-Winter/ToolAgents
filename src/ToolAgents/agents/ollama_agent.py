import json
from typing import List, Optional, Dict, Any, Generator
import ollama
from ollama import Options

from ToolAgents import FunctionTool


class OllamaAgent:
    def __init__(
            self,
            model: str,
            debug_output: bool = False,
    ):
        self.client = ollama.Client()
        self.model = model
        self.debug_output = debug_output
        self.last_messages_buffer: List[Dict[str, Any]] = []

    def get_response(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[List[FunctionTool]] = None,
            options: Optional[Options] = None,
            reset_last_messages_buffer: bool = True,
    ) -> str:
        if tools is None:
            tools = []

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        current_messages = messages

        if self.debug_output:
            print("Input messages:", json.dumps(current_messages, indent=2))

        # Convert FunctionTools to Ollama-compatible tool format
        ollama_tools = [tool.to_openai_tool() for tool in tools]

        response = self.client.chat(
            model=self.model,
            messages=current_messages,
            options=options,
            tools=ollama_tools
        )

        if response['message'].get('tool_calls') is None:
            self.last_messages_buffer.append(response['message'])
            return response['message']['content']

        self.last_messages_buffer.append(response['message'])
        current_messages.append(response['message'])

        for tool_call in response['message']['tool_calls']:
            tool = next((t for t in tools if t.model.__name__ == tool_call['function']['name']), None)
            if tool:
                function_args = json.loads(tool_call['function']['arguments']) if isinstance(
                    tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                call = tool.model(**function_args)
                function_response = call.run(**tool.additional_parameters)
                # Add function response to the conversation
                self.last_messages_buffer.append({
                    'role': 'tool',
                    'content': str(function_response),
                    'name': tool_call['function']['name'],
                })
                current_messages.append(
                    {
                        'role': 'tool',
                        'content': str(function_response),
                        'name': tool_call['function']['name'],
                    }
                )

        # Second API call: Get final response from the model
        final_response = self.get_response(messages=current_messages, tools=tools, reset_last_messages_buffer=False)
        return final_response

    def get_streaming_response(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[List[FunctionTool]] = None,
            options: Optional[Options] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[str, None, None]:
        if tools is None:
            tools = []

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        current_messages = messages

        if self.debug_output:
            print("Input messages:", json.dumps(current_messages, indent=2))

        ollama_tools = [tool.to_openai_tool() for tool in tools]

        result = ""
        for chunk in self.client.chat(
                model=self.model,
                messages=current_messages,
                tools=ollama_tools,
                options=options,
                stream=True,
        ):
            ch = chunk['message']['content']
            result += ch
            yield ch

        # Check if the model decided to use any of the provided functions
        if result.strip().startswith("{") and result.strip().endswith("}"):
            tool_calls = json.loads(result)
            tool_calls = [tool_calls]
            for tool_call in tool_calls:
                tool = next((t for t in tools if t.model.__name__ == tool_call['name']), None)
                if tool:
                    function_args = json.loads(tool_call['parameters']) if isinstance(
                        tool_call['parameters'], str) else tool_call['parameters']
                    call = tool.model(**function_args)
                    function_response = call.run(**tool.additional_parameters)
                    # Add function response to the conversation
                    self.last_messages_buffer.append({
                        'role': 'tool',
                        'content': str(function_response),
                        'name': tool_call['name'],
                    })
                    current_messages.append(
                        {
                            'role': 'tool',
                            'content': str(function_response),
                            'name': tool_call['name'],
                        }
                    )

            yield "\n"
            yield from self.get_streaming_response(messages=current_messages, tools=tools, reset_last_messages_buffer=False)
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": result})

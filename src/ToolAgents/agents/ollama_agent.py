import json
from typing import List, Optional, Dict, Any, Generator, Tuple
import ollama
from ollama import Options

from ToolAgents import ToolRegistry
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent


class OllamaAgent(BaseToolAgent):
    def __init__(self, model: str, debug_output: bool = False):
        super().__init__()
        self.client = ollama.Client()
        self.model = model
        self.debug_output = debug_output
        self.last_messages_buffer: List[Dict[str, Any]] = []
        self.tool_registry = ToolRegistry()

    def step(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: Optional[ToolRegistry] = None,
            settings: Optional[Options] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Tuple[Any, bool]:
        """
        Performs a single step of interaction with Ollama.
        """
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(messages, indent=2))

        # Convert tools to Ollama-compatible format
        ollama_tools = [tool.to_openai_tool() for tool in tools]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=settings,
            tools=ollama_tools
        )

        # Check if the response contains tool calls
        if response['message'].get('tool_calls') is not None:
            return response['message'], True

        return response['message']['content'], False

    def stream_step(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: Optional[ToolRegistry] = None,
            settings: Optional[Options] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[Tuple[str, bool], None, None]:
        """
        Performs a streaming step of interaction with Ollama.
        """
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(messages, indent=2))

        ollama_tools = [tool.to_openai_tool() for tool in tools]

        result = ""
        for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                tools=ollama_tools,
                options=settings,
                stream=True,
        ):
            ch = chunk['message']['content']
            result += ch
            yield ch, False

        # Check if the result is a tool call
        if result.strip().startswith("{") and result.strip().endswith("}"):
            try:
                tool_call = json.loads(result)
                if "name" in tool_call and "parameters" in tool_call:
                    tool_calls = [tool_call]
                    yield {
                        'content': result,
                        'tool_calls': [{
                            'function': {
                                'name': tc['name'],
                                'arguments': json.dumps(tc['parameters'])
                            }
                        } for tc in tool_calls]
                    }, True
            except json.JSONDecodeError:
                pass

    def handle_function_calling_response(
            self,
            tool_calls_result: Dict[str, Any],
            current_messages: List[Dict[str, Any]]
    ) -> None:
        """
        Handles the response containing function calls.
        """
        tool_calls = tool_calls_result.get('tool_calls', [])

        # Add the original message
        self.last_messages_buffer.append(tool_calls_result)
        current_messages.append(tool_calls_result)

        for tool_call in tool_calls:
            function_call = tool_call['function']
            tool_name = function_call['name']
            tool = self.tool_registry.get_tool(tool_name)

            if tool:
                arguments = json.loads(function_call['arguments']) if isinstance(
                    function_call['arguments'], str) else function_call['arguments']

                output = tool.execute(arguments)

                tool_message = {
                    'role': 'tool',
                    'content': str(output),
                    'name': tool_name,
                }
                self.last_messages_buffer.append(tool_message)
                current_messages.append(tool_message)

    def get_response(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: Optional[ToolRegistry] = None,
            settings: Optional[Options] = None,
            reset_last_messages_buffer: bool = True,
    ) -> str:
        """
        Gets a complete response from Ollama, handling any tool calls.
        """
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        current_messages = messages
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(current_messages, indent=2))

        # Convert tools to Ollama-compatible format
        ollama_tools = [tool.to_openai_tool() for tool in tools]

        response = self.client.chat(
            model=self.model,
            messages=current_messages,
            options=settings,
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
                tool_message = {
                    'role': 'tool',
                    'content': str(function_response),
                    'name': tool_call['function']['name'],
                }
                self.last_messages_buffer.append(tool_message)
                current_messages.append(tool_message)

        # Get final response
        final_response = self.get_response(
            messages=current_messages,
            tool_registry=tool_registry,
            settings=settings,
            reset_last_messages_buffer=False
        )
        return final_response

    def get_streaming_response(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: Optional[ToolRegistry] = None,
            settings: Optional[Options] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Generator[str, None, None]:
        """
        Gets a streaming response from Ollama, handling any tool calls.
        """
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        current_messages = messages
        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(current_messages, indent=2))

        ollama_tools = [tool.to_openai_tool() for tool in tools]

        result = ""
        for chunk in self.client.chat(
                model=self.model,
                messages=current_messages,
                tools=ollama_tools,
                options=settings,
                stream=True,
        ):
            ch = chunk['message']['content']
            result += ch
            yield ch

        # Check if the model decided to use any of the provided functions
        if result.strip().startswith("{") and result.strip().endswith("}"):
            try:
                tool_call = json.loads(result)
                if "name" in tool_call and "parameters" in tool_call:
                    tool = next((t for t in tools if t.model.__name__ == tool_call['name']), None)
                    if tool:
                        function_args = tool_call['parameters']
                        if isinstance(function_args, str):
                            function_args = json.loads(function_args)

                        output = tool.execute(function_args)

                        # Add tool response to the conversation
                        tool_message = {
                            'role': 'tool',
                            'content': str(output),
                            'name': tool_call['name'],
                        }
                        self.last_messages_buffer.append(tool_message)
                        current_messages.append(tool_message)

                        yield "\n"
                        yield from self.get_streaming_response(
                            messages=current_messages,
                            tool_registry=tool_registry,
                            settings=settings,
                            reset_last_messages_buffer=False
                        )
            except json.JSONDecodeError:
                pass

        self.last_messages_buffer.append({"role": "assistant", "content": result})
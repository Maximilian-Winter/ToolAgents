import json
from typing import List, Optional, Dict, Any, Generator
import ollama

from VirtualGameMasterFunctionCalling.function_calling import FunctionTool


class OllamaAgent:
    def __init__(
            self,
            model: str,
            system_prompt: Optional[str] = None,
            debug_output: bool = False,
    ):
        self.client = ollama.Client()
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.debug_output = debug_output
        self.system_prompt = system_prompt

        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    def get_response(
            self,
            message: Optional[str] = None,
            tools: Optional[List[FunctionTool]] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            override_system_prompt: Optional[str] = None,
    ) -> str:
        if tools is None:
            tools = []

        # Use provided messages if available, otherwise use internal messages
        current_messages = messages if messages is not None else self.messages.copy()

        if self.debug_output:
            print("Input messages:", json.dumps(current_messages, indent=2))

        # Override system prompt if provided
        if override_system_prompt is not None:
            current_messages = [msg for msg in current_messages if msg["role"] != "system"]
            current_messages.insert(0, {"role": "system", "content": override_system_prompt})
        elif self.system_prompt is not None and not any(msg["role"] == "system" for msg in current_messages):
            current_messages.insert(0, {"role": "system", "content": self.system_prompt})

        if message is not None:
            current_messages.append({"role": "user", "content": message})



        # Convert FunctionTools to Ollama-compatible tool format
        ollama_tools = [tool.to_openai_tool() for tool in tools]

        # First API call: Send the query and function descriptions to the model
        response = self.client.chat(
            model=self.model,
            messages=current_messages,
            tools=ollama_tools,
        )

        current_messages.append(response['message'])

        # Check if the model decided to use any of the provided functions
        if not response['message'].get('tool_calls'):
            return response['message']['content']

        # Process function calls made by the model
        for tool_call in response['message']['tool_calls']:
            tool = next((t for t in tools if t.model.__name__ == tool_call['function']['name']), None)
            if tool:
                function_args = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                call = tool.model(**function_args)
                function_response = call.run(**tool.additional_parameters)
                # Add function response to the conversation
                current_messages.append(
                    {
                        'role': 'tool',
                        'content': str(function_response),
                        'name': tool_call['function']['name'],
                    }
                )

        # Second API call: Get final response from the model
        final_response = self.get_response(messages=current_messages, tools=tools, override_system_prompt=override_system_prompt)
        return final_response

    def get_streaming_response(
            self,
            message: Optional[str] = None,
            tools: Optional[List[FunctionTool]] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            override_system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        if tools is None:
            tools = []

        # Use provided messages if available, otherwise use internal messages
        current_messages = messages if messages is not None else self.messages.copy()

        # Override system prompt if provided
        if override_system_prompt is not None:
            current_messages = [msg for msg in current_messages if msg["role"] != "system"]
            current_messages.insert(0, {"role": "system", "content": override_system_prompt})
        elif self.system_prompt is not None and not any(msg["role"] == "system" for msg in current_messages):
            current_messages.insert(0, {"role": "system", "content": self.system_prompt})

        if message is not None:
            current_messages.append({"role": "user", "content": message})

        if self.debug_output:
            print("Input messages:", json.dumps(current_messages, indent=2))

        # Convert FunctionTools to Ollama-compatible tool format
        ollama_tools = [tool.to_openai_tool() for tool in tools]

        # First API call: Send the query and function descriptions to the model
        for chunk in self.client.chat(
                model=self.model,
                messages=current_messages,
                tools=ollama_tools,
                stream=True,
        ):
            yield chunk['message']['content']

        # Get the full response to process tool calls
        response = self.client.chat(
            model=self.model,
            messages=current_messages,
            tools=ollama_tools,
        )

        current_messages.append(response['message'])

        # Check if the model decided to use any of the provided functions
        if not response['message'].get('tool_calls'):
            return

        # Process function calls made by the model
        for tool_call in response['message']['tool_calls']:
            tool = next((t for t in tools if t.model.__name__ == tool_call['function']['name']), None)
            if tool:
                function_args = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                call = tool.model(**function_args)
                function_response = call.run(**tool.additional_parameters)
                # Add function response to the conversation
                current_messages.append(
                    {
                        'role': 'tool',
                        'content': str(function_response),
                        'name': tool_call['function']['name'],
                    }
                )

        yield "\n"

        # Second API call: Get final response from the model
        for chunk in self.get_streaming_response(messages=current_messages, override_system_prompt=override_system_prompt, tools=tools):
            yield chunk

import json
from typing import Optional, Dict, List, Any, Tuple
from ToolAgents.utilities import ChatHistory
from ToolAgents import ToolRegistry
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent
from ToolAgents.interfaces.llm_tool_call import generate_id
from ToolAgents.interfaces.llm_provider import ChatAPIProvider


class ChatAPIAgent(BaseToolAgent):

    def __init__(self, chat_api: ChatAPIProvider, debug_output: bool = False):
        super().__init__()
        self.chat_api = chat_api
        self.debug_output = debug_output
        self.last_messages_buffer = []
        self.tool_registry = ToolRegistry()
        self.last_response_has_tool_calls = False

    def step(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> Tuple[Any, bool]:
        """
        Performs a single step of interaction with the chat API, returning the result
        and whether it contains tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            Tuple of (result, contains_tool_calls)
        """
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(messages, indent=2))

        result = self.chat_api.get_response(messages, settings=settings, tools=tools)

        if "tool_calls" in result:
            parsed_result = json.loads(result)
            return parsed_result, True
        else:
            return result.strip(), False

    def stream_step(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ):
        """
        Performs a single streaming step of interaction with the chat API,
        yielding chunks and whether they contain tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Tuples of (chunk, contains_tool_calls)
        """
        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        if self.debug_output:
            print("Input messages:", json.dumps(messages, indent=2))

        last_message = {"role": "assistant", "content": ""}
        for chunk in self.chat_api.get_streaming_response(messages, settings=settings, tools=tools):
            if "tool_calls" in chunk:
                if "tool_calls" not in last_message:
                    last_message["tool_calls"] = []
                last_message["tool_calls"].append(chunk)
                yield chunk, True
            else:
                last_message["content"] += chunk
                yield chunk, False

    def handle_function_calling_response(
            self,
            tool_calls_result: Dict[str, Any],
            current_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Handles the response containing function calls by executing tools and updating messages.

        Args:
            tool_calls_result: Dictionary containing tool calls information
            current_messages: List of current conversation messages
        """
        tool_calls = tool_calls_result["tool_calls"]
        tool_calls_prepared = []
        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool = self.tool_registry.get_tool(tool_name)

            if tool:
                call_parameters = tool_call["function"]["arguments"]
                if isinstance(call_parameters, str):
                    call_parameters = json.loads(call_parameters)

                output = tool.execute(call_parameters)

                tool_call_id = tool_call["function"].get("id", generate_id(length=9))
                tool_calls_prepared.append(
                    self.chat_api.generate_tool_use_message(
                        content=tool_calls_result["content"],
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        tool_args=call_parameters
                    )
                )
                tool_messages.append(
                    self.chat_api.generate_tool_response_message(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        tool_response=str(output)
                    )
                )

        if "role" in tool_calls_prepared[0]:
            self.last_messages_buffer.extend(tool_calls_prepared)
            current_messages.extend(tool_calls_prepared)
        else:
            self.last_messages_buffer.append(
                {"role": "assistant", "content": tool_calls_result["content"], "tool_calls": tool_calls_prepared}
            )
            current_messages.append(
                {"role": "assistant", "content": tool_calls_result["content"], "tool_calls": tool_calls_prepared}
            )

        self.last_messages_buffer.extend(tool_messages)
        current_messages.extend(tool_messages)

    def get_response(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ) -> str:
        """
        Gets a complete response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Returns:
            The final response string
        """
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False
        result, contains_tool_calls = self.step(messages, tool_registry, settings, reset_last_messages_buffer)

        if contains_tool_calls:
            self.last_response_has_tool_calls = True
            self.handle_function_calling_response(result, messages)
            return self.get_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": result})
            return result

    def get_streaming_response(
            self,
            messages: List[Dict[str, Any]],
            tool_registry: ToolRegistry = None,
            settings: Optional[Any] = None,
            reset_last_messages_buffer: bool = True,
    ):
        """
        Gets a streaming response from the chat API, handling any tool calls.

        Args:
            messages: List of message dictionaries
            tool_registry: Optional ToolRegistry containing available tools
            settings: Optional settings for the API call
            reset_last_messages_buffer: Whether to reset the message buffer

        Yields:
            Response chunks
        """
        if reset_last_messages_buffer:
            self.last_response_has_tool_calls = False

        if tool_registry is None:
            tool_registry = ToolRegistry()

        if reset_last_messages_buffer:
            self.last_messages_buffer = []

        self.tool_registry = tool_registry
        tools = [tool for tool in tool_registry.tools.values()]

        last_message = {"role": "assistant", "content": ""}
        for chunk in self.chat_api.get_streaming_response(messages, settings=settings, tools=tools):
            yield chunk
            if "tool_calls" in chunk:
                if "tool_calls" not in last_message:
                    last_message["tool_calls"] = []
                last_message["tool_calls"].append(chunk)
            else:
                last_message["content"] += chunk

        if "tool_calls" in last_message:
            if self.debug_output:
                print("Tool Calls Message:", json.dumps(last_message))
            self.last_response_has_tool_calls = True

            last_message["tool_calls"] = [
                json.loads(tool_call)["tool_calls"][0]
                for tool_call in last_message["tool_calls"]
            ]

            tool_messages = []
            tool_calls_prepared = []

            for tool_call in last_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool = self.tool_registry.get_tool(tool_name)

                if tool:
                    call_parameters = tool_call["function"]["arguments"]
                    if isinstance(call_parameters, str):
                        call_parameters = json.loads(call_parameters)

                    output = tool.execute(call_parameters)
                    tool_call_id = tool_call["function"].get("id", generate_id(length=9))

                    tool_calls_prepared.append(
                        self.chat_api.generate_tool_use_message(
                            content=last_message["content"],
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            tool_args=call_parameters
                        )
                    )

                    if "role" in tool_calls_prepared[0]:
                        tool_calls_prepared.append(
                            self.chat_api.generate_tool_response_message(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                tool_response=str(output)
                            )
                        )
                    else:
                        tool_messages.append(
                            self.chat_api.generate_tool_response_message(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                tool_response=str(output)
                            )
                        )

            if "role" in tool_calls_prepared[0]:
                self.last_messages_buffer.extend(tool_calls_prepared)
                messages.extend(tool_calls_prepared)
            else:
                self.last_messages_buffer.append(
                    {"role": "assistant", "content": last_message["content"], "tool_calls": tool_calls_prepared}
                )
                messages.append(
                    {"role": "assistant", "content": last_message["content"], "tool_calls": tool_calls_prepared}
                )
                self.last_messages_buffer.extend(tool_messages)
                messages.extend(tool_messages)

            yield "\n"
            yield from self.get_streaming_response(
                messages=messages,
                tool_registry=tool_registry,
                settings=settings,
                reset_last_messages_buffer=False
            )
        else:
            self.last_messages_buffer.append({"role": "assistant", "content": last_message["content"]})


import json
import random
import string
from typing import Optional, Dict, List, Any

from FunkyFlow import FunctionTool
from FunkyFlow.utilities.chat_api_with_tools import ChatAPI


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


class ChatAPIAgent:
    def __init__(
            self,
            chat_api: ChatAPI,
            system_prompt: Optional[str] = None,
            debug_output: bool = False,
    ):
        self.chat_api = chat_api
        self.messages: List[Dict[str, Any]] = []
        self.debug_output = debug_output
        self.system_prompt = system_prompt

        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    def get_response(
            self,
            message: Optional[str] = None,
            tools: Optional[List[FunctionTool]] = None,
            settings: Optional[Any] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            override_system_prompt: Optional[str] = None,
    ) -> str:
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

        result = self.chat_api.get_response(current_messages, settings=settings, tools=tools)

        try:
            parsed_result = json.loads(result)
            if "tool_calls" in parsed_result:
                tool_calls = parsed_result["tool_calls"]
                content = parsed_result.get("content", "")

                tool_calls_prepared = []
                tool_messages = []
                for tool_call in tool_calls:
                    tool = next((t for t in tools if t.model.__name__ == tool_call["function"]["name"]), None)
                    if tool:
                        call_parameters = tool_call["function"]["arguments"]
                        if isinstance(call_parameters, str):
                            call_parameters = json.loads(call_parameters)
                        call = tool.model(**call_parameters)
                        output = call.run(**tool.additional_parameters)
                        tool_call_id = tool_call["function"].get("id", tool_call.get("id", generate_id(length=9)))
                        tool_calls_prepared.append(
                            self.chat_api.generate_tool_use_message(content=parsed_result["content"],
                                                                    tool_call_id=tool_call_id,
                                                                    tool_name=tool_call["function"]["name"],
                                                                    tool_args=call_parameters))
                        tool_messages.append(
                            self.chat_api.generate_tool_response_message(
                                tool_call_id=tool_call_id,
                                tool_name=tool_call["function"]["name"],
                                tool_response=str(output)
                            )
                        )
                if "role" in tool_calls_prepared[0]:
                    current_messages.extend(tool_calls_prepared)
                else:
                    current_messages.append(
                        {"role": "assistant", "content": parsed_result["content"], "tool_calls": tool_calls_prepared})
                current_messages.extend(tool_messages)
                return self.get_response(settings=settings, tools=tools, messages=current_messages)
            else:
                current_messages.append({"role": "assistant", "content": result})
                return result
        except json.JSONDecodeError:
            current_messages.append({"role": "assistant", "content": result})
            return result

    def get_streaming_response(
            self,
            message: Optional[str] = None,
            tools: Optional[List[FunctionTool]] = None,
            settings: Optional[Any] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            override_system_prompt: Optional[str] = None,
    ):
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

        for chunk in self.chat_api.get_streaming_response(current_messages, settings=settings, tools=tools):
            yield chunk

        # After streaming is complete, update the message history
        last_message = {"role": "assistant", "content": ""}
        for chunk in self.chat_api.get_streaming_response(current_messages, settings=settings, tools=tools):
            try:
                parsed_chunk = json.loads(chunk)
                if "tool_calls" in parsed_chunk:
                    last_message["tool_calls"] = parsed_chunk["tool_calls"]
                    if parsed_chunk.get("content"):
                        last_message["content"] += parsed_chunk["content"]
                else:
                    last_message["content"] += chunk
            except json.JSONDecodeError:
                last_message["content"] += chunk

        if "tool_calls" in last_message:
            tool_messages = []
            tool_calls_prepared = []
            for tool_call in last_message["tool_calls"]:
                tool = next((t for t in tools if t.model.__name__ == tool_call["function"]["name"]), None)
                if tool:
                    call_parameters = tool_call["function"]["arguments"]
                    if isinstance(call_parameters, str):
                        call_parameters = json.loads(call_parameters)
                    call = tool.model(**call_parameters)
                    output = call.run(**tool.additional_parameters)
                    tool_call_id = tool_call["function"].get("id", generate_id(length=9))
                    tool_calls_prepared.append(
                        self.chat_api.generate_tool_use_message(content=last_message["content"],
                                                                tool_call_id=tool_call_id,
                                                                tool_name=tool_call["function"]["name"],
                                                                tool_args=call_parameters))
                    tool_messages.append(
                        self.chat_api.generate_tool_response_message(
                            tool_call_id=tool_call_id,
                            tool_name=tool_call["function"]["name"],
                            tool_response=str(output)
                        )
                    )
            if "role" in tool_calls_prepared[0]:
                current_messages.extend(tool_calls_prepared)
            else:
                current_messages.append(
                    {"role": "assistant", "content": last_message["content"], "tool_calls": tool_calls_prepared})

            current_messages.extend(tool_messages)
            yield "\n"
            yield from self.get_streaming_response(settings=settings, tools=tools, messages=current_messages)

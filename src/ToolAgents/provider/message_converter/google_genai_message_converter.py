# src/ToolAgents/provider/message_converter/google_genai_message_converter.py
import uuid
import datetime
import json
import base64
import httpx
from typing import List, Dict, Any, Generator, Optional, AsyncGenerator

from .message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.messages.chat_message import (
    ChatMessage,
    ChatMessageRole,
    TextContent,
    ToolCallContent,
    BinaryContent,
    BinaryStorageType,
    ToolCallResultContent,
)
from ToolAgents.provider.llm_provider import StreamingChatMessage, ProviderSettings
from ToolAgents import FunctionTool


class GoogleGenAIMessageConverter(BaseMessageConverter):
    """
    Converts ToolAgents messages to Google GenAI format
    """

    def prepare_request(
        self,
        model: str,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare the request for the Google GenAI API
        """
        converted_messages = self.to_provider_format(messages)
        google_tools = self._prepare_tools(tools) if tools else None
        system_msg = None
        for message in messages:
            if message.role == ChatMessageRole.System:
                system_msg = message.get_as_text()
                break

        # Extract relevant parameters from settings
        generation_config = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "max_output_tokens": settings.max_output_tokens,
        }

        request_kwargs = {
            "contents": converted_messages,
            "generation_config": generation_config,
            "system_instruction": system_msg,
        }

        # Add tools if provided
        if google_tools:
            request_kwargs["tools"] = google_tools

        return request_kwargs

    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert ToolAgents messages to Google GenAI format
        """
        converted_messages = []
        for message in messages:
            role = self._map_role(message.role.value)
            parts = []
            if role == "system":
                continue
            # Process each content item in the message
            for content in message.content:
                if isinstance(content, TextContent):
                    parts.append({"text": content.content})

                elif isinstance(content, BinaryContent):
                    if "image" in content.mime_type:
                        if content.storage_type == BinaryStorageType.Base64:
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": content.mime_type,
                                        "data": content.content,
                                    }
                                }
                            )
                        elif content.storage_type == BinaryStorageType.Url:
                            # Fetch image from URL and convert to base64
                            response = httpx.get(content.content)
                            base64_data = base64.b64encode(response.content).decode(
                                "utf-8"
                            )
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": content.mime_type,
                                        "data": base64_data,
                                    }
                                }
                            )

                elif isinstance(content, ToolCallContent):
                    parts.append(
                        {
                            "function_call": {
                                "id": content.tool_call_id,
                                "name": content.tool_call_name,
                                "args": {"content": content.tool_call_arguments},
                            }
                        }
                    )

                elif isinstance(content, ToolCallResultContent):
                    # Add tool response as a system message
                    parts.append(
                        {
                            "function_response": {
                                "id": content.tool_call_id,
                                "name": content.tool_call_name,
                                "response": {"result": content.tool_call_result},
                            }
                        }
                    )
            # Only add message if it has parts
            if parts:
                converted_messages.append({"role": role, "parts": parts})

        return converted_messages

    def _map_role(self, role: str) -> str:
        """Map ToolAgents role to Google GenAI role"""
        role_map = {
            "user": "user",
            "assistant": "model",
            "system": "system",
            "tool": "function",
        }
        return role_map.get(role, "user")

    def _prepare_tools(self, tools: List[FunctionTool]) -> List[Dict[str, Any]]:
        """Convert ToolAgents tools to Google GenAI tools format"""
        google_tools = []

        for tool in tools:
            function_info = tool.to_openai_tool()

            google_tool = {
                "function_declarations": [
                    {
                        "name": function_info["function"]["name"],
                        "description": function_info["function"]["description"],
                        "parameters": function_info["function"]["parameters"],
                    }
                ]
            }

            google_tools.append(google_tool)

        return google_tools


class GoogleGenAIResponseConverter(BaseResponseConverter):
    """
    Converts Google GenAI responses to ToolAgents messages
    """

    def from_provider_response(self, response_data: Any) -> ChatMessage:
        """
        Convert a Google GenAI response to a ToolAgents ChatMessage
        """
        contents = []

        # Extract the text content
        if hasattr(response_data, "text"):
            text_content = response_data.text
            contents.append(TextContent(content=text_content))
        elif hasattr(response_data, "parts"):
            for part in response_data.parts:
                if hasattr(part, "text") and part.text:
                    contents.append(TextContent(content=part.text))

        # Handle function calls
        if hasattr(response_data, "candidates") and response_data.candidates:
            candidate = response_data.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        contents.append(
                            ToolCallContent(
                                tool_call_id=str(uuid.uuid4()),
                                tool_call_name=function_call.name,
                                tool_call_arguments=function_call.args,
                            )
                        )

        # Create and return the chat message
        return ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=contents,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
        )

    def yield_from_provider(
        self, stream_generator: Any
    ) -> Generator[StreamingChatMessage, None, None]:
        """
        Yield streaming chunks from a Google GenAI response
        """
        current_text = ""
        has_tool_call = False
        function_name = None
        function_args = ""

        for chunk in stream_generator:
            # Process text content
            if hasattr(chunk, "text"):
                delta_text = chunk.text
                current_text += delta_text
                yield StreamingChatMessage(
                    chunk=delta_text, is_tool_call=False, finished=False
                )

            # Process function calls
            if (
                hasattr(chunk, "candidates")
                and chunk.candidates
                and hasattr(chunk.candidates[0], "content")
                and hasattr(chunk.candidates[0].content, "parts")
            ):

                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        has_tool_call = True
                        function_call = part.function_call

                        if not function_name:
                            function_name = function_call.name

                        if hasattr(function_call, "args"):
                            function_args += function_call.args

        # Final chunk with full message
        tool_call = None
        if has_tool_call:
            try:
                args = json.loads(function_args)
            except json.JSONDecodeError:
                args = function_args

            tool_content = ToolCallContent(
                tool_call_id=str(uuid.uuid4()),
                tool_call_name=function_name,
                tool_call_arguments=args,
            )

            tool_call = tool_content.model_dump(exclude_none=True)

        # Create message contents
        contents = []
        if current_text:
            contents.append(TextContent(content=current_text))

        if has_tool_call:
            contents.append(
                ToolCallContent(
                    tool_call_id=tool_call["tool_call_id"],
                    tool_call_name=tool_call["tool_call_name"],
                    tool_call_arguments=tool_call["tool_call_arguments"],
                )
            )

        # Create the final message
        final_message = ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=contents,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
        )

        # Yield the final message
        yield StreamingChatMessage(
            chunk="",
            is_tool_call=has_tool_call,
            tool_call=tool_call,
            finished=True,
            finished_chat_message=final_message,
        )

    async def async_yield_from_provider(
        self, stream_generator: Any
    ) -> AsyncGenerator[StreamingChatMessage, None]:
        """
        Yield streaming chunks from a Google GenAI response asynchronously
        """
        current_text = ""
        has_tool_call = False
        function_name = None
        function_args = ""

        async for chunk in stream_generator:
            # Process text content
            if hasattr(chunk, "text"):
                delta_text = chunk.text
                current_text += delta_text
                yield StreamingChatMessage(
                    chunk=delta_text, is_tool_call=False, finished=False
                )

            # Process function calls
            if (
                hasattr(chunk, "candidates")
                and chunk.candidates
                and hasattr(chunk.candidates[0], "content")
                and hasattr(chunk.candidates[0].content, "parts")
            ):

                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        has_tool_call = True
                        function_call = part.function_call

                        if not function_name:
                            function_name = function_call.name

                        if hasattr(function_call, "args"):
                            function_args += function_call.args

        # Final chunk with full message
        tool_call = None
        if has_tool_call:
            try:
                args = json.loads(function_args)
            except json.JSONDecodeError:
                args = function_args

            tool_content = ToolCallContent(
                tool_call_id=str(uuid.uuid4()),
                tool_call_name=function_name,
                tool_call_arguments=args,
            )

            tool_call = tool_content.model_dump(exclude_none=True)

        # Create message contents
        contents = []
        if current_text:
            contents.append(TextContent(content=current_text))

        if has_tool_call:
            contents.append(
                ToolCallContent(
                    tool_call_id=tool_call["tool_call_id"],
                    tool_call_name=tool_call["tool_call_name"],
                    tool_call_arguments=tool_call["tool_call_arguments"],
                )
            )

        # Create the final message
        final_message = ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=contents,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
        )

        # Yield the final message
        yield StreamingChatMessage(
            chunk="",
            is_tool_call=has_tool_call,
            tool_call=tool_call,
            finished=True,
            finished_chat_message=final_message,
        )

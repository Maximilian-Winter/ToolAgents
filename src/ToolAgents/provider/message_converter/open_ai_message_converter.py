import random
import string
import uuid
import datetime
import json
from typing import List, Dict, Any, Generator, Optional, AsyncGenerator

from .message_converter import BaseMessageConverter, BaseResponseConverter
from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    TextContent,
    ToolCallContent,
    BinaryContent,
    BinaryStorageType,
    ToolCallResultContent,
    TokenUsage,
    ReasoningContent,
)
from ToolAgents.provider.llm_provider import StreamingChatMessage, ProviderSettings
from ToolAgents import FunctionTool


def generate_tool_call_id(length=9):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


class OpenAIMessageConverter(BaseMessageConverter):

    def __init__(self, without_tool_call_content: bool = True):
        super().__init__()
        self.without_tool_call_content = without_tool_call_content

    def prepare_request(
        self,
        model: str,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Dict[str, Any]:
        other_messages = self.to_provider_format(messages)
        open_ai_tools  = [tool.to_openai_tool() for tool in tools] if tools else None

        request_kwargs = settings.to_dict()["REQUEST_SETTINGS"]
        request_kwargs["model"]    = model
        request_kwargs["messages"] = other_messages
        if open_ai_tools and len(open_ai_tools) > 0:
            request_kwargs["tools"] = open_ai_tools
        else:
            request_kwargs.pop("tool_choice", None)
        return request_kwargs

    def to_provider_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in messages:
            role = message.role.value
            if role == ChatMessageRole.Custom.value:
                role = message.additional_information.get("custom_role", role)
            new_content = []
            tool_calls  = []
            reasoning_text = None  # collected from ReasoningContent blocks

            for content in message.content:
                if isinstance(content, ReasoningContent):
                    # Pass reasoning back as message.reasoning for OpenRouter
                    # multi-turn (ignored by providers that don't support it)
                    if not content.is_redacted and content.thinking:
                        reasoning_text = content.thinking
                elif isinstance(content, TextContent):
                    if len(content.content) > 0:
                        new_content.append({"type": "text", "text": content.content})
                elif isinstance(content, BinaryContent):
                    if "image" in content.mime_type and content.storage_type == BinaryStorageType.Url:
                        new_content.append({"type": "image_url", "image_url": {"url": content.content}})
                    else:
                        new_content.append({"type": "image_url", "image_url": {"url": f"data:{content.mime_type};base64,{content.content}"}})
                elif isinstance(content, ToolCallContent):
                    tool_calls.append({
                        "id": content.tool_call_id,
                        "function": {"name": content.tool_call_name, "arguments": json.dumps(content.tool_call_arguments)},
                        "type": "function",
                    })
                elif isinstance(content, ToolCallResultContent):
                    converted_messages.append({
                        "tool_call_id": content.tool_call_id,
                        "role": "tool",
                        "name": content.tool_call_name,
                        "content": content.tool_call_result,
                    })

            if len(new_content) > 0:
                msg_dict = {}
                if len(tool_calls) > 0:
                    msg_dict = {"role": role, "content": new_content, "tool_calls": tool_calls}
                elif len(new_content) == 1 and new_content[0]["type"] == "text":
                    msg_dict = {"role": role, "content": new_content[0]["text"]}
                else:
                    msg_dict = {"role": role, "content": new_content}
                # Attach reasoning for OpenRouter multi-turn passback
                if reasoning_text and role == "assistant":
                    msg_dict["reasoning"] = reasoning_text
                converted_messages.append(msg_dict)
            elif len(tool_calls) > 0:
                msg_dict = {"role": role, "tool_calls": tool_calls} if self.without_tool_call_content \
                           else {"role": role, "content": "", "tool_calls": tool_calls}
                if reasoning_text:
                    msg_dict["reasoning"] = reasoning_text
                converted_messages.append(msg_dict)

        return converted_messages


class OpenAIResponseConverter(BaseResponseConverter):

    def __init__(self, tool_call_id_style: str = "openai"):
        super().__init__()
        self.tool_call_id_style = tool_call_id_style

    def from_provider_response(self, response_data: Any) -> ChatMessage:
        if hasattr(response_data, "model_extra") and "error" in response_data.model_extra:
            raise Exception(json.dumps(response_data.model_extra["error"]))

        message    = response_data.choices[0].message
        content    = []

        # ── Reasoning content (OpenRouter normalised field) ───────────────────
        reasoning_text = getattr(message, "reasoning", None)
        if reasoning_text:
            content.append(ReasoningContent(thinking=reasoning_text))

        # ── Text content ──────────────────────────────────────────────────────
        if message.content is not None:
            content.append(TextContent(content=message.content))

        # ── Tool calls ────────────────────────────────────────────────────────
        tool_calls = message.tool_calls
        if tool_calls:
            for tc in tool_calls:
                if tc.function.arguments.strip() == "":
                    arguments = {}
                else:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        arguments = f"Exception during JSON decoding: {e}"
                if self.tool_call_id_style == "mistral":
                    tc.id = generate_tool_call_id()
                content.append(ToolCallContent(
                    tool_call_id=tc.id,
                    tool_call_name=tc.function.name,
                    tool_call_arguments=arguments,
                ))

        # ── Token usage (including reasoning_tokens count) ────────────────────
        additional_information = response_data.model_dump()
        additional_information.pop("choices")
        token_usage = None
        usage_data  = additional_information.get("usage")
        if usage_data and isinstance(usage_data, dict):
            details = {
                k: v for k, v in usage_data.items()
                if k not in ("prompt_tokens", "completion_tokens", "total_tokens") and v is not None
            }
            # completion_tokens_details.reasoning_tokens is the key one for o-series
            ctd = usage_data.get("completion_tokens_details")
            if ctd and isinstance(ctd, dict) and ctd.get("reasoning_tokens"):
                details["reasoning_tokens"] = ctd["reasoning_tokens"]
            token_usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                details=details,
            )

        return ChatMessage(
            id=str(uuid.uuid4()),
            role=ChatMessageRole.Assistant,
            content=content,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
            additional_information=additional_information,
            token_usage=token_usage,
        )

    def yield_from_provider(
        self, stream_generator: Any
    ) -> Generator[StreamingChatMessage, None, None]:
        current_content    = ""
        current_reasoning  = ""    # OpenRouter streams reasoning in delta.reasoning
        current_tool_calls = []
        alt_index          = 0

        for chunk in stream_generator:
            delta = chunk.choices[0].delta

            # ── Reasoning delta (OpenRouter) ──────────────────────────────────
            reasoning_chunk = getattr(delta, "reasoning", None)
            if reasoning_chunk:
                current_reasoning += reasoning_chunk
                # Emit with a flag callers can use to distinguish reasoning
                yield StreamingChatMessage(chunk=reasoning_chunk, is_tool_call=False, is_reasoning_chunk=True)

            # ── Text delta ────────────────────────────────────────────────────
            if delta.content:
                current_content += delta.content
                yield StreamingChatMessage(chunk=delta.content, is_tool_call=False, finished=False)

            # ── Tool call deltas ──────────────────────────────────────────────
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not hasattr(tool_call, "index") or tool_call.index is None:
                        tool_call.index = alt_index; alt_index += 1
                    if len(current_tool_calls) <= tool_call.index:
                        if len(current_tool_calls) > 0:
                            yield StreamingChatMessage(chunk="", is_tool_call=True,
                                tool_call=ToolCallContent(
                                    tool_call_id=current_tool_calls[-1]["function"]["id"],
                                    tool_call_name=current_tool_calls[-1]["function"]["name"],
                                    tool_call_arguments=json.loads(current_tool_calls[-1]["function"]["arguments"]),
                                ).model_dump(exclude_none=True), finished=False)
                        current_tool_calls.append({"function": {"id": tool_call.id, "name": tool_call.function.name, "arguments": ""}})
                        yield StreamingChatMessage(chunk="", is_tool_call=True,
                            tool_call=ToolCallContent(
                                tool_call_id=current_tool_calls[-1]["function"]["id"],
                                tool_call_name=current_tool_calls[-1]["function"]["name"],
                                tool_call_arguments=None,
                            ).model_dump(exclude_none=True), finished=False)
                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

            if chunk.choices[0].finish_reason is not None:
                contents     = []
                has_tool_call = False
                # Add reasoning block first if we collected any
                if current_reasoning:
                    contents.append(ReasoningContent(thinking=current_reasoning))
                contents.append(TextContent(content=current_content))
                if len(current_tool_calls) > 0:
                    has_tool_call = True
                    for tc in current_tool_calls:
                        try:
                            arguments = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError as e:
                            arguments = f"Exception during JSON decoding: {e}"
                        if self.tool_call_id_style == "mistral":
                            tc["function"]["id"] = generate_tool_call_id()
                        contents.append(ToolCallContent(
                            tool_call_id=tc["function"]["id"],
                            tool_call_name=tc["function"]["name"],
                            tool_call_arguments=arguments,
                        ))
                additional_data = chunk.__dict__; additional_data.pop("choices")
                token_usage = None
                if additional_data.get("usage") is not None:
                    usage_dict = chunk.usage.model_dump(); additional_data["usage"] = usage_dict
                    details = {k: v for k, v in usage_dict.items() if k not in ("prompt_tokens", "completion_tokens", "total_tokens") and v is not None}
                    ctd = usage_dict.get("completion_tokens_details")
                    if ctd and isinstance(ctd, dict) and ctd.get("reasoning_tokens"):
                        details["reasoning_tokens"] = ctd["reasoning_tokens"]
                    token_usage = TokenUsage(
                        input_tokens=usage_dict.get("prompt_tokens", 0),
                        output_tokens=usage_dict.get("completion_tokens", 0),
                        total_tokens=usage_dict.get("total_tokens", 0),
                        details=details,
                    )
                else:
                    additional_data.pop("usage", None)
                yield StreamingChatMessage(
                    chunk="", is_tool_call=has_tool_call,
                    tool_call=contents[-1].model_dump() if has_tool_call else None,
                    finished=True,
                    finished_chat_message=ChatMessage(
                        id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=contents,
                        created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(),
                        additional_information=additional_data, token_usage=token_usage,
                    ),
                )

    async def async_yield_from_provider(
        self, stream_generator: Any
    ) -> AsyncGenerator[StreamingChatMessage, None]:
        current_content    = ""
        current_reasoning  = ""
        current_tool_calls = []
        alt_index          = 0

        async for chunk in await stream_generator:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta

            reasoning_chunk = getattr(delta, "reasoning", None)
            if reasoning_chunk:
                current_reasoning += reasoning_chunk
                yield StreamingChatMessage(chunk=reasoning_chunk, is_tool_call=False, is_reasoning_chunk=True)

            if delta.content:
                current_content += delta.content
                yield StreamingChatMessage(chunk=delta.content, is_tool_call=False, finished=False)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not hasattr(tool_call, "index") or tool_call.index is None:
                        tool_call.index = alt_index; alt_index += 1
                    if len(current_tool_calls) <= tool_call.index:
                        if len(current_tool_calls) > 0:
                            yield StreamingChatMessage(chunk="", is_tool_call=True,
                                tool_call=ToolCallContent(
                                    tool_call_id=current_tool_calls[-1]["function"]["id"],
                                    tool_call_name=current_tool_calls[-1]["function"]["name"],
                                    tool_call_arguments=json.loads(current_tool_calls[-1]["function"]["arguments"]),
                                ).model_dump(exclude_none=True), finished=False)
                        current_tool_calls.append({"function": {"id": tool_call.id, "name": tool_call.function.name, "arguments": ""}})
                        yield StreamingChatMessage(chunk="", is_tool_call=True,
                            tool_call=ToolCallContent(
                                tool_call_id=current_tool_calls[-1]["function"]["id"],
                                tool_call_name=current_tool_calls[-1]["function"]["name"],
                                tool_call_arguments=None,
                            ).model_dump(exclude_none=True), finished=False)
                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

            if chunk.choices[0].finish_reason is not None:
                contents = []
                has_tool_call = False
                if current_reasoning:
                    contents.append(ReasoningContent(thinking=current_reasoning))
                contents.append(TextContent(content=current_content))
                if len(current_tool_calls) > 0:
                    has_tool_call = True
                    for tc in current_tool_calls:
                        try:
                            arguments = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError as e:
                            arguments = f"Exception during JSON decoding: {e}"
                        if self.tool_call_id_style == "mistral":
                            tc["function"]["id"] = generate_tool_call_id()
                        contents.append(ToolCallContent(
                            tool_call_id=tc["function"]["id"],
                            tool_call_name=tc["function"]["name"],
                            tool_call_arguments=arguments,
                        ))
                additional_data = chunk.__dict__; additional_data.pop("choices")
                token_usage = None
                if additional_data.get("usage") is not None:
                    usage_dict = chunk.usage.model_dump(); additional_data["usage"] = usage_dict
                    details = {k: v for k, v in usage_dict.items() if k not in ("prompt_tokens", "completion_tokens", "total_tokens") and v is not None}
                    ctd = usage_dict.get("completion_tokens_details")
                    if ctd and isinstance(ctd, dict) and ctd.get("reasoning_tokens"):
                        details["reasoning_tokens"] = ctd["reasoning_tokens"]
                    token_usage = TokenUsage(
                        input_tokens=usage_dict.get("prompt_tokens", 0),
                        output_tokens=usage_dict.get("completion_tokens", 0),
                        total_tokens=usage_dict.get("total_tokens", 0),
                        details=details,
                    )
                else:
                    additional_data.pop("usage", None)
                yield StreamingChatMessage(
                    chunk="", is_tool_call=has_tool_call,
                    tool_call=contents[-1].model_dump(exclude_none=True) if has_tool_call else None,
                    finished=True,
                    finished_chat_message=ChatMessage(
                        id=str(uuid.uuid4()), role=ChatMessageRole.Assistant, content=contents,
                        created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(),
                        additional_information=additional_data, token_usage=token_usage,
                    ),
                )

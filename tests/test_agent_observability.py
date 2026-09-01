import asyncio
import datetime

from ToolAgents.agents.base_llm_agent import AgentObservabilityHandler
from ToolAgents.agents.chat_tool_agent import AsyncChatToolAgent, ChatToolAgent
from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    StreamingChatMessage,
    TextContent,
    ToolCallContent,
    ToolCallResultContent,
)
from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.provider.llm_provider import (
    AsyncChatAPIProvider,
    ChatAPIProvider,
    ProviderSettings,
)


class RecordingObservabilityHandler(AgentObservabilityHandler):
    def __init__(self):
        self.requests = []
        self.streaming_requests = []
        self.tool_calls = []

    def on_request(
        self,
        messages,
        tool_registry,
        settings,
        reset_last_messages_buffer,
        result_chat_message,
    ):
        self.requests.append(
            {
                "messages": messages,
                "tool_registry": tool_registry,
                "settings": settings,
                "reset_last_messages_buffer": reset_last_messages_buffer,
                "result_chat_message": result_chat_message,
            }
        )

    def on_streaming_request(
        self,
        messages,
        tool_registry,
        settings,
        reset_last_messages_buffer,
        result_chat_message,
    ):
        self.streaming_requests.append(
            {
                "messages": messages,
                "tool_registry": tool_registry,
                "settings": settings,
                "reset_last_messages_buffer": reset_last_messages_buffer,
                "result_chat_message": result_chat_message,
            }
        )

    def on_tool_call(
        self, tool_call: ToolCallContent, tool_call_result: ToolCallResultContent
    ):
        self.tool_calls.append(
            {
                "tool_call": tool_call,
                "tool_call_result": tool_call_result,
            }
        )


class FakeChatAPI(ChatAPIProvider):
    def __init__(self, responses=None, streaming_responses=None):
        self.responses = list(responses or [])
        self.streaming_responses = list(streaming_responses or [])
        self.default_settings = ProviderSettings()

    def get_response(self, messages, settings=None, tools=None):
        return self.responses.pop(0)

    def get_streaming_response(self, messages, settings=None, tools=None):
        yield from self.streaming_responses.pop(0)

    def get_default_settings(self):
        return self.default_settings

    def set_default_settings(self, settings) -> None:
        self.default_settings = settings

    def get_provider_identifier(self) -> str:
        return "fake"


class FakeAsyncChatAPI(AsyncChatAPIProvider):
    def __init__(self, responses=None, streaming_responses=None):
        self.responses = list(responses or [])
        self.streaming_responses = list(streaming_responses or [])
        self.default_settings = ProviderSettings()

    async def get_response(self, messages, settings=None, tools=None):
        return self.responses.pop(0)

    async def get_streaming_response(self, messages, settings=None, tools=None):
        async def stream():
            for chunk in self.streaming_responses.pop(0):
                yield chunk

        return stream()

    def get_default_settings(self):
        return self.default_settings

    def set_default_settings(self, settings) -> None:
        self.default_settings = settings

    def get_provider_identifier(self) -> str:
        return "fake-async"


def AddValue(value: int) -> str:
    """
    Add a value.

    Args:
        value: Value to add.
    """
    return f"added {value}"


def assistant_message(message_id: str, text: str) -> ChatMessage:
    now = datetime.datetime.now()
    return ChatMessage(
        id=message_id,
        role=ChatMessageRole.Assistant,
        content=[TextContent(content=text)],
        created_at=now,
        updated_at=now,
    )


def assistant_tool_call_message(message_id: str) -> ChatMessage:
    now = datetime.datetime.now()
    return ChatMessage(
        id=message_id,
        role=ChatMessageRole.Assistant,
        content=[
            ToolCallContent(
                tool_call_id="call-1",
                tool_call_name="AddValue",
                tool_call_arguments={"value": 3},
            )
        ],
        created_at=now,
        updated_at=now,
    )


def test_chat_tool_agent_emits_request_and_tool_call_observability_events():
    handler = RecordingObservabilityHandler()
    registry = ToolRegistry()
    registry.add_tool(FunctionTool(AddValue))
    messages = [ChatMessage.create_user_message("Use a tool.")]
    tool_call_message = assistant_tool_call_message("assistant-tool-call")
    final_message = assistant_message("assistant-final", "done")
    agent = ChatToolAgent(
        chat_api=FakeChatAPI(responses=[tool_call_message, final_message]),
        observability_handler=handler,
    )

    response = agent.get_response(messages=messages, tool_registry=registry)

    assert response.response == "done"
    assert [event["result_chat_message"].id for event in handler.requests] == [
        "assistant-tool-call",
        "assistant-final",
    ]
    assert handler.requests[0]["tool_registry"] is registry
    assert handler.requests[0]["reset_last_messages_buffer"] is True
    assert handler.requests[1]["reset_last_messages_buffer"] is False
    assert len(handler.tool_calls) == 1
    assert handler.tool_calls[0]["tool_call"] is tool_call_message.get_tool_calls()[0]
    assert handler.tool_calls[0]["tool_call_result"].tool_call_name == "AddValue"
    assert handler.tool_calls[0]["tool_call_result"].tool_call_result == "added 3"


def test_chat_tool_agent_emits_streaming_request_observability_event():
    handler = RecordingObservabilityHandler()
    messages = [ChatMessage.create_user_message("Stream a response.")]
    final_message = assistant_message("assistant-stream-final", "stream done")
    agent = ChatToolAgent(
        chat_api=FakeChatAPI(
            streaming_responses=[
                [
                    StreamingChatMessage(chunk="stream done"),
                    StreamingChatMessage(
                        chunk="",
                        finished=True,
                        finished_chat_message=final_message,
                    ),
                ]
            ]
        ),
        observability_handler=handler,
    )

    chunks = list(agent.get_streaming_response(messages=messages))

    assert chunks[-1].finished is True
    assert len(handler.streaming_requests) == 1
    assert handler.streaming_requests[0]["result_chat_message"] is final_message
    assert handler.streaming_requests[0]["reset_last_messages_buffer"] is True


def test_async_chat_tool_agent_emits_request_and_streaming_observability_events():
    async def run_agent():
        handler = RecordingObservabilityHandler()
        messages = [ChatMessage.create_user_message("Hello.")]
        final_message = assistant_message("assistant-async-final", "async done")
        streaming_final_message = assistant_message(
            "assistant-async-stream-final", "stream done"
        )
        agent = AsyncChatToolAgent(
            chat_api=FakeAsyncChatAPI(
                responses=[final_message],
                streaming_responses=[
                    [
                        StreamingChatMessage(chunk="stream done"),
                        StreamingChatMessage(
                            chunk="",
                            finished=True,
                            finished_chat_message=streaming_final_message,
                        ),
                    ]
                ],
            ),
            observability_handler=handler,
        )

        response = await agent.get_response(messages=messages)
        stream_chunks = [
            chunk
            async for chunk in agent.get_streaming_response(
                messages=messages,
                reset_last_messages_buffer=True,
            )
        ]

        assert response.response == "async done"
        assert stream_chunks[-1].finished is True
        assert handler.requests[0]["result_chat_message"] is final_message
        assert (
            handler.streaming_requests[0]["result_chat_message"]
            is streaming_final_message
        )

    asyncio.run(run_agent())

import datetime
import uuid
from typing import Optional

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.agents import AgentObservabilityHandler, ChatToolAgent
from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    TextContent,
    ToolCallContent,
    ToolCallResultContent,
)
from ToolAgents.provider.llm_provider import ChatAPIProvider, ProviderSettings


class DebugPrintObserver(AgentObservabilityHandler):
    def on_request(
        self,
        messages: list[ChatMessage],
        tool_registry: ToolRegistry,
        settings: Optional[ProviderSettings],
        reset_last_messages_buffer: bool,
        result_chat_message: ChatMessage,
    ):
        tool_names = ", ".join(tool_registry.tools.keys()) or "none"
        print("[observer] request")
        print(f"  input messages: {len(messages)}")
        print(f"  available tools: {tool_names}")
        print(f"  reset buffer: {reset_last_messages_buffer}")
        print(f"  assistant output: {result_chat_message.get_as_text()}")

    def on_streaming_request(
        self,
        messages: list[ChatMessage],
        tool_registry: ToolRegistry,
        settings: Optional[ProviderSettings],
        reset_last_messages_buffer: bool,
        result_chat_message: ChatMessage,
    ):
        print("[observer] streaming request")
        print(f"  assistant output: {result_chat_message.get_as_text()}")

    def on_tool_call(
        self, tool_call: ToolCallContent, tool_call_result: ToolCallResultContent
    ):
        print("[observer] tool call")
        print(f"  name: {tool_call.tool_call_name}")
        print(f"  arguments: {tool_call.tool_call_arguments}")
        print(f"  result: {tool_call_result.tool_call_result}")


class DemoChatAPI(ChatAPIProvider):
    """A deterministic provider so the example runs without API keys."""

    def __init__(self):
        self.settings = ProviderSettings()

    def get_response(
        self,
        messages: list[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[list[FunctionTool]] = None,
    ) -> ChatMessage:
        if any(message.contains_tool_call_results() for message in messages):
            return create_assistant_message("The tool says it is 21 degrees in Berlin.")

        return create_tool_call_message(
            tool_name="GetWeather",
            arguments={"city": "Berlin"},
        )

    def get_streaming_response(
        self,
        messages: list[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[list[FunctionTool]] = None,
    ):
        raise NotImplementedError("This basic example only uses get_response().")

    def get_default_settings(self):
        return self.settings

    def set_default_settings(self, settings) -> None:
        self.settings = settings

    def get_provider_identifier(self) -> str:
        return "demo"


def GetWeather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
        city: City to get weather for.
    """
    return f"{city}: 21 C and clear"


def create_assistant_message(text: str) -> ChatMessage:
    now = datetime.datetime.now()
    return ChatMessage(
        id=str(uuid.uuid4()),
        role=ChatMessageRole.Assistant,
        content=[TextContent(content=text)],
        created_at=now,
        updated_at=now,
    )


def create_tool_call_message(tool_name: str, arguments: dict) -> ChatMessage:
    now = datetime.datetime.now()
    return ChatMessage(
        id=str(uuid.uuid4()),
        role=ChatMessageRole.Assistant,
        content=[
            ToolCallContent(
                tool_call_id=str(uuid.uuid4()),
                tool_call_name=tool_name,
                tool_call_arguments=arguments,
            )
        ],
        created_at=now,
        updated_at=now,
    )


if __name__ == "__main__":
    agent = ChatToolAgent(
        chat_api=DemoChatAPI(),
        observability_handler=DebugPrintObserver(),
    )

    tool_registry = ToolRegistry()
    tool_registry.add_tool(FunctionTool(GetWeather))

    response = agent.get_response(
        messages=[ChatMessage.create_user_message("What is the weather in Berlin?")],
        tool_registry=tool_registry,
    )

    print()
    print(f"Final response: {response.response}")

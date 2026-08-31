from unittest.mock import MagicMock

from ToolAgents.agent_harness import HarnessEvent, create_async_harness, create_harness
from ToolAgents.agent_harness.async_harness import AsyncAgentHarness
from ToolAgents.agent_harness.harness import AgentHarness
from ToolAgents.agent_harness.prompt_composer import create_prompt_composer
from ToolAgents.agent_harness.smart_messages import SmartMessageManager
from ToolAgents.context_manager import ContextEvent, create_context_manager
from ToolAgents.data_models.messages import (
    ChatMessage,
    ChatMessageRole,
    TokenUsage,
    ToolCallContent,
    ToolCallResultContent,
)


def test_context_state_snapshot_is_not_mutable_internal_state():
    manager = create_context_manager(max_context_tokens=20, reserve_tokens=0)

    snapshot = manager.state
    snapshot.pinned_message_ids.add("message-id")

    assert not manager.is_pinned("message-id")


def test_set_pinned_message_ids_updates_real_trimming_state():
    manager = create_context_manager(max_context_tokens=20, reserve_tokens=0)
    pinned = ChatMessage.create_user_message("keep")
    removable = ChatMessage.create_user_message("drop")
    system = ChatMessage.create_system_message("system")

    manager.set_pinned_message_ids({pinned.id})
    manager.tracker.state.current_context_tokens = 90

    kept = manager.prepare_messages([system, removable, pinned])

    assert pinned in kept
    assert removable not in kept
    assert manager.is_pinned(pinned.id)


def test_sync_harness_pinned_smart_messages_survive_trimming():
    harness = AgentHarness(
        provider=MagicMock(),
        context_manager=create_context_manager(max_context_tokens=20, reserve_tokens=0),
    )
    pinned = ChatMessage.create_user_message("keep")
    removable = ChatMessage.create_user_message("drop")

    harness.add_smart_message(removable)
    harness.add_pinned_message(pinned)
    harness.context_manager.tracker.state.current_context_tokens = 90

    prepared = harness._prepare_messages()

    assert pinned in prepared
    assert removable not in prepared
    assert harness.context_manager.is_pinned(pinned.id)


def test_reset_clears_messages_context_state_and_stop_flags():
    harness = create_harness(
        provider=MagicMock(),
        max_turns=1,
        total_budget_tokens=10,
    )
    harness.add_pinned_message(ChatMessage.create_user_message("remember"))
    harness.context_manager.tracker.state.total_tokens_used = 10
    harness.context_manager.tracker.state.current_context_tokens = 20
    harness.context_manager.pin_message("external")
    harness._runtime.turn_count = 1
    harness._runtime.stopped = True
    harness._runtime.budget_exceeded = True

    harness.reset()

    assert harness.messages == []
    assert harness.turn_count == 0
    assert not harness.is_stopped
    assert harness.context_state.total_tokens_used == 0
    assert harness.context_state.current_context_tokens == 0
    assert harness.context_state.pinned_message_ids == set()


def test_async_harness_has_prompt_composer_and_smart_message_parity():
    composer = create_prompt_composer("Base")
    manager = SmartMessageManager()

    harness = AsyncAgentHarness(
        provider=MagicMock(),
        prompt_composer=composer,
        smart_message_manager=manager,
    )
    message = ChatMessage.create_system_message("short lived")

    harness.add_ephemeral_message(message, ttl=2)

    assert harness.prompt_composer is composer
    assert harness.smart_messages is manager
    assert message in harness.messages


def test_sync_and_async_factories_share_extension_catalog_behavior():
    extension_manager = MagicMock()
    extension_manager.build_catalog.return_value = "<available_skills />"
    extension_manager.get_tools.return_value = []

    sync_harness = create_harness(
        provider=MagicMock(),
        system_prompt="Base.",
        extension_manager=extension_manager,
    )
    async_harness = create_async_harness(
        provider=MagicMock(),
        system_prompt="Base.",
        extension_manager=extension_manager,
    )

    assert "Base." in sync_harness.config.system_prompt
    assert "<available_skills />" in sync_harness.config.system_prompt
    assert "Base." in async_harness.config.system_prompt
    assert "<available_skills />" in async_harness.config.system_prompt
    assert "extension_catalog" in async_harness.prompt_composer


def test_runtime_emits_turn_and_context_tool_events():
    harness = create_harness(provider=MagicMock())
    harness_events = []
    context_events = []

    harness.context_manager.events.on(ContextEvent.TOOL_CALL, context_events.append)
    harness.context_manager.events.on(ContextEvent.TOOL_RESULT, context_events.append)
    harness.events.on(HarnessEvent.TURN_START, harness_events.append)
    harness.events.on(HarnessEvent.TURN_END, harness_events.append)

    assistant = ChatMessage.create_empty_assistant_message()
    assistant.content.append(
        ToolCallContent(
            tool_call_id="call-1",
            tool_call_name="lookup",
            tool_call_arguments={},
        )
    )
    tool = ChatMessage(
        id="tool-message",
        role=ChatMessageRole.Tool,
        content=[
            ToolCallResultContent(
                tool_call_result_id="result-1",
                tool_call_id="call-1",
                tool_call_name="lookup",
                tool_call_result="done",
            )
        ],
        created_at=assistant.created_at,
        updated_at=assistant.updated_at,
    )

    harness._runtime.begin_turn("hi")
    harness._runtime.process_agent_buffer([assistant, tool])
    harness._runtime.complete_turn()

    assert [event.event for event in harness_events] == [
        "turn_start",
        "turn_end",
    ]
    assert [event.event for event in context_events] == [
        ContextEvent.TOOL_CALL,
        ContextEvent.TOOL_RESULT,
    ]


def test_budget_exceeded_still_stops_harness_when_configured():
    harness = create_harness(
        provider=MagicMock(),
        total_budget_tokens=10,
    )
    response = ChatMessage.create_assistant_message("expensive")
    response.token_usage = TokenUsage(input_tokens=8, output_tokens=3, total_tokens=11)

    harness._runtime.process_agent_buffer([response])

    assert harness.is_stopped

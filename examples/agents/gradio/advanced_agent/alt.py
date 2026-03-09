import gradio as gr
from typing import Iterator, List, Tuple
import time
import base64

from ToolAgents.data_models.messages import ChatMessageRole, ChatMessage
from agent import configurable_agent


# --- Core Logic ---

def format_chat_history_as_markdown(history: List[ChatMessage]) -> str:
    """Formats the chat history into a clean Markdown string."""
    if not history:
        return "*No messages yet. Start a conversation!*"

    md_string = ""
    for entry in history:
        role = entry.role
        content = entry.get_as_text()

        if role == ChatMessageRole.User:
            md_string += f"ðŸ‘¤ You:\n{content}\n\n\n\n"
        elif role == ChatMessageRole.Assistant:
            md_string += f"ðŸ¤– Assistant:\n{content}\n\n\n\n"
        #elif role == ChatMessageRole.System:
        #    md_string += f"âš™ï¸ System:\n```text\n{content}```\n\n\n\n"

    return md_string.strip()


def add_user_message(user_message: str, history: List[ChatMessage]):
    """Adds user message to history."""
    if not user_message.strip():
        return "", history, format_chat_history_as_markdown(history)

    updated_history = history + [ChatMessage.create_user_message(user_message)]
    return "", updated_history, format_chat_history_as_markdown(updated_history)


def stream_chat_response(history: List[ChatMessage]) -> Iterator[Tuple[List[ChatMessage], str]]:
    """Handles streaming chat responses."""
    if not history:
        return

    user_message = history[-1]
    working_history = history.copy()
    working_history.append(None)

    partial_message = ""
    for chunk in configurable_agent.stream_chat_with_agent(user_message.get_as_text()):
        partial_message += chunk.chunk
        working_history[-1] = ChatMessage.create_assistant_message(partial_message)
        yield working_history, format_chat_history_as_markdown(working_history)

    configurable_agent.save_agent()
    yield working_history, format_chat_history_as_markdown(working_history)


def clear_history():
    """Clears chat history."""
    configurable_agent.chat_history.clear_history()
    configurable_agent.save_agent()
    return [], ""


# --- Styling ---

css = """
.chat-container {
    border: 1px solid #333;
    border-radius: 8px;
    background: #1a1a1a;
    padding: 20px;
    margin: 20px 0;
}

.input-row {
    gap: 8px;
}

.gr-button {
    min-width: 80px;
}
"""

# --- Interface ---

with gr.Blocks(css=css, title="Chat Agent") as demo:
    # Header
    gr.Markdown("# ðŸ¤– Personal Chat Agent")

    # State
    initial_history = configurable_agent.get_current_chat_history()
    chat_history_state = gr.State(value=initial_history)

    # Chat Display
    chat_display = gr.Markdown(
        value=format_chat_history_as_markdown(initial_history),
        elem_classes=["chat-container"],
        line_breaks=True,
        show_copy_button=False,
    )

    # Input Section
    with gr.Row(elem_classes=["input-row"]):
        chat_input = gr.Textbox(
            placeholder="Type your message...",
            scale=4,
            show_label=False,
            autofocus=True
        )
        send_button = gr.Button("Send", scale=1, variant="primary")
        clear_button = gr.Button("Clear", scale=1, variant="secondary")

    # Events
    send_button.click(
        fn=add_user_message,
        inputs=[chat_input, chat_history_state],
        outputs=[chat_input, chat_history_state, chat_display],
        queue=False,
    ).then(
        fn=stream_chat_response,
        inputs=[chat_history_state],
        outputs=[chat_history_state, chat_display]
    )

    chat_input.submit(
        fn=add_user_message,
        inputs=[chat_input, chat_history_state],
        outputs=[chat_input, chat_history_state, chat_display],
        queue=False,
    ).then(
        fn=stream_chat_response,
        inputs=[chat_history_state],
        outputs=[chat_history_state, chat_display]
    )

    clear_button.click(
        fn=clear_history,
        outputs=[chat_history_state, chat_display]
    )

demo.launch()

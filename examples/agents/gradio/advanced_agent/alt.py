import gradio as gr
from typing import Iterator, List, Tuple
import time
import base64

from ToolAgents.data_models.messages import ChatMessageRole, ChatMessage
from agent import configurable_agent


# --- Core Logic for the new approach ---

def format_chat_history_as_markdown(history: List[ChatMessage]) -> str:
    """
    Formats the chat history into a styled Markdown string.
    """
    md_string = ""
    for entry in history:
        role = entry.role
        content = entry.get_as_text()

        if role == ChatMessageRole.User:
            md_string += f"""### User:\n{content}\n"""
        elif role == ChatMessageRole.Assistant:
            md_string += f"""### Assistant:\n{content}\n"""
        elif role == ChatMessageRole.System:
            md_string += f"""### System:\n```text\n{content}\n```\n"""
    return md_string


def add_user_message(user_message: str, history: List[Tuple[str, str]]):
    """
    Adds the user's message to the history and updates the display.
    """
    if not user_message.strip():
        return "", history, format_chat_history_as_markdown(history)
    history.append(ChatMessage.create_user_message(user_message))
    return "", history, format_chat_history_as_markdown(history)


def stream_chat_response(history: List[ChatMessage]) -> Iterator[Tuple[List[ChatMessage], str]]:
    """
    Handles streaming chat responses by updating the history and the HTML display.
    """
    user_message = history[-1]
    history.append(None)

    partial_message = ""
    for chunk in configurable_agent.stream_chat_with_agent(user_message.get_as_text()):
        partial_message += chunk.chunk
        history[-1] = ChatMessage.create_assistant_message(partial_message)
        yield history, format_chat_history_as_markdown(history)

    configurable_agent.save_agent()
    yield history, format_chat_history_as_markdown(history)


css = """
body { background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.block { background-color: #20262c; }
.gradio-container { background-color: #121212; color: #E0E0E0; }
.gr-button { background-color: #0b57d0; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-weight: bold; }
.gr-button:hover, .gr-button:active { background-color: #0a4cb5; }
.gr-textbox { background-color: #2B2B2B; color: #E0E0E0; border: 1px solid #444; border-radius: 8px; }

"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Personal Chat Agent")

    initial_history = configurable_agent.get_current_chat_history()
    chat_history_state = gr.State(value=initial_history)

    chat_display = gr.Markdown(
        value=format_chat_history_as_markdown(initial_history), padding="20px"
    )

    with gr.Row():
        chat_input = gr.Textbox(
            label="Chat Input",
            placeholder="Type your message here...",
            scale=9,
            autofocus=True,
        )
        send_button = gr.Button("Send", scale=1)

    # Event handlers
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

demo.launch()
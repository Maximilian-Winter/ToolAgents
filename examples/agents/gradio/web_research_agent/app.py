import gradio as gr
from typing import Iterator

from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole
from agent import agent, tool_registry, system_prompt

chat = ChatHistory()


def stream_chat_response(chat_history: list) -> Iterator[list]:
    """Handles streaming chat responses"""
    chat_history.append(gr.ChatMessage(role="assistant", content=""))
    partial_message = ""
    for chunk in agent.get_streaming_response(
        [
            ChatMessage.create_system_message(system_prompt),
            ChatMessage.create_user_message(chat_history[-2]["content"]),
        ],
        tool_registry=tool_registry,
    ):
        partial_message += chunk.chunk
        chat_history[-1].content = partial_message
        yield chat_history
    yield chat_history


def user(user_message, history: list):
    return "", history + [gr.ChatMessage(role="user", content=user_message)]


css = """
body {
    background-color: #121212;
    body-background-fill: #121212
    color: #E0E0E0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.block{
    background-color: #20262c;
}
.gradio-container {
    background-color: #121212;
    color: #E0E0E0;
}

.gr-button {
    background-color: #333333;
    color: #FFFFFF;
    border: 2px solid #555555;
    padding: 10px 20px;
    border-radius: 5px;
    transition: background-color 0.3s;
}

.gr-button:hover, .gr-button:active {
    background-color: #555555;
    color: #FFFFFF;
}

.gr-textbox, .gr-markdown, .gr-chatbox, .gr-file, .gr-output-textbox {
    background-color: #2B2B2B;
    color: #E0E0E0;
    border: 1px solid #444444;
    border-radius: 5px;
    padding: 10px;
}

.gr-row {
    display: flex;
    justify-content: space-between;
    gap: 20px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Web Research Chat Agent")

    def update_chat_history():
        value = []
        for chat_entry in chat.get_messages():
            if (
                chat_entry.role == ChatMessageRole.Assistant
                or chat_entry.role == ChatMessageRole.User
            ):
                value.append(
                    gr.ChatMessage(
                        role=chat_entry.role.value, content=chat_entry.get_as_text()
                    )
                )
        return value

    chatbox = gr.Chatbot(
        value=update_chat_history,
        editable="all",
        type="messages",
        show_copy_button=True,
        height=1000,
    )

    with gr.Row():
        chat_input = gr.Textbox(
            label="Chat Input",
            placeholder="Type your message here...",
            scale=9,
        )
        send_button = gr.Button("Send", scale=1)

    submit_click = send_button.click(
        fn=user,
        inputs=[chat_input, chatbox],
        outputs=[chat_input, chatbox],
        queue=False,
    ).then(stream_chat_response, chatbox, chatbox)

    chat_input.submit(
        fn=user,
        inputs=[chat_input, chatbox],
        outputs=[chat_input, chatbox],
        queue=False,
    ).then(stream_chat_response, chatbox, chatbox)

demo.queue()

if __name__ == "__main__":
    demo.launch(inbrowser=True)

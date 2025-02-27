import gradio as gr
from typing import Iterator

from ToolAgents.messages import ChatMessageRole, ChatHistory, ChatMessage
from agent import agent, tool_registry, system_prompt

chat = ChatHistory()


def stream_chat_response(chat_history: list) -> Iterator[list]:
    """Handles streaming chat responses"""
    chat_history.append(gr.ChatMessage(role="assistant", content=""))
    partial_message = ""
    # Get the streaming response from the agent
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


# Define the Gradio interface with custom CSS
css = """
body {
    background-color: #121212;  /* Dark background for the entire body */
    body-background-fill: #121212
    color: #E0E0E0;  /* Light grey text color for readability */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  /* Modern, readable font */
}

.block{
    background-color: #20262c;
}
.gradio-container {
    background-color: #121212;  /* Ensuring the container matches the body background */
    color: #E0E0E0;  /* Uniform text color throughout the app */
}

.gr-button {
    background-color: #333333;  /* Dark grey background for buttons */
    color: #FFFFFF;  /* White text for better contrast */
    border: 2px solid #555555;  /* Slightly lighter border for depth */
    padding: 10px 20px;  /* Adequate padding for touch targets */
    border-radius: 5px;  /* Rounded corners for modern feel */
    transition: background-color 0.3s;  /* Smooth transition for hover effect */
}

.gr-button:hover, .gr-button:active {
    background-color: #555555;  /* Lighter grey on hover/active for feedback */
    color: #FFFFFF;
}

.gr-textbox, .gr-markdown, .gr-chatbox, .gr-file, .gr-output-textbox {
    background-color: #2B2B2B;  /* Darker element backgrounds to distinguish from body */
    color: #E0E0E0;  /* Light grey text for readability */
    border: 1px solid #444444;  /* Slightly darker borders for subtle separation */
    border-radius: 5px;  /* Consistent rounded corners */
    padding: 10px;  /* Uniform padding for all input elements */
}

.gr-row {
    display: flex;
    justify-content: space-between;
    gap: 20px;  /* Adequate spacing between columns */
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Web Research Chat Agent")

    # Initialize chat history
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
            scale=9,  # Makes the textbox wider
        )
        send_button = gr.Button("Send", scale=1)  # Makes the button narrower

    # Event handlers
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

demo.launch()

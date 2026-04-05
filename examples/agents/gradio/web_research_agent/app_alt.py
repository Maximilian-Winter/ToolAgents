import gradio as gr
from typing import Iterator, List
import html as html_lib
import re

from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole
from agent import agent, tool_registry, system_prompt

chat = ChatHistory()


# --- HTML Formatting ---

def escape_and_format(text: str) -> str:
    """Escape HTML special characters and convert basic markdown to HTML."""
    escaped = html_lib.escape(text)

    # Fenced code blocks: ```lang\n...\n```
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        lang_label = f'<span class="code-lang">{lang}</span>' if lang else ""
        return f'<div class="code-block-wrapper">{lang_label}<pre class="code-block"><code>{code}</code></pre></div>'

    escaped = re.sub(
        r"```(\w*)\n(.*?)```", replace_code_block, escaped, flags=re.DOTALL
    )
    # Inline code
    escaped = re.sub(r"`([^`]+)`", r'<code class="inline-code">\1</code>', escaped)
    # Bold
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    # Italic
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
    # Line breaks (but not inside <pre> blocks — handled by white-space: pre there)
    escaped = escaped.replace("\n", "<br>")
    return escaped


def build_message_html(role: str, content: str) -> str:
    """Build a single message bubble's HTML."""
    formatted = escape_and_format(content)

    if role == "user":
        return f'''
        <div class="message-row user-row">
            <div class="message user-message">
                <div class="message-header">
                    <span class="role-badge user-badge">You</span>
                    <button class="copy-btn" onclick="
                        const text = this.closest('.message').querySelector('.message-content').innerText;
                        navigator.clipboard.writeText(text).then(() => {{
                            this.textContent = '✓ Copied';
                            setTimeout(() => this.textContent = 'Copy', 1500);
                        }});
                    " title="Copy message">Copy</button>
                </div>
                <div class="message-content">{formatted}</div>
            </div>
        </div>'''
    else:
        return f'''
        <div class="message-row assistant-row">
            <div class="message assistant-message">
                <div class="message-header">
                    <span class="role-badge assistant-badge">Assistant</span>
                    <button class="copy-btn" onclick="
                        const text = this.closest('.message').querySelector('.message-content').innerText;
                        navigator.clipboard.writeText(text).then(() => {{
                            this.textContent = '✓ Copied';
                            setTimeout(() => this.textContent = 'Copy', 1500);
                        }});
                    " title="Copy message">Copy</button>
                </div>
                <div class="message-content">{formatted}</div>
            </div>
        </div>'''


def render_chat(history: list) -> str:
    """Render the full chat history list into HTML."""
    if not history:
        return '<div class="empty-state">Start a conversation below.</div>'

    parts = []
    for msg in history:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if role in ("user", "assistant"):
            parts.append(build_message_html(role, content))

    return f'<div class="chat-wrapper">{"".join(parts)}</div>'


# --- Chat Logic ---

def user(user_message: str, history: list):
    """Add the user message to the history."""
    if not user_message.strip():
        return "", history, render_chat(history)
    new_history = history + [{"role": "user", "content": user_message}]
    return "", new_history, render_chat(new_history)


def stream_chat_response(history: list) -> Iterator[tuple]:
    """Stream the assistant response chunk by chunk."""
    if not history:
        return

    # Build the messages for the agent
    user_text = history[-1]["content"]

    # Add a placeholder for the assistant
    working = history + [{"role": "assistant", "content": ""}]

    partial = ""
    for chunk in agent.get_streaming_response(
        [
            ChatMessage.create_system_message(system_prompt),
            ChatMessage.create_user_message(user_text),
        ],
        tool_registry=tool_registry,
    ):
        partial += chunk.chunk
        working[-1]["content"] = partial
        yield working, render_chat(working)

    yield working, render_chat(working)


# --- Load persisted chat history ---

def load_initial_history() -> tuple:
    """Load existing chat history into our dict-based format."""
    history = []
    for entry in chat.get_messages():
        if entry.role in (ChatMessageRole.Assistant, ChatMessageRole.User):
            history.append(
                {"role": entry.role.value, "content": entry.get_as_text()}
            )
    return history, render_chat(history)


# --- Interface ---

initial_history, initial_html = load_initial_history()

with gr.Blocks(title="Web Research Chat Agent") as demo:

    gr.Markdown("## 🔍 Web Research Chat Agent")

    # State: list of {"role": ..., "content": ...} dicts
    chat_state = gr.State(value=initial_history)

    # Chat display
    chat_display = gr.HTML(
        value=initial_html,
        max_height=700,
        autoscroll=True,
        css_template="""
            /* ---- Container ---- */
            .chat-wrapper {
                display: flex;
                flex-direction: column;
                gap: 10px;
                padding: 16px 12px;
            }

            .empty-state {
                text-align: center;
                padding: 80px 20px;
                color: #71717a;
                font-size: 15px;
                font-style: italic;
            }

            /* ---- Rows ---- */
            .message-row {
                display: flex;
                width: 100%;
                animation: fadeIn 0.15s ease-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to   { opacity: 1; transform: translateY(0); }
            }
            .user-row    { justify-content: flex-end; }
            .assistant-row { justify-content: flex-start; }

            /* ---- Bubbles ---- */
            .message {
                max-width: 80%;
                padding: 10px 14px 12px;
                border-radius: 16px;
                line-height: 1.55;
                font-size: 14px;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            .user-message {
                background: #2563eb;
                color: #eef2ff;
                border-bottom-right-radius: 4px;
            }
            .assistant-message {
                background: #27272a;
                color: #e4e4e7;
                border-bottom-left-radius: 4px;
                border: 1px solid #3f3f46;
            }

            /* ---- Header row ---- */
            .message-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
            }
            .role-badge {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                opacity: 0.65;
            }
            .user-badge      { color: #bfdbfe; }
            .assistant-badge  { color: #a1a1aa; }

            /* ---- Copy button ---- */
            .copy-btn {
                background: none;
                border: 1px solid transparent;
                cursor: pointer;
                font-size: 11px;
                padding: 2px 8px;
                border-radius: 6px;
                opacity: 0;
                transition: opacity 0.2s, background 0.15s, border-color 0.15s;
                font-weight: 500;
                line-height: 1.4;
            }
            .message:hover .copy-btn {
                opacity: 0.65;
            }
            .copy-btn:hover {
                opacity: 1 !important;
                background: rgba(255,255,255,0.08);
                border-color: rgba(255,255,255,0.15);
            }
            .user-message .copy-btn      { color: #bfdbfe; }
            .assistant-message .copy-btn  { color: #a1a1aa; }

            /* ---- Content ---- */
            .message-content { white-space: normal; }

            .code-block-wrapper {
                position: relative;
                margin: 8px 0;
            }
            .code-lang {
                position: absolute;
                top: 6px;
                right: 10px;
                font-size: 10px;
                text-transform: uppercase;
                opacity: 0.45;
                font-weight: 600;
                letter-spacing: 0.04em;
            }
            .code-block {
                background: rgba(0,0,0,0.35);
                padding: 12px 14px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace;
                font-size: 13px;
                white-space: pre;
                line-height: 1.5;
            }
            .inline-code {
                background: rgba(0,0,0,0.25);
                padding: 1px 5px;
                border-radius: 4px;
                font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace;
                font-size: 13px;
            }
            .message-content strong { font-weight: 700; }
            .message-content em     { font-style: italic; }

            /* ---- Light theme ---- */
            @media (prefers-color-scheme: light) {
                .assistant-message {
                    background: #f4f4f5;
                    color: #18181b;
                    border-color: #e4e4e7;
                }
                .assistant-badge     { color: #52525b; }
                .assistant-message .copy-btn { color: #52525b; }
                .assistant-message .code-block {
                    background: rgba(0,0,0,0.05);
                }
                .assistant-message .inline-code {
                    background: rgba(0,0,0,0.06);
                }
                .copy-btn:hover {
                    background: rgba(0,0,0,0.06);
                    border-color: rgba(0,0,0,0.12);
                }
            }
        """,
    )

    # Input row
    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=9,
            autofocus=True,
        )
        send_button = gr.Button("Send", scale=1, variant="primary")

    # Wire events
    send_button.click(
        fn=user,
        inputs=[chat_input, chat_state],
        outputs=[chat_input, chat_state, chat_display],
        queue=False,
    ).then(
        fn=stream_chat_response,
        inputs=[chat_state],
        outputs=[chat_state, chat_display],
    )

    chat_input.submit(
        fn=user,
        inputs=[chat_input, chat_state],
        outputs=[chat_input, chat_state, chat_display],
        queue=False,
    ).then(
        fn=stream_chat_response,
        inputs=[chat_state],
        outputs=[chat_state, chat_display],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(inbrowser=True)
import gradio as gr
import os
import shutil

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.knowledge.default_providers import (
    PDFOCRProvider,
    ChromaDbVectorDatabaseProvider,
    SentenceTransformerEmbeddingProvider,
    MXBAIRerankingProvider,
)
from ToolAgents.utilities import SimpleTextSplitter
from ToolAgents.messages.chat_history import ChatHistory
from agent import answer_agent

has_ingested = False
vector_database = ChromaDbVectorDatabaseProvider(
    SentenceTransformerEmbeddingProvider(), MXBAIRerankingProvider()
)
pdf_provider = PDFOCRProvider("uploaded_files", SimpleTextSplitter(512, 0))
history = ChatHistory()
history.add_system_message(
    "Your task is write detailed and comprehensive answers to the requests of the user, based on PDFs the user uploaded. Use the 'query_pdf_information' tool to retrieve information from the PDF."
)


def query_pdf_information(query: str):
    """
    Query information from PDFs uploaded by the user.
    Args:
        query (str): The query to use for searching information.
    Returns:
        results (list[str]): The results of the query.
    """
    global vector_database
    result = vector_database.query(query, k=25)
    return result.chunks


def ingest():
    """
    Ingest documents and update interface visibility state
    Returns:
        tuple: Status message and updated visibility values
    """
    global has_ingested, pdf_provider, vector_database
    upload_folder = "uploaded_files"
    if not os.path.isdir(upload_folder) or not os.listdir(upload_folder):
        return (
            "No uploaded files!",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    documents = pdf_provider.get_documents()
    vector_database.add_documents(documents)
    has_ingested = True

    return (
        "Documents pre-processed successfully. You can now start the chat.",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


# Function to handle chat messages
def chat_response(message, chat_history=None):
    if chat_history is None:
        chat_history = []
    global has_ingested, history
    if not has_ingested:
        return "Please pre-process the files before!", chat_history

    tools = [FunctionTool(query_pdf_information)]
    tool_registry = ToolRegistry()

    tool_registry.add_tools(tools)

    history.add_user_message(message)
    history_list = history.get_messages()
    response = answer_agent.get_response(
        messages=history_list, tool_registry=tool_registry
    )

    chat_history.append(gr.ChatMessage(role="user", content=message))
    chat_history.append(gr.ChatMessage(role="assistant", content=response.response))
    history.add_messages(response.messages)
    return "", chat_history


# Function to handle file uploads
def upload_files(files):
    if files is None:
        return "No files to upload."
    upload_folder = "uploaded_files"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    for file in files:
        shutil.copy(file.name, os.path.join(upload_folder, os.path.basename(file.name)))
    return f"{len(files)} file(s) uploaded successfully."


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

.gr-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;  /* Consistent gap between widgets within a column */
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Chat with Documents")

    with gr.Row():
        with gr.Column():
            chatbox = gr.Chatbot(type="messages", show_copy_button=True, visible=False)
            chat_input = gr.Textbox(
                label="Chat Input",
                placeholder="Type your message here...",
                visible=False,
            )
            send_button = gr.Button("Send", visible=False)

        with gr.Column():
            status_output = gr.Textbox(
                label="Document Status",
                interactive=False,
                placeholder="Please upload your documents and pre-process them below",
            )
            file_uploader = gr.File(label="Upload Files", file_count="multiple")
            upload_button = gr.Button("Upload")
            ingest_documents = gr.Button("Preprocess documents")
            clear_button = gr.Button("Delete uploaded documents")

    # Chat response handler
    send_button.click(chat_response, [chat_input, chatbox], [chat_input, chatbox])

    # Ingest documents handler with proper visibility updates
    ingest_documents.click(
        ingest, outputs=[status_output, chatbox, chat_input, send_button]
    )

    def clear():
        global has_ingested
        has_ingested = False
        upload_folder = "uploaded_files"
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        return (
            "Documents cleared.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # Clear button handler
    clear_button.click(clear, outputs=[status_output, chatbox, chat_input, send_button])

    # File upload handler
    upload_button.click(upload_files, [file_uploader], status_output)

demo.launch()

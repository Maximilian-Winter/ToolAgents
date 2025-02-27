import requests

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.knowledge.vector_database import RAG
from ToolAgents.knowledge.vector_database.implementations.chroma_db import (
    ChromaDbVectorDatabaseProvider,
)
from ToolAgents.knowledge.vector_database.implementations.mbxai_reranking import (
    MXBAIRerankingProvider,
)
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)
from ToolAgents.messages import ChatMessage
from ToolAgents.provider import CompletionProvider
from ToolAgents.provider.completion_provider.default_implementations import (
    LlamaCppServer,
)


def get_wikipedia_page(title: str):
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    headers = {"User-Agent": "MadWizard"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


from typing import List

from pydantic import BaseModel, Field

from ToolAgents.knowledge.text_processing.text_splitter import (
    RecursiveCharacterTextSplitter,
)

rag = RAG(
    ChromaDbVectorDatabaseProvider(
        SentenceTransformerEmbeddingProvider(), MXBAIRerankingProvider()
    )
)

# Initialize a recursive character text splitter with the correct chunk size of the embedding model.
length_function = len
splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n\n\n\n",
        "\n\n\n\n",
        "\n\n\n",
        "\n\n",
        "\n",
        ".",
        "!",
        "?",
        " ",
        "",
    ],
    chunk_size=512,
    chunk_overlap=30,
    length_function=length_function,
    keep_separator=True,
)

# Use the helper function to get the content of a wikipedia page.
page = get_wikipedia_page("Synthetic_diamond")

# Split the text of the wikipedia page into chunks for the vector database.
splits = splitter.get_chunks(page)

# Add the splits into the vector database
for split in splits:
    if split.strip() == "":
        continue
    rag.add_document(split)

api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))

agent = ChatToolAgent(chat_api=api, debug_output=True)

system_message = "You are an advanced AI assistant, trained by OpenAI."


settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.75

api.set_default_settings(settings)
# Define the query we want to ask based on the retrieved information
query = "What is a BARS apparatus?"

chat_history = [
    ChatMessage.create_system_message(system_message),
    ChatMessage.create_user_message(query),
]

# Ask the query without retrieved information.
result = agent.get_response(chat_history)
print(result.response)


# Define a pydantic class to represent a query extension as additional queries to the original query.
class QueryExtension(BaseModel):
    """
    Represents an extension of a query as additional queries.
    """

    queries: List[str] = Field(default_factory=list, description="List of queries.")


output_settings = QueryExtension()


def create_query_extension(query_extension: QueryExtension):
    """
    Creates a query extension from the given query extension.
    Args:
        query_extension (QueryExtension): The query extension.
    """
    global output_settings
    output_settings = query_extension


# Define a query extension agent which will extend the query with additional queries.
system_message = "You are a world class query extension algorithm capable of extending queries by writing new queries. Do not answer the queries, use your 'create_query_extension' tool to create new queries."

chat_history = [
    ChatMessage.create_system_message(system_message),
    ChatMessage.create_user_message(f"Consider the following query: {query}"),
]

tool_registry = ToolRegistry()

tool_registry.add_tool(FunctionTool(create_query_extension))
# Perform the query extension with the agent.
output = agent.get_response(chat_history, tool_registry=tool_registry)
print(output.response)

# Load the query extension in JSON format and create an instance of the query extension model.
queries = output_settings

# Define the final prompt for the query with the retrieved information
prompt = "Consider the following context:\n==========Context===========\n"

# Retrieve the most fitting document chunks based on the original query and add them to the prompt.
documents = rag.retrieve_documents(query, k=3)
for doc in documents.chunks:
    prompt += doc + "\n\n"

# Retrieve the most fitting document chunks based on the extended queries and add them to the prompt.
for qu in queries.queries:
    documents = rag.retrieve_documents(qu, k=3)
    for doc in documents.chunks:
        if doc["text"] not in prompt:
            prompt += doc["text"] + "\n\n"
prompt += "\n======================\nQuestion: " + query


# Define a query extension agent which will extend the query with additional queries.
system_message = "You are an advanced AI assistant, trained by OpenAI. Only answer question based on the context information provided."
chat_history = [
    ChatMessage.create_system_message(system_message),
    ChatMessage.create_user_message(prompt),
]

# Perform the query extension with the agent.
output = agent.get_response(chat_history)
print(output.response)

import requests

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import MistralAgent


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

# Define a llamacpp server endpoint.
from ToolAgents.provider import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

# Define a test agent to see the answer without retrieved information.
agent = MistralAgent(model, debug_output=True)

settings = model.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.75

model.set_default_settings(settings)


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


tool_registry = ToolRegistry()

tool_registry.add_tool(FunctionTool(create_query_extension))
# Perform the query extension with the agent.

for _ in range(100):
    query = "What is a BARS apparatus?"
    chat_history = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Consider the following query: {query}"},
    ]
    output = agent.step(chat_history, tool_registry=tool_registry)
    print(output)

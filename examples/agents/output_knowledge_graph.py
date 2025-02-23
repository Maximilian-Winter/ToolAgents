# Example that uses the StructuredOutputAgent class to create a dataset entry of a book, out of unstructured data.
import json
from typing import List

from graphviz import Digraph
from pydantic import BaseModel, Field, TypeAdapter

from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.json_schema_generator.schema_generator import custom_json_schema

api = OpenAIChatAPI(api_key="unknown", base_url="http://127.0.0.1:8080/v1",
                    model="unknown")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0
settings.set_max_new_tokens(16378)

class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)



def visualize_knowledge_graph(kg):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph6.gv", view=True)


def generate_graph(user_input: str):
    prompt = f"""Help me understand the following by describing it as a extremely detailed knowledge graph with at least 20 nodes: {user_input}""".strip()
    schema = custom_json_schema(KnowledgeGraph)
    settings.set_response_format({"type": "json_object", "schema": schema})
    messages = [
        ChatMessage.create_system_message(
            f"""You are knowledge graph builder. You will build the knowledge graph according to the following JSON-Schema:\n\n{json.dumps(schema, indent=2)}"""),
        ChatMessage.create_user_message(
            prompt)
    ]

    chat_response = agent.get_response(
        messages=messages,
        settings=settings)

    print(chat_response.response.strip())

    return KnowledgeGraph(**json.loads(chat_response.response))


graph = generate_graph("Industrial Military Complex")
visualize_knowledge_graph(graph)



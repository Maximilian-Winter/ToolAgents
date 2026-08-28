"""Generate a small knowledge graph and write it as a DOT file.

Set OPENROUTER_API_KEY to run the example. The generated DOT file is ignored by
git so local runs do not create tracked artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.json_schema_generator.schema_generator import custom_json_schema


api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("Set OPENROUTER_API_KEY to run this example.")

api = OpenAIChatAPI(
    api_key=api_key,
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)

agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.3
settings.top_p = 1.0


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
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_dot_file(graph: KnowledgeGraph, path: Path) -> None:
    lines = ["digraph KnowledgeGraph {"]
    for node in graph.nodes:
        lines.append(
            f'  {node.id} [label="{dot_escape(node.label)}", color="{dot_escape(node.color)}"];'
        )
    for edge in graph.edges:
        lines.append(
            f'  {edge.source} -> {edge.target} [label="{dot_escape(edge.label)}", color="{dot_escape(edge.color)}"];'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_graph(user_input: str) -> KnowledgeGraph:
    schema = custom_json_schema(KnowledgeGraph)
    settings.response_format = KnowledgeGraph
    messages = [
        ChatMessage.create_system_message(
            f"""Build a concise knowledge graph as JSON matching this schema.

```json
{json.dumps(schema, indent=2)}
```"""
        ),
        ChatMessage.create_user_message(
            f"Describe this topic as a knowledge graph with 8 to 12 nodes: {user_input}"
        ),
    ]

    chat_response = agent.get_response(messages=messages, settings=settings)
    return KnowledgeGraph.model_validate_json(chat_response.response)


graph = generate_graph("The Industrial Military Complex")
output_path = Path(__file__).with_name("generated_knowledge_graph.gv")
write_dot_file(graph, output_path)
print(f"Wrote {output_path}")


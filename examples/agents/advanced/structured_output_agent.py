"""Extract a typed Pydantic object from model output.

Set OPENROUTER_API_KEY to run the example. Override OPENROUTER_MODEL and
OPENROUTER_BASE_URL if you want a different OpenAI-compatible provider.
"""

from __future__ import annotations

import json
import os
from enum import Enum

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
settings.temperature = 0.2
settings.top_p = 1.0


class Category(Enum):
    FICTION = "Fiction"
    NON_FICTION = "Non-Fiction"


class Book(BaseModel):
    """Structured book metadata extracted from prose."""

    book_title: str = Field(..., description="Title of the book.")
    author: str = Field(..., description="Author of the book.")
    published_year: int = Field(..., description="Publishing year of the book.")
    keywords: list[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


schema = custom_json_schema(model=Book)
settings.response_format = Book

messages = [
    ChatMessage.create_system_message(
        f"""Extract book information as JSON that matches this schema.

```json
{json.dumps(schema, indent=2)}
```"""
    ),
    ChatMessage.create_user_message(
        """The book 'The Feynman Lectures on Physics' is a physics textbook
based on lectures by Richard Feynman, a Nobel laureate sometimes called
"The Great Explainer". The lectures were presented to undergraduates at the
California Institute of Technology from 1961 to 1963. The co-authors are
Feynman, Robert B. Leighton, and Matthew Sands."""
    ),
]

chat_response = agent.get_response(messages=messages, settings=settings)

print(chat_response.response)
book = Book.model_validate_json(chat_response.response)
print(book)
print(json.dumps(book.model_dump(), indent=2))


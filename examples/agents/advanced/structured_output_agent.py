import json
import os

from enum import Enum
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.utilities.json_schema_generator.schema_generator import (
    custom_json_schema,
)
load_dotenv()
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", model="openai/o3-mini"
)

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.4
settings.top_p = 1.0
settings.set_max_new_tokens(8192)


# Example enum for our output model
class Category(Enum):
    Fiction = "Fiction"
    NonFiction = "Non-Fiction"


# Example output model
class Book(BaseModel):
    """
    Represents an entry about a book.
    """

    book_title: str = Field(..., description="Title of the book.")
    author: str = Field(..., description="Author of the book.")
    published_year: int = Field(..., description="Publishing year of the book.")
    keywords: List[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


schema = custom_json_schema(model=Book)

print(json.dumps(schema, indent=2))

settings.set_response_format({"type": "json_object", "schema": schema})
messages = [
    ChatMessage.create_system_message(
        f"""You are an advanced information extraction system designed to extract structured data from unstructured or semi-structured input. Your task is to extract user information based on a provided JSON schema and format it according to that schema.

Here is the JSON schema that defines the structure of the information you need to extract:

<json_schema>
{json.dumps(schema, indent=2)}
</json_schema>"""
    ),
    ChatMessage.create_user_message(
        """The book 'The Feynman Lectures on Physics' is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""
    ),
]

chat_response = agent.get_response(messages=messages, settings=settings)

print(chat_response.response, flush=True)

json_data = json.loads(chat_response.response)

book = Book(**json_data)

print(book)
print(json.dumps(json_data, indent=2))

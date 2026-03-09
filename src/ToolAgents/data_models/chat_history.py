import json
import uuid
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ToolAgents.data_models.messages import ChatMessageRole
from ToolAgents.data_models.messages import (
    ChatMessage,
    TextContent,
)


class ChatHistory(BaseModel):
    """
    Model for managing chat history with functionality to load and save to JSON files.

    Attributes:
        id (str): Unique identifier of the chat history.
        title (str): Title of the chat history.
        messages: List of chat messages in the history
        metadata: Optional metadata about the chat history
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID")
    title: str = Field(default_factory=lambda: "New Chat", description="Chat title")
    messages: List[ChatMessage] = Field(
        default_factory=list, description="List of chat messages"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the chat history"
    )

    def clear(self):
        self.messages = []

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def add_system_message(self, message: str) -> None:
        date = datetime.now()
        self.messages.append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.System,
                content=[TextContent(content=message)],
                created_at=date,
                updated_at=date,
            )
        )

    def add_user_message(self, message: str) -> None:
        date = datetime.now()
        self.messages.append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.User,
                content=[TextContent(content=message)],
                created_at=date,
                updated_at=date,
            )
        )

    def add_assistant_message(self, message: str) -> None:
        date = datetime.now()
        self.messages.append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role=ChatMessageRole.Assistant,
                content=[TextContent(content=message)],
                created_at=date,
                updated_at=date,
            )
        )

    def add_messages(self, messages: List[ChatMessage]) -> None:
        self.messages.extend(messages)

    def add_message_from_dictionary(self, message: Dict[str, str]) -> None:
        self.messages.extend(ChatMessage.from_dictionaries([message]))

    def add_messages_from_dictionaries(self, messages: List[Dict[str, str]]) -> None:
        self.messages.extend(ChatMessage.from_dictionaries(messages))

    def get_messages(self) -> List[ChatMessage]:
        return self.messages

    def remove_last_message(self):
        self.messages.remove(self.messages[-1])

    def get_last_k_messages(self, k: int) -> List[ChatMessage]:
        if k < 0:
            raise ValueError("k must be non-negative")
        return self.messages[-k:]

    def clear_history(self) -> None:
        self.messages.clear()

    def save_to_json(self, filepath: str) -> None:
        data = self.model_dump()

        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "ChatHistory":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for message in data["messages"]:
            message["created_at"] = datetime.fromisoformat(message["created_at"])
            message["updated_at"] = datetime.fromisoformat(message["updated_at"])

        return cls(**data)

    def get_last_message(self) -> ChatMessage | None:
        if self.messages:
            return self.messages[-1]
        return None

    def get_message_count(self) -> int:
        return len(self.messages)


class Chats(BaseModel):
    chats: Dict[str, ChatHistory] = Field(
        default_factory=dict, description="List of chats"
    )
    chats_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the chats"
    )

    def create_chat(self, title: str):
        chat_id = str(uuid.uuid4())
        chat = ChatHistory(id=chat_id, title=title)
        self.chats[chat_id] = chat
        return chat_id

    def get_messages(self, chat_id: str) -> List[ChatMessage]:
        return self.chats[chat_id].get_messages()

    def get_last_k_messages(self, chat_id: str, k: int) -> List[ChatMessage]:
        return self.chats[chat_id].get_last_k_messages(k)

    def add_message(self, chat_id: str, message: ChatMessage) -> None:
        self.chats[chat_id].add_message(message)

    def add_system_message(self, chat_id: str, message: str) -> None:
        self.chats[chat_id].add_system_message(message)

    def add_user_message(self, chat_id: str, message: str) -> None:
        self.chats[chat_id].add_user_message(message)

    def add_assistant_message(self, chat_id: str, message: str) -> None:
        self.chats[chat_id].add_assistant_message(message)

    def save_to_json(self, filepath: str) -> None:
        data = self.model_dump()

        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "Chats":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, chat in data["chats"].items():
            for message in chat["messages"]:
                message["created_at"] = datetime.fromisoformat(message["created_at"])
                message["updated_at"] = datetime.fromisoformat(message["updated_at"])

        return cls(**data)


if __name__ == "__main__":
    test_chats = Chats()
    test_chat_id = test_chats.create_chat("New Chat")
    test_date = datetime.now()
    test_message = ChatMessage(
        id="1",
        role=ChatMessageRole.User,
        content=[TextContent(content="Hello, how can I help you today?")],
        created_at=test_date,
        updated_at=test_date,
    )

    test_chats.add_message(test_chat_id, test_message)
    test_chats.save_to_json("chat_history.json")
    test_loaded_history = Chats.load_from_json("chat_history.json")

    print(
        test_loaded_history.get_last_k_messages(test_chat_id, 1)[0].model_dump_json(
            indent=2
        )
    )

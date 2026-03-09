import json
import uuid
from collections.abc import Iterable
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
        """
        Clears the chat history.
        """
        self.messages = []

    def add_message(self, message: ChatMessage) -> None:
        """
        Add a new message to the chat history.

        Args:
            message: ChatMessage object to add
        """
        self.messages.append(message)

    def add_system_message(self, message: str) -> None:
        """
        Add a new system message to the chat history.

        Args:
            message: ChatMessage object to add
        """
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
        """
        Add a new user message to the chat history.

        Args:
            message: ChatMessage object to add
        """
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
        """
        Add a new assistant message to the chat history.

        Args:
            message: ChatMessage object to add
        """
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
        """
        Add a new messages to the chat history.

        Args:
            messages: ChatMessage list to add
        """
        self.messages.extend(messages)

    def add_message_from_dictionary(self, message: Dict[str, str]) -> None:
        self.messages.extend(ChatMessage.from_dictionaries([message]))

    def add_messages_from_dictionaries(self, messages: List[Dict[str, str]]) -> None:
        self.messages.extend(ChatMessage.from_dictionaries(messages))

    # Backwards-compatible alias used throughout older examples/tests.
    def add_list_of_dicts(self, messages: Iterable[Dict[str, Any] | ChatMessage]) -> None:
        for message in messages:
            if isinstance(message, ChatMessage):
                self.messages.append(message)
            else:
                self.add_message_from_dictionary(message)

    def get_messages(self) -> List[ChatMessage]:
        """
        Get all messages in the chat history.

        Returns:
            List of ChatMessage objects
        """
        return self.messages

    def remove_last_message(self):
        self.messages.remove(self.messages[-1])

    def get_last_k_messages(self, k: int) -> List[ChatMessage]:
        """
        Get the last k messages from the chat history.

        Args:
            k: Number of most recent messages to return

        Returns:
            List of the last k ChatMessage objects, or all messages if k > total messages

        Raises:
            ValueError: If k is negative
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        return self.messages[-k:]

    def clear_history(self) -> None:
        """Clear all messages from the chat history."""
        self.messages.clear()

    def save_to_json(self, filepath: str) -> None:
        """
        Save the chat history to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        data = self.model_dump()

        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)

    # Backwards-compatible alias used by older examples/tests.
    def save_history(self, filepath: str) -> None:
        self.save_to_json(filepath)

    @classmethod
    def load_from_json(cls, filepath: str) -> "ChatHistory":
        """
        Load chat history from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            ChatHistory object with loaded messages
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Older history files stored a simple list of role/content dictionaries.
        if isinstance(data, list):
            return cls(messages=ChatMessage.from_dictionaries(data))

        for message in data["messages"]:
            message["created_at"] = datetime.fromisoformat(message["created_at"])
            message["updated_at"] = datetime.fromisoformat(message["updated_at"])

        return cls(**data)

    # Backwards-compatible alias used by older examples/tests.
    def load_history(self, filepath: str) -> None:
        loaded_history = self.load_from_json(filepath)
        self.id = loaded_history.id
        self.title = loaded_history.title
        self.messages = loaded_history.messages
        self.metadata = loaded_history.metadata

    def get_last_message(self) -> ChatMessage | None:
        """
        Get the last message in the chat history.

        Returns:
            Last ChatMessage or None if history is empty
        """
        if self.messages:
            return self.messages[-1]
        return None

    def get_message_count(self) -> int:
        """
        Get the total number of messages in the chat history.

        Returns:
            Number of messages
        """
        return len(self.messages)

    # Backwards-compatible alias used by older examples/tests.
    def to_list(self) -> List[ChatMessage]:
        return self.get_messages()


class Chats(BaseModel):
    """
    Model for managing multiple chat histories with functionality to load and save to JSON files.
    Attributes:
        chats: List of chat histories in the history
    """

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
        """
        Save all chats to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
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
        """
        Load chats from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Chats object with loaded chat histories
        """
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

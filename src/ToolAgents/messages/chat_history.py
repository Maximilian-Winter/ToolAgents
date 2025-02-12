import json
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ToolAgents.messages.chat_message import ChatMessageRole
from chat_message import ChatMessage, TextContent, BinaryContent, ToolCallContent, ToolCallResultContent


class ChatHistory(BaseModel):
    """
    Model for managing chat history with functionality to load and save to JSON files.

    Attributes:
        messages: List of chat messages in the history
        metadata: Optional metadata about the chat history
    """
    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chat history"
    )

    def add_message(self, message: ChatMessage) -> None:
        """
        Add a new message to the chat history.

        Args:
            message: ChatMessage object to add
        """
        self.messages.append(message)

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

    def get_messages(self) -> List[ChatMessage]:
        """
        Get all messages in the chat history.

        Returns:
            List of ChatMessage objects
        """
        return self.messages

    def clear_history(self) -> None:
        """Clear all messages from the chat history."""
        self.messages.clear()

    def save_to_json(self, filepath: str) -> None:
        """
        Save the chat history to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        # Convert to dictionary and handle datetime serialization
        data = self.model_dump()

        # Custom JSON encoder to handle datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)

        # Write to file with pretty printing
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'ChatHistory':
        """
        Load chat history from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            ChatHistory object with loaded messages
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert ISO format strings back to datetime objects
        for message in data['messages']:
            message['created_at'] = datetime.fromisoformat(message['created_at'])
            message['updated_at'] = datetime.fromisoformat(message['updated_at'])

        return cls(**data)

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


# Example usage
if __name__ == "__main__":
    # Create a new chat history
    history = ChatHistory()

    # Create a sample message
    message = ChatMessage(
        id="1",
        role=ChatMessageRole.User,
        content=[
            TextContent(content="Hello, how can I help you today?")
        ],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    # Add message to history
    history.add_message(message)

    # Save to JSON
    history.save_to_json("chat_history.json")

    # Load from JSON
    loaded_history = ChatHistory.load_from_json("chat_history.json")

    # Print the loaded message
    print(loaded_history.get_last_message().model_dump_json(indent=2))
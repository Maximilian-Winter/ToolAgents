import datetime

from sqlalchemy import Column, Integer, Text, DateTime, Enum
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.declarative import declarative_base

import json

from ToolAgents.utilities.chat_history import ChatMessageRole

Base = declarative_base()


class ChatMessageDB(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    role = Column(Enum(ChatMessageRole))
    timestamp = Column(DateTime, index=True)
    content = Column(Text)
    message_keywords = Column(JSON)  # Storing keywords as JSON string

    def __str__(self):
        content = (
            f'Timestamp: {self.timestamp.strftime("%Y-%m-%d %H:%M")}\nType: {self.role.value}\n\n{self.content}'
        )
        return content

    def add_keyword(self, keyword):
        """Add a keyword to the event."""
        if self.message_keywords:
            keywords = json.loads(self.message_keywords)
        else:
            keywords = []
        keywords.append(keyword)
        self.message_keywords = json.dumps(keywords)

    def to_dict(self):
        return {
            "event_type": self.role.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "event_keywords": self.message_keywords,
        }

    @staticmethod
    def from_dict(data):
        return Event(
            event_type=ChatMessageRole(data["event_type"]),
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            event_keywords=data["metadata"],
        )

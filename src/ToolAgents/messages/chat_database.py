import datetime
from contextlib import contextmanager
from typing import List, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from sqlalchemy.exc import IntegrityError

from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent


class Base(DeclarativeBase):
    @property
    def created(self) -> datetime.datetime:
        return self.created_at

    @property
    def updated(self) -> datetime.datetime:
        return self.updated_at


class Chat(Base):
    __tablename__ = 'chats'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    # Relationship to messages
    messages = relationship("ChatMessageDb", back_populates="chat", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'messages': [message.to_dict() for message in self.messages]
        }


class ChatMessageDb(Base):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey('chats.id'), nullable=False)
    message_id = Column(String, nullable=False)  # Original UUID from the message
    role = Column(String, nullable=False)
    content = Column(JSON, nullable=False)  # Stores the entire content list as JSON
    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    additional_information = Column(JSON, default=dict)

    # Relationship to chat
    chat = relationship("Chat", back_populates="messages")

    def to_dict(self):
        return {
            'id': self.id,
            'chat_id': self.chat_id,
            'message_id': self.message_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'additional_information': self.additional_information
        }


class ChatManager:
    def __init__(self, db_url='sqlite:///chats.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def create_chat(self, title: Optional[str] = None) -> dict[str, Any]:
        """Create a new chat."""
        with self.session_scope() as session:
            chat = Chat(title=title)
            session.add(chat)
            session.flush()
            return chat.to_dict()

    def add_message(self, chat_id: int, message: ChatMessage) -> dict[str, Any]:
        """Add a message to a chat."""
        with self.session_scope() as session:
            db_message = ChatMessageDb(
                chat_id=chat_id,
                message_id=message.id,
                role=message.role,
                content=[c.model_dump() for c in message.content],
                created_at=message.created_at,
                updated_at=message.updated_at,
                additional_information=message.additional_information
            )
            session.add(db_message)
            session.flush()
            return db_message.to_dict()

    def get_chat(self, chat_id: int) -> Optional[dict]:
        """Get a chat by ID including all its messages."""
        with self.session_scope() as session:
            chat = session.query(Chat).filter_by(id=chat_id).first()
            return chat.to_dict() if chat else None

    def get_chat_messages(self, chat_id: int) -> List[dict]:
        """Get all messages for a chat."""
        with self.session_scope() as session:
            messages = session.query(ChatMessageDb).filter_by(chat_id=chat_id).order_by(ChatMessageDb.created_at).all()
            return [message.to_dict() for message in messages]

    def delete_chat(self, chat_id: int) -> bool:
        """Delete a chat and all its messages."""
        with self.session_scope() as session:
            chat = session.query(Chat).filter_by(id=chat_id).first()
            if chat:
                session.delete(chat)
                return True
            return False

    def update_chat_title(self, chat_id: int, title: str) -> bool:
        """Update a chat's title."""
        with self.session_scope() as session:
            chat = session.query(Chat).filter_by(id=chat_id).first()
            if chat:
                chat.title = title
                return True
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()


# Example usage:
if __name__ == "__main__":


    # Initialize the manager
    manager = ChatManager()

    # Create a new chat
    chat = manager.create_chat("Test Chat")

    # Create a message using your ChatMessage class
    message = ChatMessage(
        id="123",
        role=ChatMessageRole.Assistant,
        content=[TextContent(content="Test Content")],
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now()
    )

    # Add the message to the chat
    manager.add_message(chat['id'], message)

    # Retrieve the chat with its messages
    retrieved_chat = manager.get_chat(chat['id'])
    print(retrieved_chat)
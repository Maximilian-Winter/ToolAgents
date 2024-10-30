import json

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import relationship
from contextlib import contextmanager
import datetime
import enum
from typing import List, Dict, Any, Optional

Base = declarative_base()


class MessageRole(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.now())
    name = Column(String, nullable=True)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'))
    role = Column(Enum(MessageRole))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.now())
    message_metadata = Column(Text, nullable=True)  # Store additional kwargs as JSON
    session = relationship("ChatSession", back_populates="messages")


class PersistentChatHistory:
    def __init__(self, db_url='sqlite:///chat_history.db'):
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
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_session(self, name: Optional[str] = None) -> int:
        """Create a new chat session and return its ID."""
        with self.session_scope() as session:
            chat_session = ChatSession(name=name)
            session.add(chat_session)
            session.flush()
            return chat_session.id

    def add_message(self, session_id: int, role: str, content: str, **kwargs):
        """Add a message to a specific chat session."""
        with self.session_scope() as session:
            message = ChatMessage(
                session_id=session_id,
                role=MessageRole(role),
                content=content,
                message_metadata=json.dumps(kwargs) if kwargs else json.dumps({})
            )
            session.add(message)

    def add_user_message(self, session_id: int, content: str):
        self.add_message(session_id, "user", content)

    def add_assistant_message(self, session_id: int, content: str, **kwargs):
        self.add_message(session_id, "assistant", content, **kwargs)

    def add_system_message(self, session_id: int, content: str):
        self.add_message(session_id, "system", content)

    def add_tool_message(self, session_id: int, content: str, **kwargs):
        self.add_message(session_id, "tool", content, **kwargs)

    def get_messages(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a specific chat session."""
        with self.session_scope() as session:
            messages = session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
            return [
                {"role": msg.role.value,
                 "content": msg.content,
                 **json.loads(msg.message_metadata)
                 } for msg in messages]

    def delete_last_messages(self, session_id: int, k: int) -> int:
        """Delete the last k messages from a chat session."""
        with self.session_scope() as session:
            messages = session.query(ChatMessage).filter_by(session_id=session_id).order_by(
                ChatMessage.timestamp.desc()).limit(k).all()
            deleted = len(messages)
            for message in messages:
                session.delete(message)

            return deleted

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all chat sessions."""
        with self.session_scope() as session:
            sessions = session.query(ChatSession).all()
            return [
                {
                    "id": s.id,
                    "created_at": s.created_at,
                    "name": s.name,
                    "message_count": len(s.messages)
                }
                for s in sessions
            ]

    def delete_session(self, session_id: int):
        """Delete a chat session and all its messages."""
        with self.session_scope() as session:
            chat_session = session.query(ChatSession).filter_by(id=session_id).first()
            if chat_session:
                session.delete(chat_session)

    def save_session_to_file(self, session_id: int, filename: str):
        """Save a chat session to a JSON file."""
        messages = self.get_messages(session_id)
        with open(filename, 'w') as f:
            json.dump(messages, f, indent=2, default=str)

    def load_session_from_file(self, filename: str) -> int:
        """Load messages from a JSON file into a new chat session."""
        with open(filename, 'r') as f:
            messages = json.load(f)

        session_id = self.create_session()
        for msg in messages:
            role = msg.pop('role')
            content = msg.pop('content')
            self.add_message(session_id, role, content, **msg)

        return session_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()


# Example usage
if __name__ == "__main__":
    with PersistentChatHistory() as chat_history:
        # Create a new chat session
        session_id = chat_history.create_session("Example Chat")

        # Add various types of messages
        chat_history.add_system_message(session_id, "You are a helpful assistant.")
        chat_history.add_user_message(session_id, "Hello! Can you help me?")
        chat_history.add_assistant_message(session_id, "Of course! What can I help you with?")
        chat_history.add_tool_message(session_id, "Calculator result: 42", tool_name="calculator")

        # Retrieve and print messages
        messages = chat_history.get_messages(session_id)
        print("Chat messages:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")

        # Save session to file
        chat_history.save_session_to_file(session_id, "chat_backup.json")

        # Load session from file
        new_session_id = chat_history.load_session_from_file("chat_backup.json")

        # Get all sessions
        sessions = chat_history.get_all_sessions()
        print("\nAll chat sessions:")
        for session in sessions:
            print(f"Session {session['id']}: {session['name']} "
                  f"({session['message_count']} messages)")
            msgs = chat_history.get_messages(session['id'])
            print(json.dumps(msgs, indent=2))

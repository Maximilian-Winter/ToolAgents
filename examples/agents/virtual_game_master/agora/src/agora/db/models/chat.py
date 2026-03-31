from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from agora.db.base import Base
from agora.db.models.enums import MessageType


class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(100), nullable=False, index=True)
    topic = Column(Text, nullable=True)
    current_round = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_room_project_name"),
    )

    project = relationship("Project", back_populates="rooms")
    messages = relationship("Message", back_populates="room", cascade="all, delete-orphan")
    receipts = relationship("ReadReceipt", back_populates="room", cascade="all, delete-orphan")
    members = relationship("RoomMember", back_populates="room", cascade="all, delete-orphan")


class RoomMember(Base):
    __tablename__ = "room_members"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("room_id", "agent_id", name="uq_room_member"),
    )

    room = relationship("Room", back_populates="members")
    agent = relationship("Agent", back_populates="memberships")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    sender = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(
        SAEnum(MessageType, values_callable=lambda e: [x.value for x in e]),
        nullable=False,
        default=MessageType.statement,
        server_default="statement",
    )
    reply_to = Column(Integer, ForeignKey("messages.id"), nullable=True)
    to = Column(String(100), nullable=True, index=True)
    edit_history = Column(Text, nullable=True)
    edited_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    room = relationship("Room", back_populates="messages")
    reactions = relationship("Reaction", back_populates="message", cascade="all, delete-orphan")


class Reaction(Base):
    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    sender = Column(String(100), nullable=False)
    emoji = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("message_id", "sender", "emoji", name="uq_reaction"),
    )

    message = relationship("Message", back_populates="reactions")


class ReadReceipt(Base):
    __tablename__ = "read_receipts"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    agent = Column(String(100), nullable=False)
    last_read = Column(Integer, nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("room_id", "agent", name="uq_receipt"),
    )

    room = relationship("Room", back_populates="receipts")

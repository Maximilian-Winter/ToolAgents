from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from agora.db.base import Base


class AgentPersona(Base):
    __tablename__ = "agent_personas"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(String(500), nullable=True)
    system_prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agents = relationship("Agent", back_populates="persona")


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    role = Column(String(200), nullable=True)
    token_hash = Column(String(64), nullable=True)
    persona_id = Column(Integer, ForeignKey("agent_personas.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    persona = relationship("AgentPersona", back_populates="agents")
    memberships = relationship("RoomMember", back_populates="agent", cascade="all, delete-orphan")
    sessions = relationship("AgentSession", back_populates="agent", cascade="all, delete-orphan")
    project_assignments = relationship("ProjectAgent", back_populates="agent", cascade="all, delete-orphan")


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    session_token = Column(String(64), unique=True, nullable=False, index=True)
    project_slug = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)

    agent = relationship("Agent", back_populates="sessions")

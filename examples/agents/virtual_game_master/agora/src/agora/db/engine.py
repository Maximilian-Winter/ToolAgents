from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agora.config import settings
from agora.db.base import Base

# Import all models so Alembic can see them via Base.metadata
from agora.db.models import (  # noqa: F401
    Agent,
    AgentPersona,
    AgentSession,
    Issue,
    IssueActivity,
    IssueComment,
    IssueDependency,
    Label,
    Message,
    Milestone,
    Project,
    Reaction,
    ReadReceipt,
    Room,
    RoomMember,
    Team,
    TeamMember,
    issue_labels,
    KBDocument,
    Mention,
)

engine = create_async_engine(settings.database_url, echo=settings.debug)

async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

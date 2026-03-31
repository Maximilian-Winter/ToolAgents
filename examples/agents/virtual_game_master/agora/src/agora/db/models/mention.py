"""Cross-reference mention model for kb: and #N links."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from agora.db.base import Base


class Mention(Base):
    __tablename__ = "mentions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    source_type = Column(String(20), nullable=False)  # "message", "issue_comment", "issue_body"
    source_id = Column(Integer, nullable=False, index=True)
    mention_type = Column(String(10), nullable=False)  # "kb" or "issue"
    target_path = Column(String(500), nullable=True)  # for kb: mentions
    target_issue_number = Column(Integer, nullable=True)  # for #N mentions
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

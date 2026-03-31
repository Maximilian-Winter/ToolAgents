from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from agora.db.base import Base
from agora.db.models.enums import IssueState, Priority


issue_labels = Table(
    "issue_labels",
    Base.metadata,
    Column("issue_id", Integer, ForeignKey("issues.id"), primary_key=True),
    Column("label_id", Integer, ForeignKey("labels.id"), primary_key=True),
)


class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    number = Column(Integer, nullable=False)
    title = Column(String(500), nullable=False)
    body = Column(Text, nullable=True)
    state = Column(
        SAEnum(IssueState, values_callable=lambda e: [x.value for x in e]),
        nullable=False,
        default=IssueState.open,
        server_default="open",
    )
    priority = Column(
        SAEnum(Priority, values_callable=lambda e: [x.value for x in e]),
        nullable=False,
        default=Priority.none,
        server_default="none",
    )
    assignee = Column(String(100), ForeignKey("agents.name"), nullable=True, index=True)
    reporter = Column(String(100), nullable=False)
    milestone_id = Column(Integer, ForeignKey("milestones.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    closed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "number", name="uq_issue_project_number"),
    )

    project = relationship("Project", back_populates="issues")
    comments = relationship("IssueComment", back_populates="issue", cascade="all, delete-orphan")
    activities = relationship("IssueActivity", back_populates="issue", cascade="all, delete-orphan")
    labels = relationship("Label", secondary=issue_labels, back_populates="issues")
    dependencies = relationship(
        "IssueDependency",
        foreign_keys="IssueDependency.issue_id",
        back_populates="issue",
        cascade="all, delete-orphan",
    )


class IssueComment(Base):
    __tablename__ = "issue_comments"

    id = Column(Integer, primary_key=True, index=True)
    issue_id = Column(Integer, ForeignKey("issues.id"), nullable=False, index=True)
    author = Column(String(100), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    issue = relationship("Issue", back_populates="comments")


class IssueActivity(Base):
    __tablename__ = "issue_activities"

    id = Column(Integer, primary_key=True, index=True)
    issue_id = Column(Integer, ForeignKey("issues.id"), nullable=False, index=True)
    actor = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    detail_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    issue = relationship("Issue", back_populates="activities")


class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(100), nullable=False)
    color = Column(String(7), nullable=True)
    description = Column(String(500), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_label_project_name"),
    )

    project = relationship("Project", back_populates="labels")
    issues = relationship("Issue", secondary=issue_labels, back_populates="labels")


class IssueDependency(Base):
    __tablename__ = "issue_dependencies"

    id = Column(Integer, primary_key=True, index=True)
    issue_id = Column(Integer, ForeignKey("issues.id"), nullable=False)
    depends_on_id = Column(Integer, ForeignKey("issues.id"), nullable=False)

    __table_args__ = (
        UniqueConstraint("issue_id", "depends_on_id", name="uq_issue_dependency"),
        CheckConstraint("issue_id != depends_on_id", name="ck_no_self_dependency"),
    )

    issue = relationship("Issue", foreign_keys=[issue_id], back_populates="dependencies")
    depends_on = relationship("Issue", foreign_keys=[depends_on_id])


class Milestone(Base):
    __tablename__ = "milestones"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    due_date = Column(DateTime, nullable=True)
    state = Column(
        SAEnum(IssueState, values_callable=lambda e: [x.value for x in e]),
        nullable=False,
        default=IssueState.open,
        server_default="open",
    )
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("project_id", "title", name="uq_milestone_project_title"),
    )

    project = relationship("Project", back_populates="milestones")

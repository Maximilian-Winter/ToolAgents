from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Boolean
from sqlalchemy.orm import relationship

from agora.db.base import Base


class CustomFieldDefinition(Base):
    """Defines a custom field that can be attached to agents or projects."""

    __tablename__ = "custom_field_definitions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # machine name, e.g. "expertise"
    label = Column(String(200), nullable=False)  # display label, e.g. "Area of Expertise"
    field_type = Column(String(20), nullable=False)  # string | number | boolean | enum
    entity_type = Column(String(20), nullable=False)  # agent | project
    options_json = Column(Text, nullable=True)  # JSON array for enum choices
    default_value = Column(String(500), nullable=True)
    required = Column(Boolean, nullable=False, default=False)
    sort_order = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("name", "entity_type", name="uq_field_name_entity_type"),
    )

    values = relationship("CustomFieldValue", back_populates="field", cascade="all, delete-orphan")


class CustomFieldValue(Base):
    """Stores the value of a custom field for a specific agent or project."""

    __tablename__ = "custom_field_values"

    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("custom_field_definitions.id"), nullable=False, index=True)
    entity_id = Column(Integer, nullable=False)  # ID of agent or project (entity_type inferred from field definition)
    value = Column(Text, nullable=False)  # stored as string, cast by field_type
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("field_id", "entity_id", name="uq_field_entity"),
    )

    field = relationship("CustomFieldDefinition", back_populates="values")

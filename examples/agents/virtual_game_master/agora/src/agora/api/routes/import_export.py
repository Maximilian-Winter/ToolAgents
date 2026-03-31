"""Import/export routes -- full project data as JSON."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agora.db.engine import get_db
from agora.db.models.agent import Agent
from agora.db.models.chat import Message, Reaction, Room
from agora.db.models.enums import IssueState, MessageType, Priority
from agora.db.models.project import Project
from agora.db.models.task import Issue, IssueComment, Label
from agora.db.models.kb_document import KBDocument
from agora.db.models.template import DocumentTemplate
from agora.db.models.custom_field import CustomFieldDefinition, CustomFieldValue
from agora.api.deps import require_project
from agora.services.kb_service import fts_insert

router = APIRouter(
    prefix="/api/projects/{slug}",
    tags=["Import/Export"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_reaction(reaction: Reaction) -> dict[str, Any]:
    return {
        "emoji": reaction.emoji,
        "sender": reaction.sender,
    }


def _serialize_message(msg: Message) -> dict[str, Any]:
    # Group reactions by emoji
    reactions_by_emoji: dict[str, list[str]] = {}
    for r in msg.reactions:
        reactions_by_emoji.setdefault(r.emoji, []).append(r.sender)

    return {
        "sender": msg.sender,
        "content": msg.content,
        "message_type": msg.message_type.value if isinstance(msg.message_type, MessageType) else str(msg.message_type),
        "reply_to": msg.reply_to,
        "to": msg.to,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
        "reactions": [
            {"emoji": emoji, "senders": senders}
            for emoji, senders in reactions_by_emoji.items()
        ],
    }


def _serialize_room(room: Room) -> dict[str, Any]:
    return {
        "name": room.name,
        "topic": room.topic,
        "current_round": room.current_round,
        "messages": [_serialize_message(m) for m in sorted(room.messages, key=lambda m: m.id)],
    }


def _serialize_comment(comment: IssueComment) -> dict[str, Any]:
    return {
        "author": comment.author,
        "body": comment.body,
        "created_at": comment.created_at.isoformat() if comment.created_at else None,
    }


def _serialize_issue(issue: Issue) -> dict[str, Any]:
    return {
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "state": issue.state.value if isinstance(issue.state, IssueState) else str(issue.state),
        "priority": issue.priority.value if isinstance(issue.priority, Priority) else str(issue.priority),
        "assignee": issue.assignee,
        "reporter": issue.reporter,
        "labels": [label.name for label in issue.labels],
        "comments": [_serialize_comment(c) for c in sorted(issue.comments, key=lambda c: c.id)],
    }


def _serialize_kb_document(doc: KBDocument) -> dict[str, Any]:
    return {
        "path": doc.path,
        "title": doc.title,
        "tags": doc.tags,
        "content": doc.content,
        "created_by": doc.created_by,
        "updated_by": doc.updated_by,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


def _serialize_template(tmpl: DocumentTemplate) -> dict[str, Any]:
    return {
        "name": tmpl.name,
        "description": tmpl.description,
        "type_tag": tmpl.type_tag,
        "content": tmpl.content,
    }


def _serialize_custom_field(field: CustomFieldDefinition) -> dict[str, Any]:
    return {
        "name": field.name,
        "label": field.label,
        "field_type": field.field_type,
        "entity_type": field.entity_type,
        "options_json": field.options_json,
        "default_value": field.default_value,
        "required": field.required,
        "sort_order": field.sort_order,
    }


# ---------------------------------------------------------------------------
# GET /api/projects/{slug}/export
# ---------------------------------------------------------------------------


@router.get("/export")
async def export_project(
    slug: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Export all project data (agents, rooms, messages, issues) as JSON."""
    project = await require_project(slug, db)

    # Load rooms with messages and reactions
    rooms_result = await db.execute(
        select(Room)
        .where(Room.project_id == project.id)
        .options(
            selectinload(Room.messages).selectinload(Message.reactions),
        )
        .order_by(Room.id)
    )
    rooms = rooms_result.scalars().all()

    # Collect all unique agent names from rooms
    agent_names: set[str] = set()
    for room in rooms:
        for msg in room.messages:
            agent_names.add(msg.sender)
            for r in msg.reactions:
                agent_names.add(r.sender)

    # Load issues with comments and labels
    issues_result = await db.execute(
        select(Issue)
        .where(Issue.project_id == project.id)
        .options(
            selectinload(Issue.comments),
            selectinload(Issue.labels),
        )
        .order_by(Issue.number)
    )
    issues = issues_result.scalars().all()

    # Collect agent names from issues too
    for issue in issues:
        if issue.assignee:
            agent_names.add(issue.assignee)
        agent_names.add(issue.reporter)
        for comment in issue.comments:
            agent_names.add(comment.author)

    # Load agents
    agents_out = []
    if agent_names:
        agents_result = await db.execute(
            select(Agent).where(Agent.name.in_(agent_names)).order_by(Agent.name)
        )
        agents = agents_result.scalars().all()
        agents_out = [
            {
                "name": a.name,
                "display_name": a.display_name,
                "role": a.role,
            }
            for a in agents
        ]

    # Load KB documents
    kb_result = await db.execute(
        select(KBDocument)
        .where(KBDocument.project_id == project.id)
        .order_by(KBDocument.path)
    )
    kb_docs = kb_result.scalars().all()

    # Load project-scoped templates
    templates_result = await db.execute(
        select(DocumentTemplate)
        .where(DocumentTemplate.project_id == project.id)
        .order_by(DocumentTemplate.name)
    )
    templates = templates_result.scalars().all()

    # Load custom field definitions and values for project agents
    fields_result = await db.execute(
        select(CustomFieldDefinition).order_by(CustomFieldDefinition.sort_order)
    )
    fields = fields_result.scalars().all()

    custom_fields_out = []
    for field in fields:
        field_data = _serialize_custom_field(field)
        # Include values for this field
        values_result = await db.execute(
            select(CustomFieldValue).where(CustomFieldValue.field_id == field.id)
        )
        values = values_result.scalars().all()
        if values:
            # For agent fields, resolve agent names; for project fields, resolve project slugs
            field_values = []
            for v in values:
                if field.entity_type == "agent":
                    agent_result = await db.execute(select(Agent).where(Agent.id == v.entity_id))
                    agent = agent_result.scalar_one_or_none()
                    if agent:
                        field_values.append({"entity_name": agent.name, "value": v.value})
                elif field.entity_type == "project":
                    proj_result = await db.execute(select(Project).where(Project.id == v.entity_id))
                    proj = proj_result.scalar_one_or_none()
                    if proj:
                        field_values.append({"entity_name": proj.slug, "value": v.value})
            if field_values:
                field_data["values"] = field_values
        custom_fields_out.append(field_data)

    return {
        "project": {
            "name": project.name,
            "slug": project.slug,
            "description": project.description,
        },
        "agents": agents_out,
        "rooms": [_serialize_room(r) for r in rooms],
        "issues": [_serialize_issue(i) for i in issues],
        "kb_documents": [_serialize_kb_document(d) for d in kb_docs],
        "templates": [_serialize_template(t) for t in templates],
        "custom_fields": custom_fields_out if custom_fields_out else [],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /api/projects/{slug}/import
# ---------------------------------------------------------------------------


@router.post("/import")
async def import_project(
    slug: str,
    data: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Import project data from JSON. Skips existing agents, rooms, and issues."""
    project = await require_project(slug, db)

    summary = {
        "agents_created": 0,
        "agents_skipped": 0,
        "rooms_created": 0,
        "rooms_skipped": 0,
        "messages_created": 0,
        "issues_created": 0,
        "issues_skipped": 0,
        "comments_created": 0,
        "kb_documents_created": 0,
        "kb_documents_skipped": 0,
        "templates_created": 0,
        "templates_skipped": 0,
        "custom_fields_created": 0,
        "custom_fields_skipped": 0,
        "custom_field_values_set": 0,
    }

    # --- Import agents ---
    for agent_data in data.get("agents", []):
        name = agent_data.get("name")
        if not name:
            continue
        existing = await db.execute(select(Agent).where(Agent.name == name))
        if existing.scalar_one_or_none():
            summary["agents_skipped"] += 1
            continue
        agent = Agent(
            name=name,
            display_name=agent_data.get("display_name"),
            role=agent_data.get("role"),
        )
        db.add(agent)
        summary["agents_created"] += 1

    await db.flush()

    # --- Import rooms and messages ---
    for room_data in data.get("rooms", []):
        room_name = room_data.get("name")
        if not room_name:
            continue

        existing = await db.execute(
            select(Room).where(and_(Room.name == room_name, Room.project_id == project.id))
        )
        if existing.scalar_one_or_none():
            summary["rooms_skipped"] += 1
            continue

        room = Room(
            name=room_name,
            topic=room_data.get("topic"),
            current_round=room_data.get("current_round", 1),
            project_id=project.id,
        )
        db.add(room)
        await db.flush()

        summary["rooms_created"] += 1

        # Import messages for this room
        for msg_data in room_data.get("messages", []):
            sender = msg_data.get("sender")
            content = msg_data.get("content")
            if not sender or not content:
                continue

            # Parse message_type
            msg_type_str = msg_data.get("message_type", "statement")
            try:
                msg_type = MessageType(msg_type_str)
            except ValueError:
                msg_type = MessageType.statement

            # Parse created_at
            created_at = None
            if msg_data.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(msg_data["created_at"])
                except (ValueError, TypeError):
                    created_at = None

            msg = Message(
                room_id=room.id,
                sender=sender,
                content=content,
                message_type=msg_type,
                reply_to=msg_data.get("reply_to"),
                to=msg_data.get("to"),
                created_at=created_at or datetime.now(timezone.utc),
            )
            db.add(msg)
            await db.flush()
            summary["messages_created"] += 1

            # Import reactions for this message
            for reaction_data in msg_data.get("reactions", []):
                emoji = reaction_data.get("emoji")
                senders = reaction_data.get("senders", [])
                if not emoji:
                    continue
                for reaction_sender in senders:
                    reaction = Reaction(
                        message_id=msg.id,
                        sender=reaction_sender,
                        emoji=emoji,
                    )
                    db.add(reaction)

    # --- Import issues and comments ---
    for issue_data in data.get("issues", []):
        number = issue_data.get("number")
        title = issue_data.get("title")
        if number is None or not title:
            continue

        existing = await db.execute(
            select(Issue).where(
                and_(Issue.number == number, Issue.project_id == project.id)
            )
        )
        if existing.scalar_one_or_none():
            summary["issues_skipped"] += 1
            continue

        # Parse state and priority
        try:
            state = IssueState(issue_data.get("state", "open"))
        except ValueError:
            state = IssueState.open

        try:
            priority = Priority(issue_data.get("priority", "none"))
        except ValueError:
            priority = Priority.none

        issue = Issue(
            project_id=project.id,
            number=number,
            title=title,
            body=issue_data.get("body"),
            state=state,
            priority=priority,
            assignee=issue_data.get("assignee"),
            reporter=issue_data.get("reporter", "unknown"),
        )
        db.add(issue)
        await db.flush()

        summary["issues_created"] += 1

        # Handle labels
        for label_name in issue_data.get("labels", []):
            label_result = await db.execute(
                select(Label).where(
                    and_(Label.name == label_name, Label.project_id == project.id)
                )
            )
            label = label_result.scalar_one_or_none()
            if not label:
                label = Label(name=label_name, project_id=project.id)
                db.add(label)
                await db.flush()
            issue.labels.append(label)

        # Import comments
        for comment_data in issue_data.get("comments", []):
            author = comment_data.get("author")
            body = comment_data.get("body")
            if not author or not body:
                continue

            created_at = None
            if comment_data.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(comment_data["created_at"])
                except (ValueError, TypeError):
                    created_at = None

            comment = IssueComment(
                issue_id=issue.id,
                author=author,
                body=body,
                created_at=created_at or datetime.now(timezone.utc),
            )
            db.add(comment)
            summary["comments_created"] += 1

    # --- Import KB documents ---
    for kb_data in data.get("kb_documents", []):
        path = kb_data.get("path")
        if not path:
            continue

        existing = await db.execute(
            select(KBDocument).where(
                and_(KBDocument.path == path, KBDocument.project_id == project.id)
            )
        )
        if existing.scalar_one_or_none():
            summary["kb_documents_skipped"] += 1
            continue

        created_at = None
        if kb_data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(kb_data["created_at"])
            except (ValueError, TypeError):
                created_at = None

        updated_at = None
        if kb_data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(kb_data["updated_at"])
            except (ValueError, TypeError):
                updated_at = None

        doc = KBDocument(
            project_id=project.id,
            path=path,
            title=kb_data.get("title", path),
            tags=kb_data.get("tags"),
            content=kb_data.get("content", ""),
            created_by=kb_data.get("created_by", "import"),
            updated_by=kb_data.get("updated_by", "import"),
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=updated_at or datetime.now(timezone.utc),
        )
        db.add(doc)
        await db.flush()
        summary["kb_documents_created"] += 1

        # Sync FTS index
        await fts_insert(db, doc.id, doc.title, doc.content, doc.tags or "")

    # --- Import project-scoped templates ---
    for tmpl_data in data.get("templates", []):
        name = tmpl_data.get("name")
        if not name:
            continue

        existing = await db.execute(
            select(DocumentTemplate).where(
                and_(DocumentTemplate.name == name, DocumentTemplate.project_id == project.id)
            )
        )
        if existing.scalar_one_or_none():
            summary["templates_skipped"] += 1
            continue

        tmpl = DocumentTemplate(
            name=name,
            description=tmpl_data.get("description"),
            type_tag=tmpl_data.get("type_tag"),
            content=tmpl_data.get("content", ""),
            project_id=project.id,
        )
        db.add(tmpl)
        summary["templates_created"] += 1

    await db.flush()

    # --- Import custom fields ---
    for field_data in data.get("custom_fields", []):
        field_name = field_data.get("name")
        entity_type = field_data.get("entity_type")
        if not field_name or not entity_type:
            continue

        existing = await db.execute(
            select(CustomFieldDefinition).where(
                and_(
                    CustomFieldDefinition.name == field_name,
                    CustomFieldDefinition.entity_type == entity_type,
                )
            )
        )
        field = existing.scalar_one_or_none()
        if field:
            summary["custom_fields_skipped"] += 1
        else:
            field = CustomFieldDefinition(
                name=field_name,
                label=field_data.get("label", field_name),
                field_type=field_data.get("field_type", "string"),
                entity_type=entity_type,
                options_json=field_data.get("options_json"),
                default_value=field_data.get("default_value"),
                required=field_data.get("required", False),
                sort_order=field_data.get("sort_order", 0),
            )
            db.add(field)
            await db.flush()
            summary["custom_fields_created"] += 1

        # Import field values
        for val_data in field_data.get("values", []):
            entity_name = val_data.get("entity_name")
            value = val_data.get("value")
            if not entity_name or value is None:
                continue

            # Resolve entity ID
            entity_id = None
            if entity_type == "agent":
                agent_result = await db.execute(select(Agent).where(Agent.name == entity_name))
                agent = agent_result.scalar_one_or_none()
                if agent:
                    entity_id = agent.id
            elif entity_type == "project":
                proj_result = await db.execute(select(Project).where(Project.slug == entity_name))
                proj = proj_result.scalar_one_or_none()
                if proj:
                    entity_id = proj.id

            if entity_id is None:
                continue

            # Upsert field value
            existing_val = await db.execute(
                select(CustomFieldValue).where(
                    and_(
                        CustomFieldValue.field_id == field.id,
                        CustomFieldValue.entity_id == entity_id,
                    )
                )
            )
            if existing_val.scalar_one_or_none():
                continue  # skip existing values

            fv = CustomFieldValue(
                field_id=field.id,
                entity_id=entity_id,
                value=str(value),
            )
            db.add(fv)
            summary["custom_field_values_set"] += 1

    await db.commit()

    return {
        "status": "ok",
        "summary": summary,
    }

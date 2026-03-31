import json
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.models.task import Issue, IssueActivity, Label, issue_labels
from agora.db.models.enums import IssueState
from agora.schemas.task import IssueCreate, IssueUpdate

MAX_ISSUE_NUMBER_RETRIES = 3


async def get_next_issue_number(project_id: int, db: AsyncSession) -> int:
    """Get the next sequential issue number for a project.
    SELECT COALESCE(MAX(number), 0) + 1 FROM issues WHERE project_id = ?
    """
    result = await db.execute(
        select(func.coalesce(func.max(Issue.number), 0) + 1).where(
            Issue.project_id == project_id
        )
    )
    return result.scalar_one()


async def create_issue(
    project_id: int, data: IssueCreate, db: AsyncSession
) -> Issue:
    """Create issue with auto-number, attach labels if provided, create activity log entry.

    Retries on IntegrityError (duplicate issue number) to handle concurrent
    creation requests safely.
    """
    last_error = None
    for attempt in range(MAX_ISSUE_NUMBER_RETRIES):
        try:
            number = await get_next_issue_number(project_id, db)

            issue = Issue(
                project_id=project_id,
                number=number,
                title=data.title,
                body=data.body,
                priority=data.priority,
                assignee=data.assignee,
                reporter=data.reporter,
                milestone_id=data.milestone_id,
            )
            db.add(issue)
            await db.flush()  # get issue.id — raises IntegrityError on dup

            # Attach labels if provided
            if data.labels:
                for label_name in data.labels:
                    result = await db.execute(
                        select(Label).where(
                            Label.project_id == project_id, Label.name == label_name
                        )
                    )
                    label = result.scalar_one_or_none()
                    if label:
                        await db.execute(
                            issue_labels.insert().values(
                                issue_id=issue.id, label_id=label.id
                            )
                        )

            # Log activity
            await log_activity(issue.id, data.reporter, "created", None, db)

            return issue

        except IntegrityError as e:
            last_error = e
            await db.rollback()
            continue

    raise last_error


async def update_issue(
    issue: Issue, data: IssueUpdate, actor: str, db: AsyncSession
) -> Issue:
    """Update issue fields. For each changed field, create an IssueActivity record.
    If state changes to 'closed', set closed_at. If state changes to 'open', clear closed_at.
    """
    update_data = data.model_dump(exclude_unset=True)

    for field, new_value in update_data.items():
        old_value = getattr(issue, field)
        # Normalize enum values for comparison
        if hasattr(old_value, "value"):
            old_comparable = old_value.value
        else:
            old_comparable = old_value
        if hasattr(new_value, "value"):
            new_comparable = new_value.value
        else:
            new_comparable = new_value

        if old_comparable != new_comparable:
            setattr(issue, field, new_value)
            await log_activity(
                issue.id,
                actor,
                f"changed_{field}",
                {"old": str(old_comparable), "new": str(new_comparable)},
                db,
            )

    # Handle closed_at based on state transitions
    if "state" in update_data:
        if data.state == IssueState.closed and issue.closed_at is None:
            issue.closed_at = datetime.now(timezone.utc)
        elif data.state == IssueState.open:
            issue.closed_at = None

    return issue


async def log_activity(
    issue_id: int,
    actor: str,
    action: str,
    detail: dict | None,
    db: AsyncSession,
) -> None:
    """Create an IssueActivity record."""
    activity = IssueActivity(
        issue_id=issue_id,
        actor=actor,
        action=action,
        detail_json=json.dumps(detail) if detail else None,
    )
    db.add(activity)

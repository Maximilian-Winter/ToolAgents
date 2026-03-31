from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agora.db.engine import get_db
from agora.db.models.enums import IssueState, Priority
from agora.db.models.task import (
    Issue,
    IssueComment,
    IssueActivity,
    IssueDependency,
    Label,
    Milestone,
    issue_labels,
)
from agora.api.deps import require_project
from agora.schemas.task import (
    IssueCreate,
    IssueUpdate,
    IssueOut,
    CommentCreate,
    CommentUpdate,
    CommentOut,
    ActivityOut,
    LabelCreate,
    LabelUpdate,
    LabelOut,
    LabelAttach,
    MilestoneCreate,
    MilestoneUpdate,
    MilestoneOut,
    DependencyCreate,
    DependencyOut,
)
from agora.services.mention_service import store_mentions
from agora.services.task_service import (
    create_issue as svc_create_issue,
    update_issue as svc_update_issue,
    log_activity,
)

router = APIRouter(prefix="/api/projects/{project_slug}", tags=["Issues"])


# ── Helpers ──────────────────────────────────────────────────────────────────


async def require_issue(project_id: int, number: int, db: AsyncSession) -> Issue:
    result = await db.execute(
        select(Issue).where(Issue.project_id == project_id, Issue.number == number)
    )
    issue = result.scalar_one_or_none()
    if not issue:
        raise HTTPException(404, f"Issue #{number} not found")
    return issue


async def _issue_to_out(issue: Issue, db: AsyncSession) -> IssueOut:
    """Build an IssueOut for a single issue (used by get/create/update endpoints)."""
    label_result = await db.execute(
        select(Label)
        .join(issue_labels, Label.id == issue_labels.c.label_id)
        .where(issue_labels.c.issue_id == issue.id)
    )
    labels = label_result.scalars().all()

    count_result = await db.execute(
        select(func.count(IssueComment.id)).where(IssueComment.issue_id == issue.id)
    )
    comment_count = count_result.scalar_one()

    return IssueOut(
        id=issue.id,
        project_id=issue.project_id,
        number=issue.number,
        title=issue.title,
        body=issue.body,
        state=issue.state,
        priority=issue.priority,
        assignee=issue.assignee,
        reporter=issue.reporter,
        milestone_id=issue.milestone_id,
        created_at=issue.created_at,
        updated_at=issue.updated_at,
        closed_at=issue.closed_at,
        labels=[LabelOut.model_validate(lb) for lb in labels],
        comment_count=comment_count,
    )


async def _issues_to_out_batch(issues: list[Issue], db: AsyncSession) -> list[IssueOut]:
    """Batch-build IssueOut for multiple issues — avoids N+1 queries."""
    if not issues:
        return []

    issue_ids = [i.id for i in issues]

    # Batch load all labels for these issues
    label_rows = await db.execute(
        select(issue_labels.c.issue_id, Label)
        .join(Label, Label.id == issue_labels.c.label_id)
        .where(issue_labels.c.issue_id.in_(issue_ids))
    )
    labels_by_issue: dict[int, list[Label]] = {}
    for issue_id, label in label_rows.all():
        labels_by_issue.setdefault(issue_id, []).append(label)

    # Batch count comments
    comment_counts_result = await db.execute(
        select(IssueComment.issue_id, func.count(IssueComment.id))
        .where(IssueComment.issue_id.in_(issue_ids))
        .group_by(IssueComment.issue_id)
    )
    comment_counts: dict[int, int] = dict(comment_counts_result.all())

    return [
        IssueOut(
            id=issue.id,
            project_id=issue.project_id,
            number=issue.number,
            title=issue.title,
            body=issue.body,
            state=issue.state,
            priority=issue.priority,
            assignee=issue.assignee,
            reporter=issue.reporter,
            milestone_id=issue.milestone_id,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            closed_at=issue.closed_at,
            labels=[LabelOut.model_validate(lb) for lb in labels_by_issue.get(issue.id, [])],
            comment_count=comment_counts.get(issue.id, 0),
        )
        for issue in issues
    ]


# ── Issues ───────────────────────────────────────────────────────────────────


@router.post("/issues", response_model=IssueOut, status_code=201)
async def create_issue(
    project_slug: str,
    body: IssueCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await svc_create_issue(project.id, body, db)
    await db.commit()
    await db.refresh(issue)
    if issue.body:
        await store_mentions(project.id, "issue_body", issue.id, issue.body, db)
        await db.commit()
    return await _issue_to_out(issue, db)


@router.get("/issues", response_model=list[IssueOut])
async def list_issues(
    project_slug: str,
    state: Optional[IssueState] = None,
    assignee: Optional[str] = None,
    label: Optional[str] = None,
    milestone_id: Optional[int] = None,
    priority: Optional[Priority] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)

    stmt = select(Issue).where(Issue.project_id == project.id)

    if state is not None:
        stmt = stmt.where(Issue.state == state)
    if assignee is not None:
        stmt = stmt.where(Issue.assignee == assignee)
    if milestone_id is not None:
        stmt = stmt.where(Issue.milestone_id == milestone_id)
    if priority is not None:
        stmt = stmt.where(Issue.priority == priority)
    if label is not None:
        stmt = stmt.join(issue_labels, Issue.id == issue_labels.c.issue_id).join(
            Label, Label.id == issue_labels.c.label_id
        ).where(Label.name == label)

    stmt = stmt.order_by(Issue.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    issues = result.scalars().all()

    return await _issues_to_out_batch(issues, db)


@router.get("/issues/search", response_model=list[IssueOut])
async def search_issues(
    project_slug: str,
    q: str = Query(..., min_length=1, description="Search query (case-insensitive match on title and body)"),
    state: Optional[IssueState] = None,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Search issues by title or body text."""
    project = await require_project(project_slug, db)

    from sqlalchemy import or_
    stmt = (
        select(Issue)
        .where(Issue.project_id == project.id)
        .where(
            or_(
                Issue.title.ilike(f"%{q}%"),
                Issue.body.ilike(f"%{q}%"),
            )
        )
    )
    if state is not None:
        stmt = stmt.where(Issue.state == state)
    stmt = stmt.order_by(Issue.created_at.desc()).limit(limit)

    result = await db.execute(stmt)
    issues = result.scalars().all()
    return await _issues_to_out_batch(issues, db)


@router.get("/issues/{number}", response_model=IssueOut)
async def get_issue(
    project_slug: str,
    number: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    return await _issue_to_out(issue, db)


@router.patch("/issues/{number}", response_model=IssueOut)
async def update_issue(
    project_slug: str,
    number: int,
    body: IssueUpdate,
    actor: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    issue = await svc_update_issue(issue, body, actor, db)
    await db.commit()
    await db.refresh(issue)
    if body.body is not None:
        await store_mentions(project.id, "issue_body", issue.id, body.body, db)
        await db.commit()
    return await _issue_to_out(issue, db)


@router.delete("/issues/{number}", status_code=204)
async def delete_issue(
    project_slug: str,
    number: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    await db.delete(issue)
    await db.commit()


# ── Comments ─────────────────────────────────────────────────────────────────


@router.post("/issues/{number}/comments", response_model=CommentOut, status_code=201)
async def add_comment(
    project_slug: str,
    number: int,
    body: CommentCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    comment = IssueComment(
        issue_id=issue.id,
        author=body.author,
        body=body.body,
    )
    db.add(comment)
    await log_activity(issue.id, body.author, "commented", None, db)
    await db.commit()
    await db.refresh(comment)
    await store_mentions(project.id, "issue_comment", comment.id, body.body, db)
    await db.commit()
    return comment


@router.get("/issues/{number}/comments", response_model=list[CommentOut])
async def list_comments(
    project_slug: str,
    number: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    result = await db.execute(
        select(IssueComment)
        .where(IssueComment.issue_id == issue.id)
        .order_by(IssueComment.created_at.asc())
    )
    return result.scalars().all()


@router.patch(
    "/issues/{number}/comments/{comment_id}", response_model=CommentOut
)
async def edit_comment(
    project_slug: str,
    number: int,
    comment_id: int,
    body: CommentUpdate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    result = await db.execute(
        select(IssueComment).where(
            IssueComment.id == comment_id, IssueComment.issue_id == issue.id
        )
    )
    comment = result.scalar_one_or_none()
    if not comment:
        raise HTTPException(404, f"Comment {comment_id} not found on issue #{number}")
    comment.body = body.body
    await db.commit()
    await db.refresh(comment)
    return comment


@router.delete("/issues/{number}/comments/{comment_id}", status_code=204)
async def delete_comment(
    project_slug: str,
    number: int,
    comment_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    result = await db.execute(
        select(IssueComment).where(
            IssueComment.id == comment_id, IssueComment.issue_id == issue.id
        )
    )
    comment = result.scalar_one_or_none()
    if not comment:
        raise HTTPException(404, f"Comment {comment_id} not found on issue #{number}")
    await db.delete(comment)
    await db.commit()


# ── Activity ─────────────────────────────────────────────────────────────────


@router.get("/issues/{number}/activity", response_model=list[ActivityOut])
async def list_activity(
    project_slug: str,
    number: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    result = await db.execute(
        select(IssueActivity)
        .where(IssueActivity.issue_id == issue.id)
        .order_by(IssueActivity.created_at.desc())
    )
    return result.scalars().all()


# ── Labels (project-scoped) ─────────────────────────────────────────────────


@router.post("/labels", response_model=LabelOut, status_code=201)
async def create_label(
    project_slug: str,
    body: LabelCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    # Check uniqueness
    existing = await db.execute(
        select(Label).where(Label.project_id == project.id, Label.name == body.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"Label '{body.name}' already exists in this project")
    label = Label(
        project_id=project.id,
        name=body.name,
        color=body.color,
        description=body.description,
    )
    db.add(label)
    await db.commit()
    await db.refresh(label)
    return label


@router.get("/labels", response_model=list[LabelOut])
async def list_labels(
    project_slug: str,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Label).where(Label.project_id == project.id)
    )
    return result.scalars().all()


@router.patch("/labels/{label_id}", response_model=LabelOut)
async def update_label(
    project_slug: str,
    label_id: int,
    body: LabelUpdate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Label).where(Label.id == label_id, Label.project_id == project.id)
    )
    label = result.scalar_one_or_none()
    if not label:
        raise HTTPException(404, f"Label {label_id} not found")
    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(label, key, value)
    await db.commit()
    await db.refresh(label)
    return label


@router.delete("/labels/{label_id}", status_code=204)
async def delete_label(
    project_slug: str,
    label_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Label).where(Label.id == label_id, Label.project_id == project.id)
    )
    label = result.scalar_one_or_none()
    if not label:
        raise HTTPException(404, f"Label {label_id} not found")
    await db.delete(label)
    await db.commit()


# ── Issue-Label attachment ───────────────────────────────────────────────────


@router.post("/issues/{number}/labels", response_model=LabelOut, status_code=201)
async def attach_label(
    project_slug: str,
    number: int,
    body: LabelAttach,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    # Resolve label by name
    result = await db.execute(
        select(Label).where(
            Label.project_id == project.id, Label.name == body.label_name
        )
    )
    label = result.scalar_one_or_none()
    if not label:
        raise HTTPException(404, f"Label '{body.label_name}' not found in this project")
    # Check if already attached
    existing = await db.execute(
        select(issue_labels).where(
            issue_labels.c.issue_id == issue.id,
            issue_labels.c.label_id == label.id,
        )
    )
    if existing.first():
        raise HTTPException(409, f"Label '{body.label_name}' already attached to issue #{number}")
    await db.execute(
        issue_labels.insert().values(issue_id=issue.id, label_id=label.id)
    )
    await db.commit()
    return label


@router.delete("/issues/{number}/labels/{label_id}", status_code=204)
async def detach_label(
    project_slug: str,
    number: int,
    label_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    result = await db.execute(
        delete(issue_labels).where(
            issue_labels.c.issue_id == issue.id,
            issue_labels.c.label_id == label_id,
        )
    )
    if result.rowcount == 0:
        raise HTTPException(404, f"Label {label_id} not attached to issue #{number}")
    await db.commit()


# ── Dependencies ─────────────────────────────────────────────────────────────


@router.post(
    "/issues/{number}/dependencies", response_model=DependencyOut, status_code=201
)
async def add_dependency(
    project_slug: str,
    number: int,
    body: DependencyCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)

    # Prevent self-dependency
    if body.depends_on_number == number:
        raise HTTPException(400, "An issue cannot depend on itself")

    # Resolve depends_on issue
    depends_on = await require_issue(project.id, body.depends_on_number, db)

    # Check for circular dependency via full graph walk:
    # Starting from depends_on, follow all dependency edges. If we reach
    # the original issue, adding this edge would create a cycle.
    async def _would_create_cycle(start_id: int, target_id: int) -> bool:
        visited: set[int] = set()
        stack = [start_id]
        while stack:
            current = stack.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            result = await db.execute(
                select(IssueDependency.depends_on_id).where(
                    IssueDependency.issue_id == current
                )
            )
            for (dep_id,) in result.all():
                if dep_id not in visited:
                    stack.append(dep_id)
        return False

    if await _would_create_cycle(depends_on.id, issue.id):
        raise HTTPException(
            400,
            f"Circular dependency: issue #{body.depends_on_number} already "
            f"depends (directly or transitively) on #{number}",
        )

    # Check duplicate
    existing = await db.execute(
        select(IssueDependency).where(
            IssueDependency.issue_id == issue.id,
            IssueDependency.depends_on_id == depends_on.id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            409, f"Issue #{number} already depends on #{body.depends_on_number}"
        )

    dep = IssueDependency(issue_id=issue.id, depends_on_id=depends_on.id)
    db.add(dep)
    await db.commit()
    return DependencyOut(
        issue_number=number, depends_on_number=body.depends_on_number
    )


@router.delete(
    "/issues/{number}/dependencies/{depends_on_number}", status_code=204
)
async def remove_dependency(
    project_slug: str,
    number: int,
    depends_on_number: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    issue = await require_issue(project.id, number, db)
    depends_on = await require_issue(project.id, depends_on_number, db)

    result = await db.execute(
        select(IssueDependency).where(
            IssueDependency.issue_id == issue.id,
            IssueDependency.depends_on_id == depends_on.id,
        )
    )
    dep = result.scalar_one_or_none()
    if not dep:
        raise HTTPException(
            404, f"Dependency from #{number} on #{depends_on_number} not found"
        )
    await db.delete(dep)
    await db.commit()


# ── Milestones (project-scoped) ─────────────────────────────────────────────


@router.post("/milestones", response_model=MilestoneOut, status_code=201)
async def create_milestone(
    project_slug: str,
    body: MilestoneCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    milestone = Milestone(
        project_id=project.id,
        title=body.title,
        description=body.description,
        due_date=body.due_date,
    )
    db.add(milestone)
    await db.commit()
    await db.refresh(milestone)
    return MilestoneOut(
        id=milestone.id,
        project_id=milestone.project_id,
        title=milestone.title,
        description=milestone.description,
        due_date=milestone.due_date,
        state=milestone.state,
        created_at=milestone.created_at,
        open_issues=0,
        closed_issues=0,
    )


@router.get("/milestones", response_model=list[MilestoneOut])
async def list_milestones(
    project_slug: str,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Milestone).where(Milestone.project_id == project.id)
    )
    milestones = result.scalars().all()

    out = []
    for ms in milestones:
        # Count open and closed issues for this milestone
        open_count_result = await db.execute(
            select(func.count(Issue.id)).where(
                Issue.milestone_id == ms.id, Issue.state == IssueState.open
            )
        )
        closed_count_result = await db.execute(
            select(func.count(Issue.id)).where(
                Issue.milestone_id == ms.id, Issue.state == IssueState.closed
            )
        )
        out.append(
            MilestoneOut(
                id=ms.id,
                project_id=ms.project_id,
                title=ms.title,
                description=ms.description,
                due_date=ms.due_date,
                state=ms.state,
                created_at=ms.created_at,
                open_issues=open_count_result.scalar_one(),
                closed_issues=closed_count_result.scalar_one(),
            )
        )
    return out


@router.patch("/milestones/{milestone_id}", response_model=MilestoneOut)
async def update_milestone(
    project_slug: str,
    milestone_id: int,
    body: MilestoneUpdate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Milestone).where(
            Milestone.id == milestone_id, Milestone.project_id == project.id
        )
    )
    milestone = result.scalar_one_or_none()
    if not milestone:
        raise HTTPException(404, f"Milestone {milestone_id} not found")
    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(milestone, key, value)
    await db.commit()
    await db.refresh(milestone)

    # Count issues
    open_count_result = await db.execute(
        select(func.count(Issue.id)).where(
            Issue.milestone_id == milestone.id, Issue.state == IssueState.open
        )
    )
    closed_count_result = await db.execute(
        select(func.count(Issue.id)).where(
            Issue.milestone_id == milestone.id, Issue.state == IssueState.closed
        )
    )
    return MilestoneOut(
        id=milestone.id,
        project_id=milestone.project_id,
        title=milestone.title,
        description=milestone.description,
        due_date=milestone.due_date,
        state=milestone.state,
        created_at=milestone.created_at,
        open_issues=open_count_result.scalar_one(),
        closed_issues=closed_count_result.scalar_one(),
    )


@router.delete("/milestones/{milestone_id}", status_code=204)
async def delete_milestone(
    project_slug: str,
    milestone_id: int,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(Milestone).where(
            Milestone.id == milestone_id, Milestone.project_id == project.id
        )
    )
    milestone = result.scalar_one_or_none()
    if not milestone:
        raise HTTPException(404, f"Milestone {milestone_id} not found")
    await db.delete(milestone)
    await db.commit()

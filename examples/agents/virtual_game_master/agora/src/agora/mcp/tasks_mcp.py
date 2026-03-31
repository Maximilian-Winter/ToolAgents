"""
Agora Tasks MCP Server
Provides Claude Code agents with tools for issue/task management
scoped to projects via the Agora FastAPI service.

Usage (stdio, for Claude Code):
    python -m agora.mcp.tasks_mcp

Requires the Agora FastAPI service to be running.
Configure the service URL via AGORA_URL environment variable
(default: http://127.0.0.1:8321).
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from agora.mcp.client import api_request as _request, format_result as _format_result

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("agora_tasks")


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class CreateIssueInput(BaseModel):
    """Input for creating a new issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:  str = Field(..., description="Project slug", min_length=1)
    title:    str = Field(..., description="Issue title", min_length=1, max_length=500)
    body:     Optional[str] = Field(None, description="Issue body / description")
    reporter: str = Field(..., description="Agent or user name creating this issue", min_length=1, max_length=100)
    priority: Optional[str] = Field(None, description="Priority: none, low, medium, high, critical (default: none)")
    assignee: Optional[str] = Field(None, description="Agent name to assign this issue to", max_length=100)
    labels:   Optional[list[str]] = Field(None, description="List of label names to attach")


class ListIssuesInput(BaseModel):
    """Input for listing issues in a project."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:      str = Field(..., description="Project slug", min_length=1)
    state:        Optional[str] = Field(None, description="Filter by state: 'open' or 'closed'")
    assignee:     Optional[str] = Field(None, description="Filter by assignee agent name")
    label:        Optional[str] = Field(None, description="Filter by label name")
    milestone_id: Optional[int] = Field(None, description="Filter by milestone ID")
    priority:     Optional[str] = Field(None, description="Filter by priority: none, low, medium, high, critical")
    limit:        int = Field(default=50, description="Max issues to return", ge=1, le=200)
    offset:       int = Field(default=0, description="Offset for pagination", ge=0)


class GetIssueInput(BaseModel):
    """Input for getting a single issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    number:  int = Field(..., description="Issue number within the project")


class UpdateIssueInput(BaseModel):
    """Input for updating an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:      str = Field(..., description="Project slug", min_length=1)
    number:       int = Field(..., description="Issue number within the project")
    actor:        str = Field(..., description="Agent name making this change", min_length=1, max_length=100)
    title:        Optional[str] = Field(None, description="New title", min_length=1, max_length=500)
    body:         Optional[str] = Field(None, description="New body / description")
    state:        Optional[str] = Field(None, description="New state: 'open' or 'closed'")
    priority:     Optional[str] = Field(None, description="New priority: none, low, medium, high, critical")
    assignee:     Optional[str] = Field(None, description="New assignee agent name", max_length=100)
    milestone_id: Optional[int] = Field(None, description="New milestone ID")


class CloseIssueInput(BaseModel):
    """Input for closing an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    number:  int = Field(..., description="Issue number within the project")
    actor:   str = Field(..., description="Agent name closing this issue", min_length=1, max_length=100)


class ReopenIssueInput(BaseModel):
    """Input for reopening an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    number:  int = Field(..., description="Issue number within the project")
    actor:   str = Field(..., description="Agent name reopening this issue", min_length=1, max_length=100)


class AddCommentInput(BaseModel):
    """Input for adding a comment to an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    number:  int = Field(..., description="Issue number within the project")
    author:  str = Field(..., description="Agent name writing the comment", min_length=1, max_length=100)
    body:    str = Field(..., description="Comment body text", min_length=1)


class ListCommentsInput(BaseModel):
    """Input for listing comments on an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    number:  int = Field(..., description="Issue number within the project")


class AddLabelInput(BaseModel):
    """Input for attaching a label to an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:    str = Field(..., description="Project slug", min_length=1)
    number:     int = Field(..., description="Issue number within the project")
    label_name: str = Field(..., description="Name of the label to attach", min_length=1, max_length=100)


class RemoveLabelInput(BaseModel):
    """Input for removing a label from an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:  str = Field(..., description="Project slug", min_length=1)
    number:   int = Field(..., description="Issue number within the project")
    label_id: int = Field(..., description="ID of the label to remove")


class SetMilestoneInput(BaseModel):
    """Input for setting an issue's milestone."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:      str = Field(..., description="Project slug", min_length=1)
    number:       int = Field(..., description="Issue number within the project")
    milestone_id: int = Field(..., description="Milestone ID to set")
    actor:        str = Field(..., description="Agent name making this change", min_length=1, max_length=100)


class AddDependencyInput(BaseModel):
    """Input for adding a dependency between issues."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project:           str = Field(..., description="Project slug", min_length=1)
    number:            int = Field(..., description="Issue number that depends on another")
    depends_on_number: int = Field(..., description="Issue number that this issue depends on")


class ListMilestonesInput(BaseModel):
    """Input for listing milestones in a project."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)


class GetActivityInput(BaseModel):
    """Input for getting activity log of an issue."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug", min_length=1)
    number:  int = Field(..., description="Issue number within the project")


# ---------------------------------------------------------------------------
# Tools — Issues
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tasks_create_issue",
    annotations={
        "title": "Create Issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tasks_create_issue(params: CreateIssueInput) -> str:
    """Create a new issue (task) in a project.

    Args:
        params: Issue details (project, title, reporter, optional body/priority/assignee/labels).

    Returns:
        str: JSON with the created issue including its number.
    """
    body: dict = {
        "title": params.title,
        "reporter": params.reporter,
    }
    if params.body is not None:
        body["body"] = params.body
    if params.priority is not None:
        body["priority"] = params.priority
    if params.assignee is not None:
        body["assignee"] = params.assignee
    if params.labels is not None:
        body["labels"] = params.labels
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/issues",
        body=body,
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_list_issues",
    annotations={
        "title": "List Issues",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_list_issues(params: ListIssuesInput) -> str:
    """List issues in a project with optional filters.

    Args:
        params: Project slug and optional filters (state, assignee, label, milestone_id, priority, limit, offset).

    Returns:
        str: JSON array of issues matching the filters.
    """
    query: dict = {}
    if params.state is not None:
        query["state"] = params.state
    if params.assignee is not None:
        query["assignee"] = params.assignee
    if params.label is not None:
        query["label"] = params.label
    if params.milestone_id is not None:
        query["milestone_id"] = params.milestone_id
    if params.priority is not None:
        query["priority"] = params.priority
    query["limit"] = params.limit
    query["offset"] = params.offset
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/issues",
        params=query,
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_get_issue",
    annotations={
        "title": "Get Issue",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_get_issue(params: GetIssueInput) -> str:
    """Get details of a single issue by its number.

    Args:
        params: Project slug and issue number.

    Returns:
        str: JSON with full issue details including labels and comment count.
    """
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/issues/{params.number}",
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_update_issue",
    annotations={
        "title": "Update Issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_update_issue(params: UpdateIssueInput) -> str:
    """Update fields on an existing issue. Only provided fields are changed.

    Args:
        params: Project slug, issue number, actor, and fields to update.

    Returns:
        str: JSON with the updated issue.
    """
    body: dict = {}
    if params.title is not None:
        body["title"] = params.title
    if params.body is not None:
        body["body"] = params.body
    if params.state is not None:
        body["state"] = params.state
    if params.priority is not None:
        body["priority"] = params.priority
    if params.assignee is not None:
        body["assignee"] = params.assignee
    if params.milestone_id is not None:
        body["milestone_id"] = params.milestone_id
    result = await _request(
        "PATCH",
        f"/api/projects/{params.project}/issues/{params.number}",
        params={"actor": params.actor},
        body=body,
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_close_issue",
    annotations={
        "title": "Close Issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_close_issue(params: CloseIssueInput) -> str:
    """Close an issue.

    Args:
        params: Project slug, issue number, and actor name.

    Returns:
        str: JSON with the updated (closed) issue.
    """
    result = await _request(
        "PATCH",
        f"/api/projects/{params.project}/issues/{params.number}",
        params={"actor": params.actor},
        body={"state": "closed"},
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_reopen_issue",
    annotations={
        "title": "Reopen Issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_reopen_issue(params: ReopenIssueInput) -> str:
    """Reopen a previously closed issue.

    Args:
        params: Project slug, issue number, and actor name.

    Returns:
        str: JSON with the updated (reopened) issue.
    """
    result = await _request(
        "PATCH",
        f"/api/projects/{params.project}/issues/{params.number}",
        params={"actor": params.actor},
        body={"state": "open"},
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Comments
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tasks_add_comment",
    annotations={
        "title": "Add Comment",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tasks_add_comment(params: AddCommentInput) -> str:
    """Add a comment to an issue.

    Args:
        params: Project slug, issue number, author, and comment body.

    Returns:
        str: JSON with the created comment.
    """
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/issues/{params.number}/comments",
        body={"author": params.author, "body": params.body},
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_list_comments",
    annotations={
        "title": "List Comments",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_list_comments(params: ListCommentsInput) -> str:
    """List all comments on an issue.

    Args:
        params: Project slug and issue number.

    Returns:
        str: JSON array of comments ordered by creation time.
    """
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/issues/{params.number}/comments",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Labels
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tasks_add_label",
    annotations={
        "title": "Add Label to Issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tasks_add_label(params: AddLabelInput) -> str:
    """Attach a label to an issue. The label must already exist in the project.

    Args:
        params: Project slug, issue number, and label name.

    Returns:
        str: JSON with the attached label details.
    """
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/issues/{params.number}/labels",
        body={"label_name": params.label_name},
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_remove_label",
    annotations={
        "title": "Remove Label from Issue",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_remove_label(params: RemoveLabelInput) -> str:
    """Remove a label from an issue.

    Args:
        params: Project slug, issue number, and label ID.

    Returns:
        str: Confirmation or error message.
    """
    result = await _request(
        "DELETE",
        f"/api/projects/{params.project}/issues/{params.number}/labels/{params.label_id}",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Milestones
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tasks_set_milestone",
    annotations={
        "title": "Set Issue Milestone",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_set_milestone(params: SetMilestoneInput) -> str:
    """Set the milestone on an issue.

    Args:
        params: Project slug, issue number, milestone ID, and actor name.

    Returns:
        str: JSON with the updated issue.
    """
    result = await _request(
        "PATCH",
        f"/api/projects/{params.project}/issues/{params.number}",
        params={"actor": params.actor},
        body={"milestone_id": params.milestone_id},
    )
    return _format_result(result)


@mcp.tool(
    name="tasks_list_milestones",
    annotations={
        "title": "List Milestones",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_list_milestones(params: ListMilestonesInput) -> str:
    """List all milestones in a project with open/closed issue counts.

    Args:
        params: Project slug.

    Returns:
        str: JSON array of milestones.
    """
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/milestones",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Dependencies
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tasks_add_dependency",
    annotations={
        "title": "Add Issue Dependency",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tasks_add_dependency(params: AddDependencyInput) -> str:
    """Add a dependency: mark that an issue depends on another issue.

    Args:
        params: Project slug, issue number, and the number it depends on.

    Returns:
        str: JSON with the dependency relationship.
    """
    result = await _request(
        "POST",
        f"/api/projects/{params.project}/issues/{params.number}/dependencies",
        body={"depends_on_number": params.depends_on_number},
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Tools — Activity
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tasks_get_activity",
    annotations={
        "title": "Get Issue Activity",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tasks_get_activity(params: GetActivityInput) -> str:
    """Get the activity log for an issue (state changes, comments, assignments, etc.).

    Args:
        params: Project slug and issue number.

    Returns:
        str: JSON array of activity entries ordered by most recent first.
    """
    result = await _request(
        "GET",
        f"/api/projects/{params.project}/issues/{params.number}/activity",
    )
    return _format_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

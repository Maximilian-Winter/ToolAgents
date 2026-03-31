import typer
import json
from typing import Optional
from agora.cli.auth import require_session, api_request

tasks_app = typer.Typer()

def _project_slug(session: dict) -> str:
    slug = session.get("project")
    if not slug:
        raise SystemExit("No project set. Login with --project SLUG")
    return slug

@tasks_app.command()
def create(
    title: str = typer.Argument(..., help="Issue title"),
    body: Optional[str] = typer.Option(None, "--body", "-b", help="Issue body"),
    assignee: Optional[str] = typer.Option(None, "--assignee", "-a", help="Assignee agent name"),
    priority: str = typer.Option("none", "--priority", "-p", help="Priority: critical, high, medium, low, none"),
    labels: Optional[str] = typer.Option(None, "--labels", "-l", help="Comma-separated label names"),
):
    """Create a new issue."""
    session = require_session()
    slug = _project_slug(session)
    data = {
        "title": title,
        "reporter": session["agent_name"],
        "priority": priority,
    }
    if body:
        data["body"] = body
    if assignee:
        data["assignee"] = assignee
    if labels:
        data["labels"] = [l.strip() for l in labels.split(",")]
    result = api_request("POST", f"/api/projects/{slug}/issues", body=data)
    typer.echo(f"Created issue #{result['number']}: {result['title']}")

@tasks_app.command(name="list")
def list_issues(
    state: Optional[str] = typer.Option(None, "--state", "-s", help="Filter: open or closed"),
    assignee: Optional[str] = typer.Option(None, "--assignee", "-a"),
    label: Optional[str] = typer.Option(None, "--label", "-l"),
    priority: Optional[str] = typer.Option(None, "--priority", "-p"),
    limit: int = typer.Option(50, "--limit"),
):
    """List issues."""
    session = require_session()
    slug = _project_slug(session)
    params = {"limit": limit}
    if state:
        params["state"] = state
    if assignee:
        params["assignee"] = assignee
    if label:
        params["label"] = label
    if priority:
        params["priority"] = priority
    result = api_request("GET", f"/api/projects/{slug}/issues", params=params)
    if not result:
        typer.echo("No issues found.")
        return
    for issue in result:
        labels_str = ", ".join(l["name"] for l in issue.get("labels", []))
        state_icon = "O" if issue["state"] == "open" else "X"
        typer.echo(f"  {state_icon} #{issue['number']} [{issue['priority']}] {issue['title']}")
        if labels_str:
            typer.echo(f"    Labels: {labels_str}")
        if issue.get("assignee"):
            typer.echo(f"    Assignee: {issue['assignee']}")

@tasks_app.command()
def show(number: int = typer.Argument(..., help="Issue number")):
    """Show issue detail."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/issues/{number}")
    typer.echo(f"#{result['number']} [{result['state']}] [{result['priority']}] {result['title']}")
    if result.get("body"):
        typer.echo(f"\n{result['body']}\n")
    typer.echo(f"Reporter: {result['reporter']}")
    if result.get("assignee"):
        typer.echo(f"Assignee: {result['assignee']}")
    labels = [l["name"] for l in result.get("labels", [])]
    if labels:
        typer.echo(f"Labels: {', '.join(labels)}")
    typer.echo(f"Comments: {result.get('comment_count', 0)}")

@tasks_app.command()
def update(
    number: int = typer.Argument(..., help="Issue number"),
    title: Optional[str] = typer.Option(None, "--title"),
    body_text: Optional[str] = typer.Option(None, "--body"),
    state: Optional[str] = typer.Option(None, "--state", help="open or closed"),
    priority: Optional[str] = typer.Option(None, "--priority"),
    assignee: Optional[str] = typer.Option(None, "--assignee"),
):
    """Update an issue."""
    session = require_session()
    slug = _project_slug(session)
    data = {}
    if title:
        data["title"] = title
    if body_text:
        data["body"] = body_text
    if state:
        data["state"] = state
    if priority:
        data["priority"] = priority
    if assignee:
        data["assignee"] = assignee
    if not data:
        typer.echo("Nothing to update.")
        raise typer.Exit(1)
    params = {"actor": session["agent_name"]}
    result = api_request("PATCH", f"/api/projects/{slug}/issues/{number}", body=data, params=params)
    typer.echo(f"Updated issue #{result['number']}: {result['title']}")

@tasks_app.command()
def close(number: int = typer.Argument(..., help="Issue number")):
    """Close an issue."""
    session = require_session()
    slug = _project_slug(session)
    params = {"actor": session["agent_name"]}
    result = api_request("PATCH", f"/api/projects/{slug}/issues/{number}", body={"state": "closed"}, params=params)
    typer.echo(f"Closed issue #{result['number']}")

@tasks_app.command()
def reopen(number: int = typer.Argument(..., help="Issue number")):
    """Reopen an issue."""
    session = require_session()
    slug = _project_slug(session)
    params = {"actor": session["agent_name"]}
    result = api_request("PATCH", f"/api/projects/{slug}/issues/{number}", body={"state": "open"}, params=params)
    typer.echo(f"Reopened issue #{result['number']}")

@tasks_app.command()
def comment(
    number: int = typer.Argument(..., help="Issue number"),
    body: str = typer.Argument(..., help="Comment body"),
):
    """Add a comment to an issue."""
    session = require_session()
    slug = _project_slug(session)
    data = {"author": session["agent_name"], "body": body}
    result = api_request("POST", f"/api/projects/{slug}/issues/{number}/comments", body=data)
    typer.echo(f"Comment #{result['id']} added to issue #{number}")

@tasks_app.command()
def comments(number: int = typer.Argument(..., help="Issue number")):
    """List comments on an issue."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/issues/{number}/comments")
    if not result:
        typer.echo("No comments.")
        return
    for c in result:
        typer.echo(f"  #{c['id']} {c['author']} ({c['created_at'][:19]}):")
        typer.echo(f"    {c['body']}")

@tasks_app.command()
def label(
    number: int = typer.Argument(..., help="Issue number"),
    action: str = typer.Argument(..., help="add or remove"),
    label_name: str = typer.Argument(..., help="Label name"),
):
    """Add or remove a label from an issue."""
    session = require_session()
    slug = _project_slug(session)
    if action == "add":
        api_request("POST", f"/api/projects/{slug}/issues/{number}/labels", body={"label_name": label_name})
        typer.echo(f"Added label '{label_name}' to issue #{number}")
    elif action == "remove":
        # Need to resolve label_id first
        labels = api_request("GET", f"/api/projects/{slug}/labels")
        label_id = None
        for l in labels:
            if l["name"] == label_name:
                label_id = l["id"]
                break
        if label_id is None:
            raise SystemExit(f"Label '{label_name}' not found")
        api_request("DELETE", f"/api/projects/{slug}/issues/{number}/labels/{label_id}")
        typer.echo(f"Removed label '{label_name}' from issue #{number}")
    else:
        raise SystemExit("Action must be 'add' or 'remove'")

@tasks_app.command()
def activity(number: int = typer.Argument(..., help="Issue number")):
    """Show activity log for an issue."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/issues/{number}/activity")
    if not result:
        typer.echo("No activity.")
        return
    for a in result:
        typer.echo(f"  {a['created_at'][:19]} {a['actor']}: {a['action']}")
        if a.get("detail_json"):
            typer.echo(f"    {a['detail_json']}")

@tasks_app.command()
def milestones():
    """List all milestones in the project."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/milestones")
    if not result:
        typer.echo("No milestones.")
        return
    for m in result:
        due = m.get("due_date") or "no due date"
        typer.echo(f"  [{m['state']}] {m['title']} (id:{m['id']}, due: {due})")
        if m.get("description"):
            typer.echo(f"    {m['description']}")
        typer.echo(f"    open: {m.get('open_issues', '?')}  closed: {m.get('closed_issues', '?')}")

@tasks_app.command(name="set-milestone")
def set_milestone(
    number: int = typer.Argument(..., help="Issue number"),
    milestone_id: int = typer.Argument(..., help="Milestone ID (use 'milestones' to list)"),
):
    """Set the milestone on an issue."""
    session = require_session()
    slug = _project_slug(session)
    params = {"actor": session["agent_name"]}
    result = api_request("PATCH", f"/api/projects/{slug}/issues/{number}", body={"milestone_id": milestone_id}, params=params)
    typer.echo(f"Set milestone on issue #{result['number']}")

@tasks_app.command(name="add-dependency")
def add_dependency(
    number: int = typer.Argument(..., help="Issue number"),
    depends_on: int = typer.Argument(..., help="Issue number it depends on"),
):
    """Add a dependency: issue NUMBER depends on DEPENDS_ON."""
    session = require_session()
    slug = _project_slug(session)
    body = {"depends_on_number": depends_on}
    api_request("POST", f"/api/projects/{slug}/issues/{number}/dependencies", body=body)
    typer.echo(f"Issue #{number} now depends on #{depends_on}")

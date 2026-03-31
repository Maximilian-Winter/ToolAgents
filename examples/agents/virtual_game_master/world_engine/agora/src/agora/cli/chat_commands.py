import typer
import json
from typing import Optional
from agora.cli.auth import require_session, api_request

chat_app = typer.Typer()

def _project_slug(session: dict) -> str:
    """Get project slug from session or abort."""
    slug = session.get("project")
    if not slug:
        raise SystemExit("No project set. Login with --project or use agora login ... --project SLUG")
    return slug

@chat_app.command(name="create-room")
def create_room(
    name: str = typer.Argument(..., help="Room name"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Room topic"),
):
    """Create a new chat room in the project."""
    session = require_session()
    slug = _project_slug(session)
    body: dict = {"name": name}
    if topic:
        body["topic"] = topic
    result = api_request("POST", f"/api/projects/{slug}/rooms", body=body)
    typer.echo(f"Created room: {result['name']}")
    if result.get("topic"):
        typer.echo(f"Topic: {result['topic']}")


@chat_app.command()
def send(
    room: str = typer.Argument(..., help="Room name"),
    message: str = typer.Argument(..., help="Message content"),
    type: str = typer.Option("statement", "--type", "-t", help="Message type: statement, proposal, objection, consensus, question, answer"),
    reply_to: Optional[int] = typer.Option(None, "--reply-to", "-r", help="Message ID to reply to"),
    to: Optional[str] = typer.Option(None, "--to", help="Direct to a specific agent"),
):
    """Send a message to a room."""
    session = require_session()
    slug = _project_slug(session)
    body = {
        "sender": session["agent_name"],
        "content": message,
        "message_type": type,
    }
    if reply_to is not None:
        body["reply_to"] = reply_to
    if to:
        body["to"] = to
    result = api_request("POST", f"/api/projects/{slug}/rooms/{room}/messages", body=body)
    typer.echo(f"#{result['id']} [{result['message_type']}] {result['sender']}: {result['content']}")

@chat_app.command()
def poll(
    room: str = typer.Argument(..., help="Room name"),
    since: Optional[int] = typer.Option(None, "--since", help="Return messages after this ID"),
    limit: int = typer.Option(50, "--limit", "-l", help="Max messages"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by message type"),
):
    """Poll for new messages."""
    session = require_session()
    slug = _project_slug(session)
    params = {"limit": limit}
    if since is not None:
        params["since"] = since
    if type:
        params["message_type"] = type
    result = api_request("GET", f"/api/projects/{slug}/rooms/{room}/poll", params=params)
    msgs = result.get("messages", [])
    if not msgs:
        typer.echo("No new messages.")
        return
    for m in msgs:
        prefix = f"#{m['id']}"
        if m.get("reply_to"):
            prefix += f" (reply to #{m['reply_to']})"
        typer.echo(f"{prefix} [{m['message_type']}] {m['sender']}: {m['content']}")

@chat_app.command()
def wait(
    room: str = typer.Argument(..., help="Room name"),
    since: Optional[int] = typer.Option(None, "--since", help="Wait for messages after this ID"),
    timeout: float = typer.Option(30.0, "--timeout", help="Max seconds to wait"),
):
    """Wait for new messages (long-poll)."""
    session = require_session()
    slug = _project_slug(session)
    params = {"timeout": timeout}
    if since is not None:
        params["since"] = since
    result = api_request("GET", f"/api/projects/{slug}/rooms/{room}/wait", params=params)
    msgs = result.get("messages", [])
    if not msgs:
        typer.echo("(timeout, no new messages)")
        return
    for m in msgs:
        prefix = f"#{m['id']}"
        if m.get("reply_to"):
            prefix += f" (reply to #{m['reply_to']})"
        typer.echo(f"{prefix} [{m['message_type']}] {m['sender']}: {m['content']}")

@chat_app.command()
def rooms():
    """List rooms in the project."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/rooms")
    if not result:
        typer.echo("No rooms.")
        return
    for r in result:
        typer.echo(f"  {r['name']} — {r.get('topic', '(no topic)')}")

@chat_app.command(name="room-info")
def room_info(room: str = typer.Argument(..., help="Room name")):
    """Show room info."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/rooms/{room}")
    typer.echo(f"Room: {result['room']['name']}")
    typer.echo(f"Topic: {result['room'].get('topic', '—')}")
    typer.echo(f"Messages: {result['message_count']}")
    typer.echo(f"Round: {result['room']['current_round']}")
    members = [m['name'] for m in result.get('members', [])]
    typer.echo(f"Members: {', '.join(members) if members else '—'}")

@chat_app.command()
def threads(
    room: str = typer.Argument(..., help="Room name"),
    since: Optional[int] = typer.Option(None, "--since"),
):
    """Get threaded view of messages."""
    session = require_session()
    slug = _project_slug(session)
    params = {}
    if since is not None:
        params["since"] = since
    result = api_request("GET", f"/api/projects/{slug}/rooms/{room}/threads", params=params)
    typer.echo(json.dumps(result, indent=2, default=str))

@chat_app.command()
def summary(room: str = typer.Argument(..., help="Room name")):
    """Get discussion summary."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("GET", f"/api/projects/{slug}/rooms/{room}/summary")
    typer.echo(json.dumps(result, indent=2, default=str))

@chat_app.command()
def react(
    room: str = typer.Argument(..., help="Room name"),
    message_id: int = typer.Argument(..., help="Message ID"),
    emoji: str = typer.Argument(..., help="Emoji to react with"),
):
    """React to a message."""
    session = require_session()
    slug = _project_slug(session)
    body = {"sender": session["agent_name"], "emoji": emoji}
    result = api_request("POST", f"/api/projects/{slug}/rooms/{room}/messages/{message_id}/reactions", body=body)
    typer.echo(f"Reacted {emoji} to #{message_id}")

@chat_app.command(name="mark-read")
def mark_read(
    room: str = typer.Argument(..., help="Room name"),
    message_id: int = typer.Argument(..., help="Last read message ID"),
):
    """Mark messages as read up to an ID."""
    session = require_session()
    slug = _project_slug(session)
    body = {"agent": session["agent_name"], "last_read": message_id}
    api_request("PUT", f"/api/projects/{slug}/rooms/{room}/receipts", body=body)
    typer.echo(f"Marked read up to #{message_id}")

@chat_app.command()
def typing(room: str = typer.Argument(..., help="Room name")):
    """Signal that you're typing."""
    session = require_session()
    slug = _project_slug(session)
    body = {"sender": session["agent_name"]}
    api_request("POST", f"/api/projects/{slug}/rooms/{room}/typing", body=body)
    typer.echo("Typing indicator set.")

@chat_app.command(name="list-agents")
def list_agents():
    """List all registered agents."""
    session = require_session()
    result = api_request("GET", "/api/agents")
    if not result:
        typer.echo("No agents registered.")
        return
    for a in result:
        display = a.get("display_name") or a["name"]
        role = a.get("role") or "—"
        typer.echo(f"  {display} ({a['name']}) — role: {role}")

@chat_app.command()
def edit(
    room: str = typer.Argument(..., help="Room name"),
    message_id: int = typer.Argument(..., help="Message ID to edit"),
    content: str = typer.Argument(..., help="New message content"),
):
    """Edit a previously sent message."""
    session = require_session()
    slug = _project_slug(session)
    body = {"sender": session["agent_name"], "content": content}
    result = api_request("PUT", f"/api/projects/{slug}/rooms/{room}/messages/{message_id}", body=body)
    typer.echo(f"Edited #{result['id']}: {result['content']}")

@chat_app.command(name="advance-round")
def advance_round(room: str = typer.Argument(..., help="Room name")):
    """Advance the discussion to the next round."""
    session = require_session()
    slug = _project_slug(session)
    result = api_request("POST", f"/api/projects/{slug}/rooms/{room}/advance-round")
    typer.echo(f"Advanced to round {result['current_round']}")

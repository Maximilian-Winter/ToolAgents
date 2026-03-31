"""CLI commands for knowledge base operations."""

import sys
from typing import Optional

import typer

from agora.cli.auth import api_request, require_session

kb_app = typer.Typer(help="Knowledge base commands")


def _project_slug(session: dict) -> str:
    slug = session.get("project")
    if not slug:
        raise SystemExit("No project set. Run: agora login <name> --project SLUG")
    return slug


@kb_app.command("write")
def write(
    path: str = typer.Argument(..., help="Document path, e.g. architecture/api-design.md"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Document title"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    body: Optional[str] = typer.Option(None, "--body", "-b", help="Document content (or pipe via stdin)"),
):
    """Create or replace a knowledge base document."""
    session = require_session()
    slug = _project_slug(session)

    if body is None:
        if sys.stdin.isatty():
            raise SystemExit("Provide --body or pipe content via stdin")
        body = sys.stdin.read()

    payload: dict = {
        "path": path,
        "content": body,
        "author": session["agent_name"],
    }
    if title:
        payload["title"] = title
    if tags:
        payload["tags"] = tags

    # Use direct httpx call to get status code (201=created, 200=updated)
    import httpx
    url = f"{session['server_url']}/api/projects/{slug}/kb"
    headers = {}
    if session.get("session_token"):
        headers["Authorization"] = f"Bearer {session['session_token']}"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        action = "Created" if resp.status_code == 201 else "Updated"
        typer.echo(f"{action}: {result['path']}")


@kb_app.command("read")
def read(
    path: str = typer.Argument(..., help="Document path, e.g. architecture/api-design.md"),
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Section header to extract"),
):
    """Read a knowledge base document (or a specific section)."""
    session = require_session()
    slug = _project_slug(session)

    params = {}
    if section:
        params["section"] = section

    result = api_request("GET", f"/api/projects/{slug}/kb/{path}", params=params)
    if isinstance(result, dict):
        typer.echo(result["content"])


@kb_app.command("list")
def list_docs(
    prefix: Optional[str] = typer.Argument(None, help="Path prefix to filter, e.g. architecture/"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
):
    """List knowledge base documents."""
    session = require_session()
    slug = _project_slug(session)

    params: dict = {}
    if prefix:
        params["prefix"] = prefix
    if tag:
        params["tag"] = tag

    result = api_request("GET", f"/api/projects/{slug}/kb", params=params)
    if isinstance(result, list):
        for doc in result:
            typer.echo(f"{doc['path']} — \"{doc['title']}\"")
        if not result:
            typer.echo("No documents found.")


@kb_app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
):
    """Full-text search across knowledge base documents."""
    session = require_session()
    slug = _project_slug(session)

    params: dict = {"q": query, "limit": limit}
    if tag:
        params["tag"] = tag

    result = api_request("GET", f"/api/projects/{slug}/kb/search", params=params)
    if isinstance(result, list):
        for doc in result:
            typer.echo(f"{doc['path']} — {doc['snippet']}")
        if not result:
            typer.echo("No results found.")


@kb_app.command("tree")
def tree():
    """Display the knowledge base document tree."""
    session = require_session()
    slug = _project_slug(session)

    result = api_request("GET", f"/api/projects/{slug}/kb/tree")

    def _print_tree(nodes: list, indent: int = 0) -> None:
        for node in nodes:
            prefix = "  " * indent
            if node.get("children"):
                typer.echo(f"{prefix}{node['name']}/")
                _print_tree(node["children"], indent + 1)
            else:
                typer.echo(f"{prefix}{node['name']} — \"{node.get('title', '')}\"")

    if isinstance(result, list):
        if result:
            _print_tree(result)
        else:
            typer.echo("Knowledge base is empty.")


@kb_app.command("move")
def move(
    old_path: str = typer.Argument(..., help="Current document path"),
    new_path: str = typer.Argument(..., help="New document path"),
):
    """Move or rename a document."""
    session = require_session()
    slug = _project_slug(session)

    api_request("PATCH", f"/api/projects/{slug}/kb/{old_path}/move", body={"new_path": new_path})
    typer.echo(f"Moved {old_path} → {new_path}")


@kb_app.command("delete")
def delete(
    path: str = typer.Argument(..., help="Document path to delete"),
):
    """Delete a knowledge base document."""
    session = require_session()
    slug = _project_slug(session)

    api_request("DELETE", f"/api/projects/{slug}/kb/{path}")
    typer.echo(f"Deleted {path}")

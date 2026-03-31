import typer
from agora.cli.auth import save_session, load_session, clear_session, require_session, api_request
from agora.cli.chat_commands import chat_app
from agora.cli.task_commands import tasks_app
from agora.cli.kb_commands import kb_app

app = typer.Typer(name="agora", help="Agora CLI")
app.add_typer(chat_app, name="chat", help="Chat commands")
app.add_typer(tasks_app, name="tasks", help="Task/issue commands")
app.add_typer(kb_app, name="kb", help="Knowledge base commands")

@app.command()
def login(
    name: str = typer.Argument(..., help="Agent name"),
    server: str = typer.Option("http://127.0.0.1:8321", "--server", "-s", help="Server URL"),
    token: str = typer.Option(None, "--token", "-t", help="Agent token"),
    project: str = typer.Option(None, "--project", "-p", help="Project slug"),
):
    """Login as an agent."""
    body = {"name": name}
    if token:
        body["token"] = token
    if project:
        body["project"] = project

    # Use a temporary session (no auth needed for login)
    session = {"server_url": server, "session_token": None}
    result = api_request("POST", "/api/sessions/login", body=body, session=session)

    save_session(server, result["agent_name"], result["session_token"], result.get("project"))
    typer.echo(f"Logged in as {result['agent_name']}")
    if result.get("project"):
        typer.echo(f"Project: {result['project']}")

@app.command()
def logout():
    """Logout and clear session."""
    session = load_session()
    if session and session.get("session_token"):
        try:
            api_request("POST", "/api/sessions/logout", session=session)
        except SystemExit:
            pass  # Server may be down, still clear local session
    clear_session()
    typer.echo("Logged out.")

@app.command()
def status():
    """Show current session info."""
    session = load_session()
    if not session:
        typer.echo("Not logged in.")
        raise typer.Exit(1)
    typer.echo(f"Server:  {session['server_url']}")
    typer.echo(f"Agent:   {session['agent_name']}")
    typer.echo(f"Project: {session.get('project', '(none)')}")
    # Try to verify session is still valid
    try:
        result = api_request("GET", "/api/sessions/me", session=session)
        typer.echo("Session: valid")
    except SystemExit:
        typer.echo("Session: invalid or expired")

if __name__ == "__main__":
    app()

import json
import os
import httpx
from pathlib import Path
from typing import Optional

SESSION_DIR = Path.home() / ".agora"


def _session_file() -> Path:
    """Get session file path. Supports AGORA_SESSION env var
    so multiple agents can run concurrently with separate sessions."""
    custom = os.environ.get("AGORA_SESSION")
    if custom:
        return Path(custom)
    return SESSION_DIR / "session.json"


def save_session(server_url: str, agent_name: str, session_token: str, project: str | None = None):
    """Save session to file."""
    sf = _session_file()
    sf.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "server_url": server_url,
        "agent_name": agent_name,
        "session_token": session_token,
        "project": project,
    }
    sf.write_text(json.dumps(data, indent=2))

def load_session() -> dict | None:
    """Load session from file. Returns None if no session."""
    sf = _session_file()
    if not sf.exists():
        return None
    try:
        return json.loads(sf.read_text())
    except (json.JSONDecodeError, KeyError):
        return None

def clear_session():
    """Delete session file."""
    _session_file().unlink(missing_ok=True)

def require_session() -> dict:
    """Load session or raise error."""
    session = load_session()
    if not session:
        raise SystemExit("Not logged in. Run: agora login <name> --server URL --project SLUG")
    return session

def api_request(method: str, path: str, *, params: dict | None = None, body: dict | None = None, session: dict | None = None) -> dict | list | str:
    """Make HTTP request to the API server. Uses session for auth and base URL."""
    if session is None:
        session = require_session()
    url = f"{session['server_url']}{path}"
    headers = {}
    if session.get("session_token"):
        headers["Authorization"] = f"Bearer {session['session_token']}"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.request(method, url, params=params, json=body, headers=headers)
            if resp.status_code == 204:
                return {"ok": True}
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text[:300]
        raise SystemExit(f"Error {e.response.status_code}: {detail}")
    except httpx.ConnectError:
        raise SystemExit(f"Cannot connect to {session['server_url']}. Is the server running?")

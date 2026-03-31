"""Shared HTTP client for MCP servers talking to the Agora FastAPI service."""

import httpx
import json
import os


AGORA_URL = os.environ.get("AGORA_URL", "http://127.0.0.1:8321")


async def api_request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    body: dict | None = None,
    timeout: float = 15.0,
) -> dict | list | str:
    """Make a request to the Agora service and return parsed JSON."""
    url = f"{AGORA_URL}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.request(method, url, params=params, json=body)
            resp.raise_for_status()
            if resp.status_code == 204:
                return {"ok": True}
            return resp.json()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text[:300]
            return f"Error {e.response.status_code}: {detail}"
        except httpx.ConnectError:
            return (
                f"Error: Cannot connect to Agora service at {AGORA_URL}. "
                "Is the service running?"
            )
        except httpx.TimeoutException:
            return "Error: Request to Agora service timed out."


def format_result(data) -> str:
    """Format a response for the agent."""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, default=str)

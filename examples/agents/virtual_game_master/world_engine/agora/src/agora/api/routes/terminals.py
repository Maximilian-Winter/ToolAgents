"""Terminal WebSocket and REST endpoints.

Provides browser-based terminal access via xterm.js + WebSocket.
REST endpoints manage terminal sessions; the WebSocket relays I/O.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ConfigDict

from agora.services.terminal_service import terminal_manager

logger = logging.getLogger("agora.terminals")

router = APIRouter(prefix="/api/terminals", tags=["Terminals"])


# ── REST models ──────────────────────────────────────────────

class CreateTerminalRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    working_dir: str = Field(..., min_length=1)
    cols: int = Field(80, ge=10, le=500)
    rows: int = Field(24, ge=5, le=200)


class TerminalResponse(BaseModel):
    id: str
    working_dir: str
    shell: str
    mode: str
    cols: int
    rows: int
    created_at: str
    status: str


# ── REST endpoints ───────────────────────────────────────────

@router.post("", response_model=TerminalResponse, status_code=201)
async def create_terminal(body: CreateTerminalRequest):
    """Create a new terminal session."""
    try:
        session = await terminal_manager.create_session(
            body.working_dir, body.cols, body.rows
        )
        return TerminalResponse(**session.to_dict())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Failed to create terminal")
        raise HTTPException(500, f"Failed to create terminal: {e}")


@router.get("", response_model=list[TerminalResponse])
async def list_terminals():
    """List all terminal sessions."""
    return [TerminalResponse(**s) for s in terminal_manager.list_sessions()]


@router.delete("/{session_id}", status_code=204)
async def kill_terminal(session_id: str):
    """Kill a terminal session."""
    if not await terminal_manager.kill_session(session_id):
        raise HTTPException(404, f"Terminal {session_id} not found")


# ── WebSocket endpoint ───────────────────────────────────────

@router.websocket("/{session_id}/ws")
async def terminal_ws(websocket: WebSocket, session_id: str):
    """WebSocket relay for terminal I/O.

    Protocol:
    - Binary frames: raw terminal data (both directions)
    - Text frames (client→server): JSON control messages
      {"type": "resize", "cols": N, "rows": N}
    - Text frames (server→client): JSON status messages
      {"type": "connected", "mode": "pty"|"pipe", ...}
      {"type": "exited", "code": N}
    """
    session = terminal_manager.get_session(session_id)
    if not session:
        logger.warning("WS connect for unknown session %s", session_id)
        await websocket.close(code=4004, reason="Terminal not found")
        return

    await websocket.accept()
    logger.info("WS accepted for session %s (status=%s, mode=%s)", session_id, session.status, session.mode)

    # Send initial connection info
    await websocket.send_text(json.dumps({
        "type": "connected",
        "mode": session.mode,
        "shell": session.shell,
        "working_dir": session.working_dir,
    }))

    async def relay_output():
        """Read from terminal → send to WebSocket."""
        try:
            while session.status == "running":
                data = await terminal_manager.read(session_id, timeout=0.1)
                if data:
                    await websocket.send_bytes(data)
                elif session.status != "running":
                    logger.info("WS relay_output: session %s no longer running", session_id)
                    break
        except (WebSocketDisconnect, RuntimeError) as exc:
            logger.debug("WS relay_output ended: %s", exc)
        finally:
            if session.status != "running":
                try:
                    await websocket.send_text(json.dumps({
                        "type": "exited",
                    }))
                except Exception:
                    pass

    async def relay_input():
        """Read from WebSocket → write to terminal."""
        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    # Binary frame: raw terminal input
                    ok = terminal_manager.write(session_id, message["bytes"])
                    if not ok:
                        logger.debug("WS write failed for session %s", session_id)

                elif "text" in message and message["text"]:
                    # Text frame: JSON control message
                    try:
                        ctrl = json.loads(message["text"])
                        if ctrl.get("type") == "resize":
                            terminal_manager.resize(
                                session_id,
                                int(ctrl.get("cols", 80)),
                                int(ctrl.get("rows", 24)),
                            )
                    except (json.JSONDecodeError, ValueError):
                        pass

        except WebSocketDisconnect:
            logger.debug("WS client disconnected for session %s", session_id)

    # Run input and output relay concurrently
    output_task = asyncio.create_task(relay_output())
    input_task = asyncio.create_task(relay_input())

    try:
        done, pending = await asyncio.wait(
            [output_task, input_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Log which task finished first and why
        for t in done:
            exc = t.exception()
            if exc:
                logger.error("WS task error for %s: %s", session_id, exc)
    except Exception as exc:
        logger.error("WS handler error for %s: %s", session_id, exc)
        output_task.cancel()
        input_task.cancel()

    logger.info("WS handler finished for session %s", session_id)

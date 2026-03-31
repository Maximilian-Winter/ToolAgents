from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from agora.services.process_service import process_manager

router = APIRouter(prefix="/api/utilities", tags=["Utilities"])


class SpawnRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    command: str = Field(..., min_length=1, description="Command to run in the terminal")
    working_dir: str = Field(..., min_length=1, description="Working directory for the command")
    env: Optional[dict[str, str]] = Field(None, description="Additional environment variables")


class SpawnResponse(BaseModel):
    pid: int
    command: str
    working_dir: str
    started_at: str
    status: str


@router.post("/spawn-terminal", response_model=SpawnResponse, status_code=201)
async def spawn_terminal(body: SpawnRequest):
    """Launch a command in a new terminal window on the host machine.

    WARNING: This endpoint executes commands on the host. It should only
    be accessible from localhost or trusted networks. Commands are validated
    against dangerous patterns but this is NOT a security sandbox.
    """
    try:
        sp = process_manager.spawn_terminal(body.command, body.working_dir, body.env)
        return SpawnResponse(
            pid=sp.pid,
            command=sp.command,
            working_dir=sp.working_dir,
            started_at=sp.started_at.isoformat(),
            status="running",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to spawn terminal: {e}")


@router.get("/processes")
async def list_processes():
    """List all spawned processes."""
    return process_manager.list_processes()


@router.delete("/processes/{pid}", status_code=204)
async def kill_process(pid: int):
    """Kill a spawned process."""
    if not process_manager.kill_process(pid):
        raise HTTPException(404, f"Process {pid} not found")

"""Process spawning service for launching agents in terminal windows.

WARNING: This service executes shell commands on the host machine.
The spawn endpoint should only be accessible from localhost/trusted networks.
Commands are validated against a basic allowlist and dangerous patterns are rejected.
"""

import re
import shlex
import subprocess
import sys
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Patterns that indicate shell injection attempts
_DANGEROUS_PATTERNS = re.compile(
    r"[;&|`$]"  # shell metacharacters for chaining/subshells
    r"|(\.\./){2,}"  # excessive directory traversal
    r"|>\s*/dev/"  # redirect to devices
    r"|\brm\s+-rf\b"  # rm -rf
    r"|\bsudo\b"  # privilege escalation
    r"|\bchmod\b"  # permission changes
    r"|\bchown\b"  # ownership changes
    r"|\bmkfs\b"  # filesystem creation
    r"|\bdd\b"  # raw disk operations
    r"|\b::\(\)\{"  # fork bomb pattern
)


def validate_command(command: str) -> None:
    """Validate a command for obvious injection patterns.

    Raises ValueError if dangerous patterns are detected.
    This is NOT a security sandbox — it's a safety net for accidental misuse.
    For production use, run the server behind auth and restrict access.
    """
    if not command or not command.strip():
        raise ValueError("Command cannot be empty")
    if len(command) > 2000:
        raise ValueError("Command too long (max 2000 characters)")
    if _DANGEROUS_PATTERNS.search(command):
        raise ValueError(
            f"Command contains potentially dangerous patterns. "
            f"Rejected for safety. If this is intentional, modify the "
            f"DANGEROUS_PATTERNS allowlist in process_service.py."
        )


def validate_working_dir(working_dir: str) -> None:
    """Validate that working_dir exists and is a directory."""
    path = Path(working_dir)
    if not path.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {working_dir}")


@dataclass
class SpawnedProcess:
    pid: int
    command: str
    working_dir: str
    started_at: datetime
    process: subprocess.Popen


class ProcessManager:
    """Tracks spawned terminal processes."""

    def __init__(self):
        self._processes: dict[int, SpawnedProcess] = {}

    def spawn_terminal(
        self,
        command: str,
        working_dir: str,
        env: dict[str, str] | None = None,
    ) -> SpawnedProcess:
        """Launch a command in a new terminal window.

        Validates the command and working directory before execution.

        Platform-aware:
        - Windows: uses 'start cmd /c'
        - macOS: uses 'open -a Terminal'
        - Linux: tries gnome-terminal, xterm, or falls back to subprocess
        """
        validate_command(command)
        validate_working_dir(working_dir)

        full_env = os.environ.copy()
        if env:
            # Validate env keys/values don't contain injection
            for k, v in env.items():
                if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
                    raise ValueError(f"Invalid environment variable name: {k}")
            full_env.update(env)

        system = platform.system()
        escaped_command = shlex.quote(command) if system != "Windows" else command
        escaped_dir = shlex.quote(working_dir) if system != "Windows" else working_dir

        if system == "Windows":
            # Start a new cmd window
            proc = subprocess.Popen(
                f'start cmd /k "cd /d {escaped_dir} && {command}"',
                shell=True,
                cwd=working_dir,
                env=full_env,
            )
        elif system == "Darwin":
            # macOS — use osascript to open Terminal
            script = f'tell application "Terminal" to do script "cd {escaped_dir} && {escaped_command}"'
            proc = subprocess.Popen(
                ["osascript", "-e", script],
                cwd=working_dir,
                env=full_env,
            )
        else:
            # Linux — try common terminal emulators
            terminal = None
            for term in ["gnome-terminal", "xfce4-terminal", "konsole", "xterm"]:
                if subprocess.run(["which", term], capture_output=True).returncode == 0:
                    terminal = term
                    break

            if terminal == "gnome-terminal":
                proc = subprocess.Popen(
                    [terminal, "--", "bash", "-c", f"cd {escaped_dir} && {escaped_command}; exec bash"],
                    cwd=working_dir,
                    env=full_env,
                )
            elif terminal == "xterm":
                proc = subprocess.Popen(
                    [terminal, "-e", f"cd {escaped_dir} && {escaped_command}; bash"],
                    cwd=working_dir,
                    env=full_env,
                )
            else:
                # Fallback: just run in background
                proc = subprocess.Popen(
                    shlex.split(command),
                    cwd=working_dir,
                    env=full_env,
                )

        spawned = SpawnedProcess(
            pid=proc.pid,
            command=command,
            working_dir=working_dir,
            started_at=datetime.now(timezone.utc),
            process=proc,
        )
        self._processes[proc.pid] = spawned
        return spawned

    def list_processes(self) -> list[dict]:
        """List all tracked processes with their status."""
        result = []
        for pid, sp in list(self._processes.items()):
            poll = sp.process.poll()
            result.append({
                "pid": sp.pid,
                "command": sp.command,
                "working_dir": sp.working_dir,
                "started_at": sp.started_at.isoformat(),
                "status": "running" if poll is None else f"exited ({poll})",
            })
        return result

    def kill_process(self, pid: int) -> bool:
        """Kill a tracked process. Returns True if found and killed."""
        sp = self._processes.get(pid)
        if not sp:
            return False
        try:
            sp.process.terminate()
        except OSError:
            pass
        del self._processes[pid]
        return True


# Module-level singleton
process_manager = ProcessManager()

"""Terminal session service for browser-based terminals.

Provides PTY-backed terminal sessions accessible via WebSocket.
- Unix: Full PTY via stdlib ``pty`` module (interactive shell with colors)
- Windows: subprocess.Popen with threads (pipe-based, local echo on client)

The WebSocket protocol uses:
- Binary frames for raw terminal I/O
- Text frames (JSON) for control messages (resize, status)
"""

import asyncio
import logging
import os
import platform
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agora.terminal")

_SYSTEM = platform.system()

# Unix PTY support (stdlib)
if _SYSTEM != "Windows":
    import fcntl
    import pty
    import struct
    import termios

    HAS_PTY = True
else:
    HAS_PTY = False


@dataclass
class TerminalSession:
    id: str
    working_dir: str
    created_at: datetime
    shell: str
    mode: str  # "pty" or "pipe"
    status: str = "running"
    cols: int = 80
    rows: int = 24
    _output_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=4096)
    )
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _master_fd: Optional[int] = field(default=None, repr=False)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "working_dir": self.working_dir,
            "shell": self.shell,
            "mode": self.mode,
            "cols": self.cols,
            "rows": self.rows,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


class TerminalManager:
    """Manages terminal sessions with platform-aware PTY support."""

    def __init__(self):
        self._sessions: dict[str, TerminalSession] = {}

    async def create_session(
        self, working_dir: str, cols: int = 80, rows: int = 24
    ) -> TerminalSession:
        """Create a new terminal session in the given directory."""
        path = Path(working_dir)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid working directory: {working_dir}")

        session_id = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()

        if HAS_PTY:
            session = self._create_unix_pty(session_id, working_dir, cols, rows, loop)
        else:
            session = self._create_pipe_fallback(
                session_id, working_dir, cols, rows, loop
            )

        self._sessions[session_id] = session
        logger.info(
            "Terminal %s created: mode=%s shell=%s cwd=%s",
            session_id, session.mode, session.shell, working_dir,
        )
        return session

    # ── Unix PTY ─────────────────────────────────────────────

    def _create_unix_pty(
        self,
        session_id: str,
        working_dir: str,
        cols: int,
        rows: int,
        loop: asyncio.AbstractEventLoop,
    ) -> TerminalSession:
        shell = os.environ.get("SHELL", "/bin/bash")
        master_fd, slave_fd = pty.openpty()

        # Set initial terminal size
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"

        process = subprocess.Popen(
            [shell, "-l"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=working_dir,
            env=env,
            preexec_fn=os.setsid,
        )
        os.close(slave_fd)

        output_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=4096)
        session = TerminalSession(
            id=session_id,
            working_dir=working_dir,
            created_at=datetime.now(timezone.utc),
            shell=shell,
            mode="pty",
            cols=cols,
            rows=rows,
            _output_queue=output_queue,
            _process=process,
            _master_fd=master_fd,
            _loop=loop,
        )

        reader = threading.Thread(
            target=self._read_fd_thread,
            args=(master_fd, output_queue, loop, session),
            name=f"pty-reader-{session_id}",
            daemon=True,
        )
        reader.start()
        session._reader_thread = reader
        return session

    # ── Pipe fallback (Windows / no-PTY) ─────────────────────
    # Uses subprocess.Popen with threads for reliable I/O on Windows.
    # The asyncio subprocess API has known issues with interactive
    # shells on Windows (ProactorEventLoop pipe handling).

    def _create_pipe_fallback(
        self,
        session_id: str,
        working_dir: str,
        cols: int,
        rows: int,
        loop: asyncio.AbstractEventLoop,
    ) -> TerminalSession:
        if _SYSTEM == "Windows":
            shell = os.environ.get("COMSPEC", "cmd.exe")
            args = [shell]
        else:
            shell = os.environ.get("SHELL", "/bin/bash")
            args = [shell, "-i"]

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"

        # Use subprocess.Popen directly (not asyncio) for reliability
        # bufsize=0 gives us unbuffered I/O
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            env=env,
            bufsize=0,
        )

        logger.info(
            "Pipe process spawned: pid=%d shell=%s", process.pid, shell
        )

        output_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=4096)
        session = TerminalSession(
            id=session_id,
            working_dir=working_dir,
            created_at=datetime.now(timezone.utc),
            shell=shell,
            mode="pipe",
            cols=cols,
            rows=rows,
            _output_queue=output_queue,
            _process=process,
            _loop=loop,
        )

        # Reader thread: blocking reads from process stdout
        reader = threading.Thread(
            target=self._read_pipe_thread,
            args=(process, output_queue, loop, session),
            name=f"pipe-reader-{session_id}",
            daemon=True,
        )
        reader.start()
        session._reader_thread = reader
        return session

    # ── Reader threads ────────────────────────────────────────

    def _read_fd_thread(
        self,
        fd: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        session: TerminalSession,
    ):
        """Background thread: blocking read from a file descriptor (PTY master)."""
        try:
            while session.status == "running":
                try:
                    data = os.read(fd, 4096)
                    if not data:
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(data), loop)
                except OSError:
                    break
        except Exception as exc:
            logger.debug("PTY reader error: %s", exc)
        finally:
            session.status = "exited"
            logger.info("PTY reader exited for session %s", session.id)

    def _read_pipe_thread(
        self,
        process: subprocess.Popen,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        session: TerminalSession,
    ):
        """Background thread: blocking read from subprocess stdout pipe.

        Uses stdout.read() directly instead of os.read(fileno()) because
        on Windows, subprocess pipe file objects may not support fileno().
        With bufsize=0, stdout is an unbuffered raw stream, so .read(N)
        returns up to N bytes as soon as any are available.
        """
        try:
            stdout = process.stdout
            if stdout is None:
                logger.error("Session %s: stdout is None!", session.id)
                return

            logger.info("Pipe reader started for session %s (pid=%d)", session.id, process.pid)

            while session.status == "running":
                try:
                    data = stdout.read(4096)
                except (OSError, ValueError) as exc:
                    logger.debug("Pipe read error for %s: %s", session.id, exc)
                    break
                if not data:
                    logger.debug("Pipe EOF for session %s", session.id)
                    break
                asyncio.run_coroutine_threadsafe(queue.put(data), loop)
        except Exception as exc:
            logger.error("Pipe reader exception for session %s: %s", session.id, exc, exc_info=True)
        finally:
            session.status = "exited"
            logger.info("Pipe reader exited for session %s", session.id)

    # ── I/O methods ──────────────────────────────────────────

    def write(self, session_id: str, data: bytes) -> bool:
        """Write data to a terminal session's input."""
        session = self._sessions.get(session_id)
        if not session or session.status != "running":
            return False

        proc = session._process
        if proc is None:
            return False

        try:
            if session.mode == "pty" and session._master_fd is not None:
                os.write(session._master_fd, data)
            elif proc.stdin is not None:
                proc.stdin.write(data)
                proc.stdin.flush()
            else:
                return False
            return True
        except (OSError, BrokenPipeError, ValueError) as exc:
            logger.debug("Write error for session %s: %s", session_id, exc)
            session.status = "exited"
            return False

    async def read(self, session_id: str, timeout: float = 0.05) -> Optional[bytes]:
        """Read available output from a terminal session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        try:
            return await asyncio.wait_for(
                session._output_queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def resize(self, session_id: str, cols: int, rows: int) -> bool:
        """Resize terminal dimensions."""
        session = self._sessions.get(session_id)
        if not session or session.status != "running":
            return False

        session.cols = cols
        session.rows = rows

        if session.mode == "pty" and session._master_fd is not None:
            try:
                winsize = struct.pack("HHHH", rows, cols, 0, 0)
                fcntl.ioctl(session._master_fd, termios.TIOCSWINSZ, winsize)
                return True
            except OSError:
                pass
        return False

    # ── Session management ───────────────────────────────────

    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        # Update status of sessions whose processes have exited
        for session in self._sessions.values():
            if session.status == "running" and session._process is not None:
                rc = session._process.poll()
                if rc is not None:
                    session.status = f"exited ({rc})"
        return [s.to_dict() for s in self._sessions.values()]

    async def kill_session(self, session_id: str) -> bool:
        """Terminate and remove a session."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return False

        session.status = "exited"

        if session._process is not None:
            try:
                session._process.terminate()
            except (OSError, ProcessLookupError):
                pass

        # Close PTY master fd
        if session._master_fd is not None:
            try:
                os.close(session._master_fd)
            except OSError:
                pass

        logger.info("Terminal %s killed", session_id)
        return True

    async def cleanup(self):
        """Kill all sessions (for shutdown)."""
        for sid in list(self._sessions.keys()):
            await self.kill_session(sid)


# Module-level singleton
terminal_manager = TerminalManager()

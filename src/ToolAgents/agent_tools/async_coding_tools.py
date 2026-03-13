"""
Async Claude Code-style Coding Tools
=======================================

Async version of CodingTools. Inherits all sync logic and wraps each tool
method as async:

- **bash**: True async via asyncio.create_subprocess_exec
- **All other tools**: Offloaded to thread pool via asyncio.to_thread
- **get_tools()**: Returns AsyncFunctionTool instances

Also provides async sub-agent factories using AsyncAgentHarness.
"""

import asyncio
import uuid
from typing import List, Optional

from ToolAgents import FunctionTool
from ToolAgents.function_tool import AsyncFunctionTool

from .coding_tools import CodingTools


class AsyncCodingTools(CodingTools):
    """Async version of CodingTools.

    Inherits from CodingTools to reuse all helper methods and validation logic.
    Each tool method is overridden as async:
    - bash uses asyncio.create_subprocess_exec for true async subprocess handling
    - All other tools use asyncio.to_thread to offload blocking file I/O

    Args:
        working_directory: Root directory for file operations. Defaults to cwd.
        shell: Shell to use for bash commands.
        allowed_directories: Optional list of allowed directories.
    """

    # ------------------------------------------------------------------
    # Tool: Bash (true async)
    # ------------------------------------------------------------------

    async def bash(
        self,
        command: str,
        timeout: int = 120,
    ) -> str:
        """Execute a shell command asynchronously and return its output.

        Uses asyncio.create_subprocess_exec for non-blocking execution.
        The command runs in the current working directory.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. Default is 120.

        Returns:
            str: Command output or JSON error details.
        """
        import json

        try:
            if "cmd" in self._shell.lower():
                args = [self._shell, "/c", command]
            else:
                args = [self._shell, "-c", command]

            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=self._working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return json.dumps({
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds.",
                })

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            returncode = process.returncode

            result = {
                "success": returncode == 0,
                "exit_code": returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
            }

            if not result["stderr"]:
                del result["stderr"]
            if result["success"]:
                output = stdout_text.rstrip()
                if not output:
                    return "Command completed successfully (no output)."
                return output

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
            })

    # ------------------------------------------------------------------
    # Tools: File operations (offloaded to thread pool)
    # ------------------------------------------------------------------

    async def read_file(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 0,
    ) -> str:
        """Read a file asynchronously and return its content with line numbers.

        Args:
            file_path: Path to the file (absolute or relative to working directory).
            offset: Line number to start reading from (0-based). Default is 0.
            limit: Maximum number of lines to read. 0 means read all. Default is 0.

        Returns:
            str: File content with line numbers, or an error message.
        """
        return await asyncio.to_thread(super().read_file, file_path, offset, limit)

    async def write_file(self, file_path: str, content: str) -> str:
        """Create or overwrite a file asynchronously.

        Args:
            file_path: Path to the file (absolute or relative to working directory).
            content: The full content to write to the file.

        Returns:
            str: Success or error message.
        """
        return await asyncio.to_thread(super().write_file, file_path, content)

    async def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file asynchronously by performing exact string replacement.

        Args:
            file_path: Path to the file (absolute or relative to working directory).
            old_string: The exact text to find and replace.
            new_string: The replacement text.
            replace_all: If True, replace all occurrences. Default is False.

        Returns:
            str: Success message with diff, or error message.
        """
        return await asyncio.to_thread(
            super().edit_file, file_path, old_string, new_string, replace_all
        )

    async def glob_files(self, pattern: str, path: str = "") -> str:
        """Find files matching a glob pattern asynchronously.

        Args:
            pattern: Glob pattern to match (e.g., '**/*.py').
            path: Directory to search in. Empty string means working directory.

        Returns:
            str: Newline-separated list of matching file paths.
        """
        return await asyncio.to_thread(super().glob_files, pattern, path)

    async def grep_files(
        self,
        pattern: str,
        path: str = "",
        file_glob: str = "",
        context_lines: int = 0,
        max_results: int = 50,
        case_insensitive: bool = False,
    ) -> str:
        """Search file contents asynchronously using a regular expression.

        Args:
            pattern: Regular expression pattern to search for.
            path: Directory or file to search in.
            file_glob: Glob pattern to filter files.
            context_lines: Context lines around matches. Default 0.
            max_results: Maximum matches. Default 50.
            case_insensitive: Case-insensitive matching. Default False.

        Returns:
            str: Matching lines with file paths and line numbers.
        """
        return await asyncio.to_thread(
            super().grep_files,
            pattern, path, file_glob, context_lines, max_results, case_insensitive,
        )

    async def list_directory(
        self,
        path: str = "",
        depth: int = 1,
        show_hidden: bool = False,
    ) -> str:
        """List directory contents asynchronously.

        Args:
            path: Directory to list. Empty string means working directory.
            depth: How many levels deep to recurse. Default is 1.
            show_hidden: Whether to include hidden files. Default False.

        Returns:
            str: Formatted directory listing.
        """
        return await asyncio.to_thread(
            super().list_directory, path, depth, show_hidden
        )

    async def diff_files(
        self,
        file_a: str,
        file_b: str,
        context_lines: int = 3,
    ) -> str:
        """Compare two files asynchronously and show a unified diff.

        Args:
            file_a: Path to the first file.
            file_b: Path to the second file.
            context_lines: Context lines around changes. Default 3.

        Returns:
            str: Unified diff output.
        """
        return await asyncio.to_thread(
            super().diff_files, file_a, file_b, context_lines
        )

    # ------------------------------------------------------------------
    # Tool collection (AsyncFunctionTool instances)
    # ------------------------------------------------------------------

    def get_tools(self) -> List[AsyncFunctionTool]:
        """Get all async coding tools as AsyncFunctionTool instances.

        Returns:
            List[AsyncFunctionTool]: The complete set of async coding tools:
                - bash (true async subprocess)
                - read_file, write_file, edit_file (async via thread pool)
                - glob_files, grep_files (async via thread pool)
                - list_directory, diff_files (async via thread pool)
        """
        return [
            AsyncFunctionTool(self.bash),
            AsyncFunctionTool(self.read_file),
            AsyncFunctionTool(self.write_file),
            AsyncFunctionTool(self.edit_file),
            AsyncFunctionTool(self.glob_files),
            AsyncFunctionTool(self.grep_files),
            AsyncFunctionTool(self.list_directory),
            AsyncFunctionTool(self.diff_files),
        ]

    # ------------------------------------------------------------------
    # Async Sub-Agent Factories
    # ------------------------------------------------------------------

    @staticmethod
    def create_async_sub_agent_tool(
        sub_agent_api,
        sub_agent_name: str = "sub_agent",
        sub_agent_tools: Optional[List] = None,
        max_context_tokens: int = 128000,
        reserve_tokens: int = 4096,
        strategy: str = "sliding_window",
    ) -> AsyncFunctionTool:
        """Create an async single-turn sub-agent tool.

        Args:
            sub_agent_api: An AsyncChatAPIProvider instance for the sub-agent.
            sub_agent_name: Name for the tool. Default is 'sub_agent'.
            sub_agent_tools: Optional list of tools for the sub-agent.
            max_context_tokens: Max context window. Default 128000.
            reserve_tokens: Reserved response tokens. Default 4096.
            strategy: Context trimming strategy. Default 'sliding_window'.

        Returns:
            AsyncFunctionTool: An async tool for single-turn sub-agent use.
        """
        from ToolAgents.agent_harness import create_async_harness

        async def sub_agent(
            task: str,
            system_prompt: str = "You are a helpful sub-agent. Complete the given task and return a clear, concise result.",
        ) -> str:
            """Spawn an async sub-agent to handle a complex or independent task.

            Args:
                task: A clear description of what the sub-agent should accomplish.
                system_prompt: System prompt for the sub-agent.

            Returns:
                str: The sub-agent's final response.
            """
            try:
                harness = create_async_harness(
                    provider=sub_agent_api,
                    system_prompt=system_prompt,
                    max_context_tokens=max_context_tokens,
                    reserve_tokens=reserve_tokens,
                    strategy=strategy,
                    streaming=False,
                    tools=sub_agent_tools,
                )
                response = await harness.chat(task)
                return response
            except Exception as e:
                return f"Sub-agent error: {e}"

        tool = AsyncFunctionTool(sub_agent)
        tool.model.__name__ = sub_agent_name
        return tool

    @staticmethod
    def create_async_multi_turn_sub_agent_tools(
        sub_agent_api,
        sub_agent_tools: Optional[List] = None,
        max_context_tokens: int = 128000,
        reserve_tokens: int = 4096,
        strategy: str = "sliding_window",
    ) -> List[AsyncFunctionTool]:
        """Create async multi-turn sub-agent tools with session persistence.

        Returns 3 async tools for managing persistent sub-agent sessions:
        1. multi_turn_sub_agent - Send a message to a new or existing session
        2. multi_turn_sub_agent_list_sessions - List active sessions
        3. multi_turn_sub_agent_close_session - Close a session

        Args:
            sub_agent_api: An AsyncChatAPIProvider instance.
            sub_agent_tools: Optional tools for the sub-agent.
            max_context_tokens: Max context window. Default 128000.
            reserve_tokens: Reserved response tokens. Default 4096.
            strategy: Context trimming strategy. Default 'sliding_window'.

        Returns:
            List[AsyncFunctionTool]: Three async tools for multi-turn sub-agent management.
        """
        from ToolAgents.agent_harness import create_async_harness

        # Closure-scoped session storage
        sessions: dict = {}  # session_id -> AsyncAgentHarness

        async def multi_turn_sub_agent(
            task: str,
            session_id: str = "",
            system_prompt: str = "You are a helpful sub-agent. Complete the given task and return a clear, concise result.",
        ) -> str:
            """Send a message to a new or existing async sub-agent session.

            Args:
                task: The message to send to the sub-agent.
                session_id: Empty string creates a new session. Provide an
                    existing session ID to continue a conversation.
                system_prompt: System prompt (only used when creating a new session).

            Returns:
                str: The sub-agent's response, prefixed with session ID for new sessions.
            """
            try:
                if session_id and session_id in sessions:
                    harness = sessions[session_id]
                    response = await harness.chat(task)
                    return response
                elif session_id and session_id not in sessions:
                    return f"Error: Session '{session_id}' not found. Use empty session_id to create a new session."
                else:
                    new_id = uuid.uuid4().hex[:8]
                    harness = create_async_harness(
                        provider=sub_agent_api,
                        system_prompt=system_prompt,
                        max_context_tokens=max_context_tokens,
                        reserve_tokens=reserve_tokens,
                        strategy=strategy,
                        streaming=False,
                        tools=sub_agent_tools,
                    )
                    sessions[new_id] = harness
                    response = await harness.chat(task)
                    return f"[New Session: {new_id}]\n{response}"
            except Exception as e:
                return f"Multi-turn sub-agent error: {e}"

        async def multi_turn_sub_agent_list_sessions() -> str:
            """List all active async sub-agent sessions with their turn counts.

            Returns:
                str: Formatted list of active sessions.
            """
            if not sessions:
                return "No active sub-agent sessions."

            lines = [f"Active sessions ({len(sessions)}):"]
            for sid, harness in sessions.items():
                lines.append(f"  {sid}: {harness.turn_count} turn(s)")
            return "\n".join(lines)

        async def multi_turn_sub_agent_close_session(session_id: str) -> str:
            """Close and free an async sub-agent session.

            Args:
                session_id: The session ID to close.

            Returns:
                str: Confirmation message.
            """
            if session_id in sessions:
                turns = sessions[session_id].turn_count
                del sessions[session_id]
                return f"Session '{session_id}' closed ({turns} turns completed)."
            return f"Error: Session '{session_id}' not found."

        tool_send = AsyncFunctionTool(multi_turn_sub_agent)
        tool_list = AsyncFunctionTool(multi_turn_sub_agent_list_sessions)
        tool_close = AsyncFunctionTool(multi_turn_sub_agent_close_session)

        return [tool_send, tool_list, tool_close]

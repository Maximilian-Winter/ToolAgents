"""
Claude Code-style Coding Tools
================================

A clean, focused set of tools inspired by Claude Code's tool design:

- **Bash**: Execute shell commands with timeout and persistent working directory
- **ReadFile**: Read files with line numbers, offset/limit for large files
- **WriteFile**: Create or overwrite files
- **EditFile**: Exact string replacement in files (unique match required)
- **GlobFiles**: Fast file pattern matching
- **GrepFiles**: Regex content search with context lines
- **ListDirectory**: List directory contents with depth control
- **DiffFiles**: Compare two files with unified diff output
- **SubAgent**: Spawn a sub-agent with its own ChatAPI, tools, and system prompt
- **MultiTurnSubAgent**: Multi-turn sub-agent sessions with persistence

Design principles:
- Each tool does ONE thing well
- Simple inputs, structured outputs
- Let the LLM compose simple tools into complex workflows
"""

import difflib
import glob
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

from ToolAgents import FunctionTool
from ToolAgents.function_tool import ToolRegistry


class CodingTools:
    """Claude Code-style tools for coding agents.

    Provides a focused set of tools that are easy for LLMs to use effectively.
    Each tool has a clear purpose with minimal parameter complexity.

    Args:
        working_directory: Root directory for file operations. Defaults to cwd.
        shell: Shell to use for bash commands (e.g., "bash", "cmd", "powershell").
               Defaults to platform-appropriate shell.
        allowed_directories: Optional list of directories the tools are allowed to
                             access. If None, no restriction is applied.
    """

    def __init__(
        self,
        working_directory: Optional[str] = None,
        shell: Optional[str] = None,
        allowed_directories: Optional[List[str]] = None,
    ):
        self._working_directory = os.path.abspath(working_directory or os.getcwd())
        self._shell = shell or self._default_shell()
        self._allowed_directories = (
            [os.path.abspath(d) for d in allowed_directories]
            if allowed_directories
            else None
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def working_directory(self) -> str:
        return self._working_directory

    @working_directory.setter
    def working_directory(self, value: str) -> None:
        self._working_directory = os.path.abspath(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_shell() -> str:
        if os.name == "nt":
            return "cmd"
        return os.environ.get("SHELL", "/bin/bash")

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the working directory."""
        if os.path.isabs(path):
            resolved = os.path.abspath(path)
        else:
            resolved = os.path.abspath(os.path.join(self._working_directory, path))
        self._check_allowed(resolved)
        return resolved

    def _check_allowed(self, path: str) -> None:
        """Raise if path is outside allowed directories."""
        if self._allowed_directories is None:
            return
        norm = os.path.normcase(os.path.abspath(path))
        for allowed in self._allowed_directories:
            if norm.startswith(os.path.normcase(allowed)):
                return
        raise PermissionError(
            f"Access denied: '{path}' is outside allowed directories."
        )

    # ------------------------------------------------------------------
    # Tool: Bash
    # ------------------------------------------------------------------

    def bash(
        self,
        command: str,
        timeout: int = 120,
    ) -> str:
        """Execute a shell command and return its output.

        The command runs in the current working directory. Use this for
        system commands, builds, tests, git operations, package management,
        and any other shell task.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. Default is 120.

        Returns:
            str: JSON with keys: success, exit_code, stdout, stderr, error.
        """
        try:
            # On Windows with cmd, wrap in /c; for bash-like shells use -c
            if "cmd" in self._shell.lower():
                args = [self._shell, "/c", command]
            else:
                args = [self._shell, "-c", command]

            process = subprocess.run(
                args,
                cwd=self._working_directory,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            result = {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

            # Compact output: omit empty fields
            if not result["stderr"]:
                del result["stderr"]
            if result["success"]:
                # For successful commands, just return stdout directly for brevity
                output = process.stdout.rstrip()
                if not output:
                    return "Command completed successfully (no output)."
                return output

            return json.dumps(result, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps({
                "success": False,
                "error": f"Command timed out after {timeout} seconds.",
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
            })

    # ------------------------------------------------------------------
    # Tool: ReadFile
    # ------------------------------------------------------------------

    def read_file(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 0,
    ) -> str:
        """Read a file and return its content with line numbers.

        Returns content in 'cat -n' format with line numbers for easy reference.
        For large files, use offset and limit to read specific ranges.

        Args:
            file_path: Path to the file (absolute or relative to working directory).
            offset: Line number to start reading from (0-based). Default is 0.
            limit: Maximum number of lines to read. 0 means read all. Default is 0.

        Returns:
            str: File content with line numbers, or an error message.
        """
        resolved = self._resolve_path(file_path)

        if not os.path.exists(resolved):
            return f"Error: File '{file_path}' does not exist."

        if os.path.isdir(resolved):
            return f"Error: '{file_path}' is a directory, not a file."

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply offset
            if offset > 0:
                if offset >= total_lines:
                    return f"Error: Offset {offset} exceeds file length ({total_lines} lines)."
                lines = lines[offset:]

            # Apply limit
            if limit > 0:
                lines = lines[:limit]

            # Format with line numbers
            start_num = offset + 1  # 1-based line numbers
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_num + i
                # Truncate very long lines
                line_text = line.rstrip("\n\r")
                if len(line_text) > 2000:
                    line_text = line_text[:2000] + "... (truncated)"
                numbered_lines.append(f"{line_num:>6}\t{line_text}")

            header = f"File: {file_path} ({total_lines} lines total)"
            if offset > 0 or limit > 0:
                shown = len(numbered_lines)
                header += f" | Showing lines {start_num}-{start_num + shown - 1}"

            return header + "\n" + "\n".join(numbered_lines)

        except Exception as e:
            return f"Error reading '{file_path}': {e}"

    # ------------------------------------------------------------------
    # Tool: WriteFile
    # ------------------------------------------------------------------

    def write_file(self, file_path: str, content: str) -> str:
        """Create or overwrite a file with the given content.

        Creates parent directories automatically if they don't exist.
        Use this for creating new files or completely rewriting existing ones.
        For partial modifications, use edit_file instead.

        Args:
            file_path: Path to the file (absolute or relative to working directory).
            content: The full content to write to the file.

        Returns:
            str: Success or error message.
        """
        resolved = self._resolve_path(file_path)

        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Successfully wrote {line_count} lines to '{file_path}'."
        except Exception as e:
            return f"Error writing '{file_path}': {e}"

    # ------------------------------------------------------------------
    # Tool: EditFile
    # ------------------------------------------------------------------

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by performing exact string replacement.

        Finds old_string in the file and replaces it with new_string.
        By default, old_string must appear exactly once in the file (for safety).
        Set replace_all=True to replace all occurrences.

        Args:
            file_path: Path to the file (absolute or relative to working directory).
            old_string: The exact text to find and replace. Must match exactly
                        including whitespace and indentation.
            new_string: The replacement text. Must be different from old_string.
            replace_all: If True, replace all occurrences. Default is False
                         (requires unique match).

        Returns:
            str: Success message with replacement count, or error message.
        """
        resolved = self._resolve_path(file_path)

        if not os.path.exists(resolved):
            return f"Error: File '{file_path}' does not exist."

        if old_string == new_string:
            return "Error: old_string and new_string are identical."

        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_string)

            if count == 0:
                # Provide helpful context for debugging
                lines = content.split("\n")
                # Try to find similar lines
                search_lines = old_string.strip().split("\n")
                first_search = search_lines[0].strip() if search_lines else ""
                similar = []
                if first_search:
                    for i, line in enumerate(lines):
                        if first_search in line.strip():
                            similar.append(f"  Line {i + 1}: {line.rstrip()}")
                            if len(similar) >= 3:
                                break

                msg = f"Error: old_string not found in '{file_path}'."
                if similar:
                    msg += "\n\nSimilar lines found:\n" + "\n".join(similar)
                    msg += "\n\nCheck whitespace and indentation match exactly."
                return msg

            if not replace_all and count > 1:
                return (
                    f"Error: old_string appears {count} times in '{file_path}'. "
                    f"Provide more surrounding context to make the match unique, "
                    f"or set replace_all=True to replace all occurrences."
                )

            new_content = content.replace(old_string, new_string)

            with open(resolved, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Generate unified diff for the change
            msg = f"Successfully replaced {count} occurrence(s) in '{file_path}'."
            old_lines = content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=3,
            ))
            if diff:
                # Cap diff output at 100 lines
                max_diff_lines = 100
                diff_text = "".join(diff[:max_diff_lines])
                if len(diff) > max_diff_lines:
                    diff_text += f"\n... ({len(diff) - max_diff_lines} more diff lines truncated)"
                msg += "\n\n" + diff_text

            return msg

        except Exception as e:
            return f"Error editing '{file_path}': {e}"

    # ------------------------------------------------------------------
    # Tool: GlobFiles
    # ------------------------------------------------------------------

    def glob_files(self, pattern: str, path: str = "") -> str:
        """Find files matching a glob pattern.

        Supports recursive patterns like '**/*.py'. Returns file paths
        sorted by modification time (most recent first).

        Args:
            pattern: Glob pattern to match (e.g., '**/*.py', 'src/**/*.ts').
            path: Directory to search in. Empty string means working directory.

        Returns:
            str: Newline-separated list of matching file paths (relative to
                 working directory), or a message if no matches found.
        """
        search_dir = self._resolve_path(path) if path else self._working_directory
        full_pattern = os.path.join(search_dir, pattern)

        try:
            matches = glob.glob(full_pattern, recursive=True)

            # Filter out directories, keep only files
            files = [f for f in matches if os.path.isfile(f)]

            if not files:
                return f"No files found matching '{pattern}'."

            # Sort by modification time, most recent first
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            # Convert to relative paths
            rel_paths = []
            for f in files:
                try:
                    rel = os.path.relpath(f, self._working_directory)
                except ValueError:
                    rel = f
                rel_paths.append(rel)

            header = f"Found {len(rel_paths)} file(s) matching '{pattern}':"
            return header + "\n" + "\n".join(rel_paths)

        except Exception as e:
            return f"Error searching for '{pattern}': {e}"

    # ------------------------------------------------------------------
    # Tool: GrepFiles
    # ------------------------------------------------------------------

    def grep_files(
        self,
        pattern: str,
        path: str = "",
        file_glob: str = "",
        context_lines: int = 0,
        max_results: int = 50,
        case_insensitive: bool = False,
    ) -> str:
        """Search file contents using a regular expression pattern.

        Searches through files for lines matching the pattern. Supports
        filtering by file glob and showing context lines around matches.

        Args:
            pattern: Regular expression pattern to search for.
            path: Directory or file to search in. Empty string means working directory.
            file_glob: Glob pattern to filter files (e.g., '*.py', '*.ts'). Empty string means all files.
            context_lines: Number of lines to show before and after each match. Default 0.
            max_results: Maximum number of matches to return. Default 50.
            case_insensitive: Whether to perform case-insensitive matching. Default False.

        Returns:
            str: Matching lines with file paths and line numbers, or a message
                 if no matches found.
        """
        search_path = self._resolve_path(path) if path else self._working_directory

        try:
            flags = re.IGNORECASE if case_insensitive else 0
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {e}"

        # Collect files to search
        files_to_search = []
        if os.path.isfile(search_path):
            files_to_search = [search_path]
        else:
            if file_glob:
                files_to_search = glob.glob(
                    os.path.join(search_path, "**", file_glob), recursive=True
                )
            else:
                # Search all text files
                for root, _dirs, filenames in os.walk(search_path):
                    # Skip common non-text directories
                    skip_dirs = {
                        ".git", "node_modules", "__pycache__", ".venv",
                        "venv", ".tox", ".mypy_cache", "dist", "build",
                    }
                    _dirs[:] = [d for d in _dirs if d not in skip_dirs]
                    for fname in filenames:
                        files_to_search.append(os.path.join(root, fname))

        files_to_search = [f for f in files_to_search if os.path.isfile(f)]

        results = []
        total_matches = 0

        for fpath in sorted(files_to_search):
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except (OSError, IOError):
                continue

            for i, line in enumerate(lines):
                if compiled.search(line):
                    total_matches += 1
                    if total_matches > max_results:
                        break

                    try:
                        rel_path = os.path.relpath(fpath, self._working_directory)
                    except ValueError:
                        rel_path = fpath

                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        context_block = []
                        for j in range(start, end):
                            marker = ">" if j == i else " "
                            context_block.append(
                                f"  {marker} {j + 1:>6}\t{lines[j].rstrip()}"
                            )
                        results.append(
                            f"{rel_path}:{i + 1}:\n" + "\n".join(context_block)
                        )
                    else:
                        results.append(
                            f"{rel_path}:{i + 1}:\t{line.rstrip()}"
                        )

            if total_matches > max_results:
                break

        if not results:
            scope = f"in '{path}'" if path and path.strip() else "in working directory"
            return f"No matches found for '{pattern}' {scope}."

        header = f"Found {total_matches} match(es) for '{pattern}':"
        if total_matches > max_results:
            header += f" (showing first {max_results})"

        return header + "\n" + "\n".join(results)

    # ------------------------------------------------------------------
    # Tool: ListDirectory
    # ------------------------------------------------------------------

    def list_directory(
        self,
        path: str = "",
        depth: int = 1,
        show_hidden: bool = False,
    ) -> str:
        """List directory contents with file sizes and type markers.

        Shows files and subdirectories in the given path with [DIR] and [FILE]
        markers. Supports nested listing with depth control.

        Args:
            path: Directory to list. Empty string means working directory.
            depth: How many levels deep to recurse. Default is 1 (immediate children only).
            show_hidden: Whether to include hidden files/directories (starting with '.'). Default False.

        Returns:
            str: Formatted directory listing, or error message.
        """
        target = self._resolve_path(path) if path else self._working_directory

        if not os.path.exists(target):
            return f"Error: Path '{path or '.'}' does not exist."

        if not os.path.isdir(target):
            return f"Error: '{path}' is not a directory."

        def _format_size(size: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if size < 1024:
                    return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
                size /= 1024
            return f"{size:.1f}TB"

        def _list_recursive(dir_path: str, current_depth: int, prefix: str = "") -> List[str]:
            lines = []
            try:
                entries = sorted(os.listdir(dir_path))
            except PermissionError:
                lines.append(f"{prefix}[PERMISSION DENIED]")
                return lines

            for entry in entries:
                if not show_hidden and entry.startswith("."):
                    continue

                full_path = os.path.join(dir_path, entry)

                if os.path.isdir(full_path):
                    lines.append(f"{prefix}[DIR]  {entry}/")
                    if current_depth < depth:
                        lines.extend(
                            _list_recursive(full_path, current_depth + 1, prefix + "  ")
                        )
                else:
                    try:
                        size = os.path.getsize(full_path)
                        size_str = _format_size(size)
                    except OSError:
                        size_str = "?"
                    lines.append(f"{prefix}[FILE] {entry}  ({size_str})")

            return lines

        try:
            rel_display = path if path else "."
            listing = _list_recursive(target, 1)
            if not listing:
                return f"Directory '{rel_display}' is empty."
            header = f"Directory: {rel_display} ({len(listing)} entries)"
            return header + "\n" + "\n".join(listing)
        except Exception as e:
            return f"Error listing '{path or '.'}': {e}"

    # ------------------------------------------------------------------
    # Tool: DiffFiles
    # ------------------------------------------------------------------

    def diff_files(
        self,
        file_a: str,
        file_b: str,
        context_lines: int = 3,
    ) -> str:
        """Compare two files and show a unified diff.

        Generates a unified diff between two files, similar to 'diff -u'.
        Useful for understanding changes between file versions.

        Args:
            file_a: Path to the first file (the "original").
            file_b: Path to the second file (the "modified").
            context_lines: Number of context lines around each change. Default 3.

        Returns:
            str: Unified diff output, or a message if files are identical.
        """
        resolved_a = self._resolve_path(file_a)
        resolved_b = self._resolve_path(file_b)

        for label, resolved in [("file_a", resolved_a), ("file_b", resolved_b)]:
            if not os.path.exists(resolved):
                return f"Error: {label} '{file_a if label == 'file_a' else file_b}' does not exist."
            if os.path.isdir(resolved):
                return f"Error: {label} '{file_a if label == 'file_a' else file_b}' is a directory."

        try:
            with open(resolved_a, "r", encoding="utf-8", errors="replace") as f:
                lines_a = f.readlines()
            with open(resolved_b, "r", encoding="utf-8", errors="replace") as f:
                lines_b = f.readlines()

            diff = list(difflib.unified_diff(
                lines_a, lines_b,
                fromfile=file_a,
                tofile=file_b,
                n=context_lines,
            ))

            if not diff:
                return f"Files '{file_a}' and '{file_b}' are identical."

            # Cap output at 200 lines
            max_lines = 200
            diff_text = "".join(diff[:max_lines])
            if len(diff) > max_lines:
                diff_text += f"\n... ({len(diff) - max_lines} more lines truncated)"

            return diff_text

        except Exception as e:
            return f"Error comparing files: {e}"

    # ------------------------------------------------------------------
    # Tool: SubAgent (single-turn)
    # ------------------------------------------------------------------

    @staticmethod
    def create_sub_agent_tool(
        sub_agent_api,
        sub_agent_name: str = "sub_agent",
        sub_agent_tools: Optional[List[FunctionTool]] = None,
        max_context_tokens: int = 128000,
        reserve_tokens: int = 4096,
        strategy: str = "sliding_window",
    ) -> FunctionTool:
        """Create a sub-agent tool that spawns a separate agent for complex tasks.

        The sub-agent gets its own conversation context and can use its own set
        of tools. It runs to completion and returns the final response.

        This is a static factory method - call it once to create the tool,
        then register it alongside the other coding tools.

        Args:
            sub_agent_api: A configured ChatAPIProvider instance for the sub-agent's
                           LLM. Can be a different model/provider than the parent agent.
            sub_agent_name: Name for the tool. Default is 'sub_agent'.
            sub_agent_tools: Optional list of FunctionTools available to the sub-agent.
                             If None, the sub-agent has no tools (text-only).
            max_context_tokens: Max context window for the sub-agent. Default 128000.
            reserve_tokens: Tokens to reserve for the response. Default 4096.
            strategy: Context trimming strategy. Default 'sliding_window'.

        Returns:
            FunctionTool: A tool that can be registered with the parent agent.

        Example:
            sub_api = OpenAIChatAPI(api_key="...", model="gpt-4o-mini")
            sub_tool = CodingTools.create_sub_agent_tool(
                sub_agent_api=sub_api,
                sub_agent_name="research_agent",
                sub_agent_tools=[grep_tool, read_tool],
            )
            coding_tools = CodingTools()
            all_tools = coding_tools.get_tools() + [sub_tool]
        """
        # Import here to avoid circular imports
        from ToolAgents.agent_harness import create_harness

        def sub_agent(
            task: str,
            system_prompt: str = "You are a helpful sub-agent. Complete the given task and return a clear, concise result.",
        ) -> str:
            """Spawn a sub-agent to handle a complex or independent task.

            The sub-agent runs in its own conversation context with its own
            tools. Use this for tasks that benefit from focused attention,
            parallel exploration, or when the current context is too large.

            Args:
                task: A clear description of what the sub-agent should accomplish.
                      Be specific about what information to return.
                system_prompt: System prompt for the sub-agent. Customize to give
                               the sub-agent a specific role or constraints.

            Returns:
                str: The sub-agent's final response.
            """
            try:
                harness = create_harness(
                    provider=sub_agent_api,
                    system_prompt=system_prompt,
                    max_context_tokens=max_context_tokens,
                    reserve_tokens=reserve_tokens,
                    strategy=strategy,
                    streaming=False,
                    tools=sub_agent_tools,
                )

                response = harness.chat(task)
                return response

            except Exception as e:
                return f"Sub-agent error: {e}"

        tool = FunctionTool(sub_agent)
        tool.model.__name__ = sub_agent_name
        return tool

    # ------------------------------------------------------------------
    # Tool: Multi-Turn SubAgent
    # ------------------------------------------------------------------

    @staticmethod
    def create_multi_turn_sub_agent_tools(
        sub_agent_api,
        sub_agent_tools: Optional[List[FunctionTool]] = None,
        max_context_tokens: int = 128000,
        reserve_tokens: int = 4096,
        strategy: str = "sliding_window",
    ) -> List[FunctionTool]:
        """Create multi-turn sub-agent tools with session persistence.

        Returns 3 tools for managing persistent sub-agent sessions:
        1. multi_turn_sub_agent - Send a message to a new or existing session
        2. multi_turn_sub_agent_list_sessions - List active sessions
        3. multi_turn_sub_agent_close_session - Close a session

        Sessions persist across calls, allowing multi-turn conversations with
        sub-agents. Each session maintains its own conversation history.

        Args:
            sub_agent_api: A configured ChatAPIProvider instance for the sub-agent.
            sub_agent_tools: Optional tools available to the sub-agent.
            max_context_tokens: Max context window for sub-agents. Default 128000.
            reserve_tokens: Tokens to reserve for the response. Default 4096.
            strategy: Context trimming strategy. Default 'sliding_window'.

        Returns:
            List[FunctionTool]: Three tools for multi-turn sub-agent management.

        Example:
            mt_tools = CodingTools.create_multi_turn_sub_agent_tools(
                sub_agent_api=sub_api,
                sub_agent_tools=sub_coding_tools.get_tools(),
            )
            all_tools = coding_tools.get_tools() + mt_tools
        """
        from ToolAgents.agent_harness import create_harness

        # Closure-scoped session storage
        sessions: dict = {}  # session_id -> AgentHarness

        def multi_turn_sub_agent(
            task: str,
            session_id: str = "",
            system_prompt: str = "You are a helpful sub-agent. Complete the given task and return a clear, concise result.",
        ) -> str:
            """Send a message to a new or existing sub-agent session.

            Creates a new persistent session if session_id is empty, or
            continues an existing session if a valid session_id is provided.
            The sub-agent maintains conversation history across calls.

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
                    # Continue existing session
                    harness = sessions[session_id]
                    response = harness.chat(task)
                    return response
                elif session_id and session_id not in sessions:
                    return f"Error: Session '{session_id}' not found. Use empty session_id to create a new session."
                else:
                    # Create new session
                    new_id = uuid.uuid4().hex[:8]
                    harness = create_harness(
                        provider=sub_agent_api,
                        system_prompt=system_prompt,
                        max_context_tokens=max_context_tokens,
                        reserve_tokens=reserve_tokens,
                        strategy=strategy,
                        streaming=False,
                        tools=sub_agent_tools,
                    )
                    sessions[new_id] = harness
                    response = harness.chat(task)
                    return f"[New Session: {new_id}]\n{response}"

            except Exception as e:
                return f"Multi-turn sub-agent error: {e}"

        def multi_turn_sub_agent_list_sessions() -> str:
            """List all active sub-agent sessions with their turn counts.

            Returns:
                str: Formatted list of active sessions, or a message if none exist.
            """
            if not sessions:
                return "No active sub-agent sessions."

            lines = [f"Active sessions ({len(sessions)}):"]
            for sid, harness in sessions.items():
                lines.append(f"  {sid}: {harness.turn_count} turn(s)")
            return "\n".join(lines)

        def multi_turn_sub_agent_close_session(session_id: str) -> str:
            """Close and free a sub-agent session.

            Args:
                session_id: The session ID to close.

            Returns:
                str: Confirmation message, or error if session not found.
            """
            if session_id in sessions:
                turns = sessions[session_id].turn_count
                del sessions[session_id]
                return f"Session '{session_id}' closed ({turns} turns completed)."
            return f"Error: Session '{session_id}' not found."

        # Build tools
        tool_send = FunctionTool(multi_turn_sub_agent)
        tool_list = FunctionTool(multi_turn_sub_agent_list_sessions)
        tool_close = FunctionTool(multi_turn_sub_agent_close_session)

        return [tool_send, tool_list, tool_close]

    # ------------------------------------------------------------------
    # Tool collection
    # ------------------------------------------------------------------

    def get_tools(self) -> List[FunctionTool]:
        """Get all coding tools as FunctionTool instances.

        Returns:
            List[FunctionTool]: The complete set of coding tools:
                - bash
                - read_file
                - write_file
                - edit_file
                - glob_files
                - grep_files
                - list_directory
                - diff_files
        """
        return [
            FunctionTool(self.bash),
            FunctionTool(self.read_file),
            FunctionTool(self.write_file),
            FunctionTool(self.edit_file),
            FunctionTool(self.glob_files),
            FunctionTool(self.grep_files),
            FunctionTool(self.list_directory),
            FunctionTool(self.diff_files),
        ]

    def get_tools_documentation(self) -> str:
        """Get formatted documentation for all tools.

        Returns:
            str: Human-readable documentation of all tools and their parameters.
        """
        registry = ToolRegistry()
        registry.add_tools(self.get_tools())
        return registry.get_tools_documentation()

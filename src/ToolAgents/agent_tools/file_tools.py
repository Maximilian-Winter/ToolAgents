import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import re

from ToolAgents import FunctionTool


class FilesystemTools:
    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize the FilesystemTools with a working directory.

        Args:
            working_directory (Optional[str]): The directory where filesystem operations will be performed.
                                              If None, the current working directory is used.
        """
        self.working_directory = working_directory or os.getcwd()
        os.makedirs(self.working_directory, exist_ok=True)

    def _resolve_path(self, path: str) -> str:
        """
        Resolve a path relative to the working directory.

        Args:
            path (str): The path to resolve.

        Returns:
            str: The absolute path.
        """
        if os.path.isabs(path):
            return path
        return os.path.join(self.working_directory, path)

    def glob_files(self, pattern: str) -> str:
        """
        List files matching a pattern in the working directory.

        Args:
            pattern (str): The glob pattern to match files against.

        Returns:
            str: A formatted string with the list of files.
        """
        resolved_pattern = self._resolve_path(pattern)
        files = glob.glob(resolved_pattern, recursive=True)

        if not files:
            return f"No files found matching pattern '{pattern}'."

        result = [f"Files matching pattern '{pattern}':"]
        for file in sorted(files):
            rel_path = os.path.relpath(file, self.working_directory)
            file_type = "Directory" if os.path.isdir(file) else "File"
            if "node_modules" not in rel_path:
                if file_type == "File":
                    size = os.path.getsize(file)
                    size_str = self._format_file_size(size)
                    result.append(f"- {rel_path} ({file_type}, {size_str})")
                else:
                    result.append(f"- {rel_path} ({file_type})")

        return "\n".join(result)

    def _format_file_size(self, size_in_bytes: int) -> str:
        """
        Format file size in a human-readable format.

        Args:
            size_in_bytes (int): File size in bytes.

        Returns:
            str: Formatted file size string (e.g., "4.2 KB").
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024.0 or unit == 'TB':
                break
            size_in_bytes /= 1024.0
        return f"{size_in_bytes:.1f} {unit}"



    def create_file(self, file_path: str, content: str = "") -> str:
        """
        Create a new file with the specified content.

        Args:
            file_path (str): Path to the file to create.
            content (str): Content to write to the file.

        Returns:
            str: A message indicating the result of the operation.
        """
        full_path = self._resolve_path(file_path)

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File '{file_path}' created successfully."
        except Exception as e:
            return f"Failed to create file '{file_path}': {str(e)}"

    def read_file(self, file_path: str) -> str:
        """
        Read and return the content of a file.

        Args:
            file_path (str): Path to the file to read.

        Returns:
            str: The content of the file or an error message.
        """
        full_path = self._resolve_path(file_path)

        try:
            if not os.path.exists(full_path):
                return f"Error: File '{file_path}' does not exist."

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return f"Content of file '{file_path}':\n\n{content}"
        except Exception as e:
            return f"Failed to read file '{file_path}': {str(e)}"

    def edit_file(self, file_path: str, old_text: str, new_text: str, append: bool) -> str:
        """
        Edit a file by replacing text or setting its entire content.

        Args:
            file_path (str): Path to the file to edit.
            old_text (str): Text to replace in the file.
            new_text (str): Text to replace the old_text with.
            append (bool): Whether to append content to the file instead of replacing.

        Returns:
            str: A message indicating the result of the operation.
        """
        full_path = self._resolve_path(file_path)

        if not os.path.exists(full_path):
            return f"Error: File '{file_path}' does not exist."

        try:
            if append:
                with open(full_path, 'a', encoding='utf-8') as f:
                    f.write(new_text)
                return f"Text appended successfully in '{file_path}'."
            else:
                with open(full_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                if old_text not in file_content:
                    return f"Error: Text '{old_text}' not found in file '{file_path}'."

                new_content = file_content.replace(old_text, new_text)

                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                return f"Text replaced successfully in '{file_path}'."
        except Exception as e:
            return f"Failed to edit file '{file_path}': {str(e)}"

    def delete_file(self, file_path: str) -> str:
        """
        Delete a file or directory.

        Args:
            file_path (str): Path to the file or directory to delete.

        Returns:
            str: A message indicating the result of the operation.
        """
        full_path = self._resolve_path(file_path)

        if not os.path.exists(full_path):
            return f"Error: '{file_path}' does not exist."

        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                return f"Directory '{file_path}' and all its contents deleted successfully."
            else:
                os.remove(full_path)
                return f"File '{file_path}' deleted successfully."
        except Exception as e:
            return f"Failed to delete '{file_path}': {str(e)}"

    def create_directories(self, dir_paths: list[str]) -> str:
        """
        Create new directories.

        Args:
            dir_paths (list[str]): Path of the directories to create.

        Returns:
            str: A message indicating the result of the operation.
        """
        results = []
        for dir_path in dir_paths:
            full_path = self._resolve_path(dir_path)

            try:
                os.makedirs(full_path, exist_ok=True)
                results.append(f"Directory '{dir_path}' created successfully.")
            except Exception as e:
                results.append(f"Failed to create directory '{dir_path}': {str(e)}")
        return '\n'.join(results)

    def move_file(self, source_path: str, dest_path: str) -> str:
        """
        Move a file or directory from source to destination.

        Args:
            source_path (str): Path to the source file or directory.
            dest_path (str): Destination path.

        Returns:
            str: A message indicating the result of the operation.
        """
        full_source = self._resolve_path(source_path)
        full_dest = self._resolve_path(dest_path)

        if not os.path.exists(full_source):
            return f"Error: Source '{source_path}' does not exist."

        try:
            # Create parent directories of destination if they don't exist
            os.makedirs(os.path.dirname(full_dest), exist_ok=True)

            shutil.move(full_source, full_dest)
            item_type = "Directory" if os.path.isdir(full_dest) else "File"
            return f"{item_type} moved from '{source_path}' to '{dest_path}' successfully."
        except Exception as e:
            return f"Failed to move from '{source_path}' to '{dest_path}': {str(e)}"

    def copy_file(self, source_path: str, dest_path: str) -> str:
        """
        Copy a file or directory from source to destination.

        Args:
            source_path (str): Path to the source file or directory.
            dest_path (str): Destination path.

        Returns:
            str: A message indicating the result of the operation.
        """
        full_source = self._resolve_path(source_path)
        full_dest = self._resolve_path(dest_path)

        if not os.path.exists(full_source):
            return f"Error: Source '{source_path}' does not exist."

        try:
            # Create parent directories of destination if they don't exist
            os.makedirs(os.path.dirname(full_dest), exist_ok=True)

            if os.path.isdir(full_source):
                shutil.copytree(full_source, full_dest)
                return f"Directory copied from '{source_path}' to '{dest_path}' successfully."
            else:
                shutil.copy2(full_source, full_dest)
                return f"File copied from '{source_path}' to '{dest_path}' successfully."
        except Exception as e:
            return f"Failed to copy from '{source_path}' to '{dest_path}': {str(e)}"

    def get_file_info(self, file_path: str) -> str:
        """
        Get detailed information about a file or directory.

        Args:
            file_path (str): Path to the file or directory.

        Returns:
            str: Formatted string with file information.
        """
        full_path = self._resolve_path(file_path)

        if not os.path.exists(full_path):
            return f"Error: '{file_path}' does not exist."

        try:
            stats = os.stat(full_path)
            path_obj = Path(full_path)

            info = [
                f"Information for: {file_path}",
                f"Type: {'Directory' if os.path.isdir(full_path) else 'File'}",
                f"Absolute path: {os.path.abspath(full_path)}",
                f"Size: {self._format_file_size(stats.st_size)}",
                f"Created: {self._format_timestamp(stats.st_ctime)}",
                f"Last modified: {self._format_timestamp(stats.st_mtime)}",
                f"Last accessed: {self._format_timestamp(stats.st_atime)}",
                f"Owner: {path_obj.owner()}",
                f"Group: {path_obj.group()}",
                f"Permissions: {oct(stats.st_mode)[-3:]}"
            ]

            if os.path.isdir(full_path):
                items = os.listdir(full_path)
                info.append(f"Contents: {len(items)} items")

            return "\n".join(info)
        except Exception as e:
            return f"Failed to get info for '{file_path}': {str(e)}"

    def _format_timestamp(self, timestamp: float) -> str:
        """
        Format a timestamp in a human-readable format.

        Args:
            timestamp (float): The timestamp to format.

        Returns:
            str: Formatted date string.
        """
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


    def get_working_directory(self) -> str:
        """
        Get the current working directory.

        Returns:
            str: The current working directory.
        """
        return f"{self.working_directory}"

    def set_working_directory(self, directory: str) -> str:
        """
        Set the working directory.

        Args:
            directory (str): The new working directory.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
            if os.path.isabs(directory):
                new_dir = directory
            else:
                new_dir = os.path.join(self.working_directory, directory)

            if not os.path.exists(new_dir):
                os.makedirs(new_dir, exist_ok=True)
                result = f"Created and set working directory to: {new_dir}"
            else:
                result = f"Working directory set to: {new_dir}"

            self.working_directory = new_dir
            return result
        except Exception as e:
            return f"Failed to set working directory: {str(e)}"

    def get_tools(self) -> List[FunctionTool]:
        """
        Get a list of FunctionTools for all the FilesystemTools methods.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects for all FilesystemTools methods.
        """
        return [
            FunctionTool(self.glob_files),
            FunctionTool(self.create_file),
            FunctionTool(self.read_file),
            FunctionTool(self.edit_file),
            FunctionTool(self.create_directories),
            FunctionTool(self.get_working_directory),
            FunctionTool(self.set_working_directory)
        ]

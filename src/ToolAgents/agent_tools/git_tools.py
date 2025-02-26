import os
import subprocess
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path

from ToolAgents import FunctionTool


class GitTools:
    def __init__(self, get_working_directory: Callable[[], str]):
        """
        Initialize the GitTools with a working directory.

        Args:
            get_working_directory (Callable[[], str]): A function that returns the working directory.
        """
        self.get_working_directory = get_working_directory


    def _run_git_command(self, command: List[str]) -> Tuple[str, str, int]:
        """
        Run a git command and return the result.

        Args:
            command (List[str]): The git command to run as a list of strings.

        Returns:
            Tuple[str, str, int]: A tuple containing (stdout, stderr, return_code).
        """
        try:
            process = subprocess.Popen(
                ['git'] + command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.get_working_directory(),
                text=True
            )
            stdout, stderr = process.communicate()
            return stdout.strip(), stderr.strip(), process.returncode
        except Exception as e:
            return "", str(e), 1

    def git_init(self, bare: bool = False) -> str:
        """
        Initialize a new git repository in the working directory.

        Args:
            bare (bool): If True, create a bare repository.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['init']
        if bare:
            command.append('--bare')

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Repository initialized successfully in {self.working_directory}"
        else:
            return f"Failed to initialize repository: {stderr}"

    def git_clone(self, repo_url: str, destination: Optional[str] = None) -> str:
        """
        Clone a git repository from a URL.

        Args:
            repo_url (str): The URL of the repository to clone.
            destination (Optional[str]): The directory to clone into.
                                        If None, git will create a directory based on the repository name.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['clone', repo_url]
        if destination:
            command.append(destination)

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Repository cloned successfully from {repo_url}"
        else:
            return f"Failed to clone repository: {stderr}"

    def git_add(self, files: List[str] = None) -> str:
        """
        Add files to the git staging area.

        Args:
            files (List[str]): List of file paths to add. If None, all changes will be added.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['add']
        if files:
            command.extend(files)
        else:
            command.append('.')

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return "Files added to staging area successfully"
        else:
            return f"Failed to add files: {stderr}"

    def git_commit(self, message: str, author: Optional[str] = None) -> str:
        """
        Commit staged changes to the repository.

        Args:
            message (str): The commit message.
            author (Optional[str]): The author of the commit in the format "Name <email>".
                                   If None, the git global config will be used.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['commit', '-m', message]
        if author:
            command.extend(['--author', author])

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Changes committed successfully: {stdout}"
        else:
            return f"Failed to commit changes: {stderr}"

    def git_push(self, remote: str = 'origin', branch: str = 'main', force: bool = False) -> str:
        """
        Push commits to a remote repository.

        Args:
            remote (str): The name of the remote repository.
            branch (str): The branch to push to.
            force (bool): If True, force push even if it results in a non-fast-forward update.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['push', remote, branch]
        if force:
            command.append('--force')

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Changes pushed to {remote}/{branch} successfully"
        else:
            return f"Failed to push changes: {stderr}"

    def git_pull(self, remote: str = 'origin', branch: str = 'main') -> str:
        """
        Pull changes from a remote repository.

        Args:
            remote (str): The name of the remote repository.
            branch (str): The branch to pull from.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['pull', remote, branch]

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Changes pulled from {remote}/{branch} successfully"
        else:
            return f"Failed to pull changes: {stderr}"

    def git_checkout(self, branch_or_commit: str, create_branch: bool = False) -> str:
        """
        Checkout a branch or commit.

        Args:
            branch_or_commit (str): The branch or commit to checkout.
            create_branch (bool): If True, create the branch if it doesn't exist.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['checkout']
        if create_branch:
            command.append('-b')
        command.append(branch_or_commit)

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Checked out {branch_or_commit} successfully"
        else:
            return f"Failed to checkout {branch_or_commit}: {stderr}"

    def git_status(self) -> str:
        """
        Get the status of the working directory.

        Returns:
            str: The git status output.
        """
        stdout, stderr, return_code = self._run_git_command(['status'])

        if return_code == 0:
            return stdout
        else:
            return f"Failed to get status: {stderr}"

    def git_log(self, n: int = 10, oneline: bool = False) -> str:
        """
        Get the commit history.

        Args:
            n (int): The number of commits to show.
            oneline (bool): If True, show each commit on a single line.

        Returns:
            str: The git log output.
        """
        command = ['log', f'-n{n}']
        if oneline:
            command.append('--oneline')

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return stdout
        else:
            return f"Failed to get log: {stderr}"

    def git_branch_list(self, all_branches: bool = False) -> str:
        """
        List branches in the repository.

        Args:
            all_branches (bool): If True, show both local and remote branches.

        Returns:
            str: The list of branches.
        """
        command = ['branch']
        if all_branches:
            command.append('-a')

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return stdout
        else:
            return f"Failed to list branches: {stderr}"

    def git_branch_create(self, branch_name: str, start_point: Optional[str] = None) -> str:
        """
        Create a new branch.

        Args:
            branch_name (str): The name of the new branch.
            start_point (Optional[str]): The commit or branch to start from.
                                        If None, the current HEAD will be used.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['branch', branch_name]
        if start_point:
            command.append(start_point)

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Branch '{branch_name}' created successfully"
        else:
            return f"Failed to create branch: {stderr}"

    def git_branch_delete(self, branch_name: str, force: bool = False) -> str:
        """
        Delete a branch.

        Args:
            branch_name (str): The name of the branch to delete.
            force (bool): If True, force delete the branch even if it has unmerged changes.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['branch', '-d' if not force else '-D', branch_name]

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Branch '{branch_name}' deleted successfully"
        else:
            return f"Failed to delete branch: {stderr}"

    def git_diff(self, file_path: Optional[str] = None) -> str:
        """
        Show changes between commits, commit and working tree, etc.

        Args:
            file_path (Optional[str]): The path to the file to show differences for.
                                     If None, show differences for all changed files.

        Returns:
            str: The git diff output.
        """
        command = ['diff']
        if file_path:
            command.append(file_path)

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return stdout
        else:
            return f"Failed to get diff: {stderr}"

    def git_reset(self, commit: str = 'HEAD', mode: str = 'mixed', file_path: Optional[str] = None) -> str:
        """
        Reset current HEAD to the specified state.

        Args:
            commit (str): The commit to reset to.
            mode (str): The reset mode ('soft', 'mixed', or 'hard').
            file_path (Optional[str]): The path to the file to reset.
                                     If None, reset the entire working directory.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['reset']

        if mode.lower() == 'soft':
            command.append('--soft')
        elif mode.lower() == 'hard':
            command.append('--hard')
        # 'mixed' is the default

        command.append(commit)

        if file_path:
            command.append('--')
            command.append(file_path)

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Reset to {commit} with mode '{mode}' successfully"
        else:
            return f"Failed to reset: {stderr}"

    def git_stash(self, action: str = 'push', stash_name: Optional[str] = None) -> str:
        """
        Stash changes in the working directory.

        Args:
            action (str): The stash action ('push', 'pop', 'apply', 'list', 'drop', 'clear').
            stash_name (Optional[str]): The name of the stash (only used for 'push' action).

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['stash', action]

        if action == 'push' and stash_name:
            command.extend(['save', stash_name])

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            if stdout:
                return stdout
            else:
                return f"Stash {action} executed successfully"
        else:
            return f"Failed to execute stash {action}: {stderr}"

    def git_remote_list(self, verbose: bool = False) -> str:
        """
        List remote repositories.

        Args:
            verbose (bool): If True, show URLs and additional info.

        Returns:
            str: The list of remote repositories.
        """
        command = ['remote']
        if verbose:
            command.append('-v')

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return stdout
        else:
            return f"Failed to list remotes: {stderr}"

    def git_remote_add(self, name: str, url: str) -> str:
        """
        Add a remote repository.

        Args:
            name (str): The name of the remote repository.
            url (str): The URL of the remote repository.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['remote', 'add', name, url]

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Remote '{name}' added successfully"
        else:
            return f"Failed to add remote: {stderr}"

    def git_remote_remove(self, name: str) -> str:
        """
        Remove a remote repository.

        Args:
            name (str): The name of the remote repository to remove.

        Returns:
            str: A message indicating the result of the operation.
        """
        command = ['remote', 'remove', name]

        stdout, stderr, return_code = self._run_git_command(command)

        if return_code == 0:
            return f"Remote '{name}' removed successfully"
        else:
            return f"Failed to remove remote: {stderr}"

    def get_tools(self) -> List[FunctionTool]:
        """
        Get a list of FunctionTools for all the GitTools methods.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects for all GitTools methods.
        """
        return [
            FunctionTool(self.git_init),
            FunctionTool(self.git_clone),
            FunctionTool(self.git_add),
            FunctionTool(self.git_commit),
            FunctionTool(self.git_push),
            FunctionTool(self.git_pull),
            FunctionTool(self.git_checkout),
            FunctionTool(self.git_status),
            FunctionTool(self.git_log),
            FunctionTool(self.git_branch_list),
            FunctionTool(self.git_branch_create),
            FunctionTool(self.git_diff)
        ]
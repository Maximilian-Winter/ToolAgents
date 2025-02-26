import requests
import json
import os
import base64
from typing import Dict, List, Optional, Union, Any, Tuple

from ToolAgents import FunctionTool


class GitHubTools:
    """
    A class to interact with GitHub API for common GitHub operations.
    This class complements git command-line tools or libraries by providing
    GitHub-specific functionality not available in standard git.
    """

    def __init__(self, owner: str, repo: str, token: Optional[str] = None):
        """
        Initialize GitHubTools with GitHub API token and repository information.

        Args:
            token: GitHub personal access token. If None, tries to get from GITHUB_TOKEN env var.
            owner: Repository owner (username or organization) - required
            repo: Repository name - required
        """
        self.token = token or os.environ.get('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token is required. Provide as parameter or set GITHUB_TOKEN env var.")

        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        Make a request to the GitHub API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint (without base URL)
            data: Data to send with the request

        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=data
        )

        if response.status_code >= 400:
            try:
                error_message = response.json()
            except:
                error_message = response.text
            raise Exception(f"GitHub API error ({response.status_code}): {error_message}")

        if response.text:
            return response.json()
        return {}

    def _format_response(self, response: Union[Dict, List]) -> str:
        """
        Format API response as a readable string.

        Args:
            response: API response (dictionary or list)

        Returns:
            Formatted string representation
        """
        if isinstance(response, list):
            # Format list of items
            if not response:
                return "No items found."

            formatted = []
            for i, item in enumerate(response):
                item_str = f"Item {i + 1}:"
                for key, value in item.items():
                    if key in ['name', 'full_name', 'html_url', 'title', 'state', 'created_at', 'updated_at', 'status']:
                        item_str += f"\n  {key}: {value}"
                formatted.append(item_str)
            return "\n\n".join(formatted)
        else:
            # Format dictionary
            formatted = []
            for key, value in response.items():
                if isinstance(value, (dict, list)):
                    formatted.append(f"{key}: (complex data)")
                else:
                    formatted.append(f"{key}: {value}")
            return "\n".join(formatted)

    def _ensure_repo_info(self):
        """
        Ensure that owner and repo are set.
        """
        if not self.owner or not self.repo:
            raise ValueError(
                "Repository owner and name must be set either at initialization or when calling this method.")

    def github_set_owner_and_repo(self, owner: str, repo: str):
        """
        Change the current owner and repo settings for GitHub API.

        Args:
            owner: GitHub username.
            repo: GitHub repo name.

        Returns:
            Formatted string listing repositories
        """
        self.owner = owner
        self.repo = repo
        self._ensure_repo_info()

    # Repository operations
    def github_list_repositories(self, username: Optional[str] = None) -> str:
        """
        List repositories for a user or the authenticated user.

        Args:
            username: GitHub username. If None, lists authenticated user's repositories.

        Returns:
            Formatted string listing repositories
        """
        if username:
            endpoint = f"users/{username}/repos"
        else:
            endpoint = "user/repos"

        response = self._request("GET", endpoint)

        # Create a nicely formatted string output
        repos_info = []
        for repo in response:
            repo_info = [
                f"Name: {repo['name']}",
                f"Description: {repo.get('description', 'No description')}",
                f"URL: {repo['html_url']}",
                f"Private: {repo['private']}",
                f"Stars: {repo['stargazers_count']}",
                f"Forks: {repo['forks_count']}",
                "---"
            ]
            repos_info.append("\n".join(repo_info))

        if not repos_info:
            return "No repositories found."

        return "\n".join(repos_info)

    def github_create_repository(self, name: str, description: str = "", private: bool = False) -> str:
        """
        Create a new repository.

        Args:
            name: Repository name
            description: Repository description
            private: Whether the repository should be private

        Returns:
            Formatted string with created repository details
        """
        data = {
            "name": name,
            "description": description,
            "private": private
        }

        response = self._request("POST", "user/repos", data)

        # Format the response
        repo_info = [
            f"Repository created successfully!",
            f"Name: {response['name']}",
            f"URL: {response['html_url']}",
            f"Description: {response.get('description', 'No description')}",
            f"Private: {response['private']}"
        ]

        return "\n".join(repo_info)

    def github_delete_repository(self) -> str:
        """
        Delete a repository.

        Returns:
            Success message
        """
        self._ensure_repo_info()
        self._request("DELETE", f"repos/{self.owner}/{self.repo}")
        return f"Repository {self.owner}/{self.repo} has been deleted."

    # Branch operations
    def github_list_branches(self) -> str:
        """
        List branches in a repository.

        Returns:
            Formatted string listing branches
        """
        self._ensure_repo_info()
        response = self._request("GET", f"repos/{self.owner}/{self.repo}/branches")

        # Format the response
        if not response:
            return "No branches found."

        branches_info = []
        for branch in response:
            branch_info = [
                f"Name: {branch['name']}",
                f"SHA: {branch['commit']['sha']}",
                "---"
            ]
            branches_info.append("\n".join(branch_info))

        return "\n".join(branches_info)

    def github_create_branch(self, branch_name: str, from_branch: str = "main") -> str:
        """
        Create a new branch in a repository.

        Args:
            branch_name: Name of the new branch
            from_branch: Base branch name

        Returns:
            Formatted string with created branch details
        """
        self._ensure_repo_info()

        # First get the SHA of the latest commit on the base branch
        base_branch = self._request("GET", f"repos/{self.owner}/{self.repo}/branches/{from_branch}")
        sha = base_branch["commit"]["sha"]

        # Create the new reference
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": sha
        }

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/git/refs", data)

        # Format the response
        return f"Branch '{branch_name}' created successfully from '{from_branch}' (SHA: {sha[:7]})"

    def github_delete_branch(self, branch_name: str) -> str:
        """
        Delete a branch from a repository.

        Args:
            branch_name: Branch name to delete

        Returns:
            Success message
        """
        self._ensure_repo_info()
        self._request("DELETE", f"repos/{self.owner}/{self.repo}/git/refs/heads/{branch_name}")
        return f"Branch '{branch_name}' has been deleted from repository {self.owner}/{self.repo}."

    # File operations
    def github_get_file_content(self, path: str, ref: str = "main") -> str:
        """
        Get file content from a repository.

        Args:
            path: File path in the repository
            ref: Branch name or commit SHA

        Returns:
            File content as string
        """
        self._ensure_repo_info()
        response = self._request("GET", f"repos/{self.owner}/{self.repo}/contents/{path}?ref={ref}")

        if response.get("type") != "file":
            raise ValueError(f"Path does not point to a file: {path}")

        content = base64.b64decode(response["content"]).decode("utf-8")
        return content

    def github_create_or_update_file(self, path: str, content: str, message: str,
                              branch: str = "main", sha: Optional[str] = None) -> str:
        """
        Create or update a file in a repository.

        Args:
            path: File path in the repository
            content: New file content
            message: Commit message
            branch: Branch name
            sha: File SHA (required for updating existing files)

        Returns:
            Success message
        """
        self._ensure_repo_info()

        if sha is None and "/" in path:
            # Try to get the file's SHA if updating
            try:
                response = self._request("GET", f"repos/{self.owner}/{self.repo}/contents/{path}?ref={branch}")
                if isinstance(response, dict) and "sha" in response:
                    sha = response["sha"]
            except:
                # File doesn't exist yet, which is fine for creation
                pass

        data = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch
        }

        if sha:
            data["sha"] = sha
            action = "updated"
        else:
            action = "created"

        response = self._request("PUT", f"repos/{self.owner}/{self.repo}/contents/{path}", data)
        return f"File '{path}' has been {action} in repository {self.owner}/{self.repo}."

    def github_delete_file(self, path: str, message: str, sha: str, branch: str = "main") -> str:
        """
        Delete a file from a repository.

        Args:
            path: File path in the repository
            message: Commit message
            sha: File SHA
            branch: Branch name

        Returns:
            Success message
        """
        self._ensure_repo_info()

        data = {
            "message": message,
            "sha": sha,
            "branch": branch
        }

        self._request("DELETE", f"repos/{self.owner}/{self.repo}/contents/{path}", data)
        return f"File '{path}' has been deleted from repository {self.owner}/{self.repo}."

    # Issues and Pull Requests
    def github_list_issues(self, state: str = "open") -> str:
        """
        List issues in a repository.

        Args:
            state: Issue state (open, closed, all)

        Returns:
            Formatted string listing issues
        """
        self._ensure_repo_info()
        response = self._request("GET", f"repos/{self.owner}/{self.repo}/issues?state={state}")

        if not response:
            return f"No {state} issues found."

        issues_info = []
        for issue in response:
            # Skip pull requests which are also returned in this endpoint
            if "pull_request" in issue:
                continue

            issue_info = [
                f"#{issue['number']}: {issue['title']}",
                f"State: {issue['state']}",
                f"Created: {issue['created_at']}",
                f"URL: {issue['html_url']}",
                "---"
            ]
            issues_info.append("\n".join(issue_info))

        if not issues_info:
            return f"No {state} issues found."

        return "\n".join(issues_info)

    def github_create_issue(self, title: str, body: str = "",
                     labels: List[str] = None, assignees: List[str] = None) -> str:
        """
        Create a new issue in a repository.

        Args:
            title: Issue title
            body: Issue body
            labels: List of label names
            assignees: List of usernames to assign

        Returns:
            Formatted string with created issue details
        """
        self._ensure_repo_info()

        data = {
            "title": title,
            "body": body
        }

        if labels:
            data["labels"] = labels

        if assignees:
            data["assignees"] = assignees

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/issues", data)

        return f"Issue created successfully!\nTitle: {response['title']}\nNumber: #{response['number']}\nURL: {response['html_url']}"

    def github_get_issue(self, issue_number: int) -> str:
        """
        Get details of a specific issue in a repository.

        Args:
            issue_number: The issue number to retrieve

        Returns:
            Formatted string with issue details
        """
        self._ensure_repo_info()
        response = self._request("GET", f"repos/{self.owner}/{self.repo}/issues/{issue_number}")

        # Check if the response is actually a pull request
        is_pr = "pull_request" in response

        # Format the response
        issue_info = [
            f"{'Pull Request' if is_pr else 'Issue'} #{response['number']}: {response['title']}",
            f"State: {response['state']}",
            f"Created by: {response['user']['login']} on {response['created_at']}",
            f"URL: {response['html_url']}"
        ]

        # Add assignees if any
        if response.get('assignees'):
            assignee_names = [assignee['login'] for assignee in response['assignees']]
            issue_info.append(f"Assignees: {', '.join(assignee_names)}")

        # Add labels if any
        if response.get('labels'):
            label_names = [label['name'] for label in response['labels']]
            issue_info.append(f"Labels: {', '.join(label_names)}")

        # Add milestone if any
        if response.get('milestone'):
            issue_info.append(f"Milestone: {response['milestone']['title']}")

        # Add body
        issue_info.append("\nDescription:")
        issue_info.append(response.get('body', 'No description provided.'))

        # Add comments count
        issue_info.append(f"\nComments: {response['comments']}")

        return "\n".join(issue_info)


    def github_create_pull_request(self, title: str, head: str, base: str,
                            body: str, draft: bool) -> str:
        """
        Create a new pull request.

        Args:
            title: Pull request title
            head: Branch containing changes
            base: Branch to merge into
            body: Pull request description
            draft: Whether the PR should be a draft

        Returns:
            Formatted string with created pull request details
        """
        self._ensure_repo_info()

        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
            "draft": draft
        }

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/pulls", data)

        return (f"Pull request created successfully!\n"
                f"Title: {response['title']}\n"
                f"Number: #{response['number']}\n"
                f"URL: {response['html_url']}")


    def github_merge_pull_request(self, pr_number: int, commit_message: Optional[str] = None) -> str:
        """
        Merge a pull request.

        Args:
            pr_number: Pull request number
            commit_message: Custom merge commit message

        Returns:
            Success message
        """
        self._ensure_repo_info()

        data = {}
        if commit_message:
            data["commit_message"] = commit_message

        self._request("PUT", f"repos/{self.owner}/{self.repo}/pulls/{pr_number}/merge", data)
        return f"Pull request #{pr_number} has been merged."

    def github_list_pull_requests(self, state: str = "open", sort: str = "created", direction: str = "desc",
                           limit: Optional[int] = None) -> str:
        """
        List pull requests for a repository.

        Args:
            state: State of the pull requests ("open", "closed", or "all")
            sort: Sort field ("created", "updated", or "popularity")
            direction: Sort direction ("asc" or "desc")
            limit: Maximum number of pull requests to return (None for all)

        Returns:
            List of pull request as string
        """
        self._ensure_repo_info()

        params = {
            "state": state,
            "sort": sort,
            "direction": direction
        }

        response = self._request("GET", f"repos/{self.owner}/{self.repo}/pulls", data=params)
        pull_requests = response

        if limit is not None:
            pull_requests = pull_requests[:limit]
        if not pull_requests:
            return f"No {state} pull requests found in {self.owner}/{self.repo}."

        result = f"Pull requests for {self.owner}/{self.repo} ({state}):\n"

        for pr in pull_requests:
            created_at = pr.get("created_at", "Unknown date")
            result += (
                f"#{pr['number']} - {pr['title']}\n"
                f"  Created by: {pr['user']['login']} on {created_at}\n"
                f"  Status: {pr['state']}\n"
                f"  URL: {pr['html_url']}\n\n"
            )

        return result.strip()

    # Workflows and Actions
    def github_list_workflows(self) -> str:
        """
        List GitHub Actions workflows in a repository.

        Returns:
            Formatted string listing workflows
        """
        self._ensure_repo_info()

        response = self._request("GET", f"repos/{self.owner}/{self.repo}/actions/workflows")
        workflows = response.get("workflows", [])

        if not workflows:
            return "No workflows found."

        workflows_info = []
        for workflow in workflows:
            workflow_info = [
                f"Name: {workflow['name']}",
                f"ID: {workflow['id']}",
                f"State: {workflow['state']}",
                f"Path: {workflow['path']}",
                "---"
            ]
            workflows_info.append("\n".join(workflow_info))

        return "\n".join(workflows_info)

    def github_trigger_workflow(self, workflow_id: str, ref: str = "main",
                         inputs: Optional[Dict] = None) -> str:
        """
        Trigger a GitHub Actions workflow run.

        Args:
            workflow_id: Workflow ID or filename
            ref: Branch or tag name
            inputs: Workflow inputs

        Returns:
            Success message
        """
        self._ensure_repo_info()

        data = {
            "ref": ref
        }

        if inputs:
            data["inputs"] = inputs

        self._request("POST", f"repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/dispatches", data)
        return f"Workflow '{workflow_id}' triggered successfully on ref '{ref}'."

    # Releases
    def github_list_releases(self) -> str:
        """
        List releases for a repository.

        Returns:
            Formatted string listing releases
        """
        self._ensure_repo_info()

        response = self._request("GET", f"repos/{self.owner}/{self.repo}/releases")

        if not response:
            return "No releases found."

        releases_info = []
        for release in response:
            release_info = [
                f"Name: {release['name']}",
                f"Tag: {release['tag_name']}",
                f"Created: {release['created_at']}",
                f"URL: {release['html_url']}",
                f"Draft: {release['draft']}",
                f"Prerelease: {release['prerelease']}",
                "---"
            ]
            releases_info.append("\n".join(release_info))

        return "\n".join(releases_info)

    def github_create_release(self, tag_name: str, name: str, body: str = "",
                       draft: bool = False, prerelease: bool = False) -> str:
        """
        Create a new release.

        Args:
            tag_name: Tag name for the release
            name: Release title
            body: Release description
            draft: Whether it's a draft release
            prerelease: Whether it's a prerelease

        Returns:
            Formatted string with created release details
        """
        self._ensure_repo_info()

        data = {
            "tag_name": tag_name,
            "name": name,
            "body": body,
            "draft": draft,
            "prerelease": prerelease
        }

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/releases", data)

        return (f"Release created successfully!\n"
                f"Name: {response['name']}\n"
                f"Tag: {response['tag_name']}\n"
                f"URL: {response['html_url']}")

    def github_get_pull_request(self, pr_number: int) -> str:
        """
        Get details of a specific pull request in a repository.

        Args:
            pr_number: The pull request number to retrieve

        Returns:
            Formatted string with pull request details
        """
        self._ensure_repo_info()
        response = self._request("GET", f"repos/{self.owner}/{self.repo}/pulls/{pr_number}")

        # Format the basic pull request information
        pr_info = [f"Pull Request #{response['number']}: {response['title']}", f"State: {response['state']}",
                   f"Created by: {response['user']['login']} on {response['created_at']}",
                   f"URL: {response['html_url']}",
                   f"Branches: {response['head']['label']} → {response['base']['label']}"]

        # Add branch information

        # Add mergeable status
        if 'mergeable' in response:
            pr_info.append(f"Mergeable: {response['mergeable']}")
            if 'mergeable_state' in response:
                pr_info.append(f"Mergeable state: {response['mergeable_state']}")

        # Add review status
        if response.get('requested_reviewers'):
            reviewer_names = [reviewer['login'] for reviewer in response['requested_reviewers']]
            pr_info.append(f"Requested reviewers: {', '.join(reviewer_names)}")

        # Add assignees if any
        if response.get('assignees'):
            assignee_names = [assignee['login'] for assignee in response['assignees']]
            pr_info.append(f"Assignees: {', '.join(assignee_names)}")

        # Add labels if any
        if response.get('labels'):
            label_names = [label['name'] for label in response['labels']]
            pr_info.append(f"Labels: {', '.join(label_names)}")

        # Add draft status
        pr_info.append(f"Draft: {response.get('draft', False)}")

        # Add CI status flag if available
        if 'statuses_url' in response:
            pr_info.append(f"Statuses URL: {response['statuses_url']}")

        # Add merge commit details if merged
        if response.get('merged'):
            pr_info.append(f"Merged: Yes (by {response.get('merged_by', {}).get('login', 'Unknown')})")
            pr_info.append(f"Merged at: {response.get('merged_at', 'Unknown')}")
            if 'merge_commit_sha' in response:
                pr_info.append(f"Merge commit: {response['merge_commit_sha']}")
        else:
            pr_info.append("Merged: No")

        # Add body/description
        pr_info.append("\nDescription:")
        pr_info.append(response.get('body', 'No description provided.'))

        # Add commits and changed files counts
        pr_info.append(f"\nCommits: {response['commits']}")
        pr_info.append(f"Changed files: {response['changed_files']}")
        pr_info.append(f"Additions: {response['additions']}")
        pr_info.append(f"Deletions: {response['deletions']}")

        # Add comments count
        pr_info.append(f"Comments: {response['comments']}")

        return "\n".join(pr_info)
    # Teams and Collaborators
    def github_list_collaborators(self) -> str:
        """
        List collaborators for a repository.

        Returns:
            Formatted string listing collaborators
        """
        self._ensure_repo_info()

        response = self._request("GET", f"repos/{self.owner}/{self.repo}/collaborators")

        if not response:
            return "No collaborators found."

        collaborators_info = []
        for collaborator in response:
            collab_info = [
                f"Username: {collaborator['login']}",
                f"Profile: {collaborator['html_url']}",
                f"Permissions: {collaborator.get('permissions', {})}",
                "---"
            ]
            collaborators_info.append("\n".join(collab_info))

        return "\n".join(collaborators_info)

    def github_add_collaborator(self, username: str, permission: str = "push") -> str:
        """
        Add a collaborator to a repository.

        Args:
            username: Username to add as collaborator
            permission: Permission level (pull, push, admin)

        Returns:
            Success message
        """
        self._ensure_repo_info()

        data = {
            "permission": permission
        }

        response = self._request("PUT", f"repos/{self.owner}/{self.repo}/collaborators/{username}", data)

        if "id" in response:
            return f"Invitation sent to {username} to collaborate on {self.owner}/{self.repo}."
        else:
            return f"{username} added as a collaborator to {self.owner}/{self.repo}."

    def github_remove_collaborator(self, username: str) -> str:
        """
        Remove a collaborator from a repository.

        Args:
            username: Username to remove

        Returns:
            Success message
        """
        self._ensure_repo_info()

        self._request("DELETE", f"repos/{self.owner}/{self.repo}/collaborators/{username}")
        return f"{username} has been removed as a collaborator from {self.owner}/{self.repo}."

    def get_tools(self):
        """
        Return a list of FunctionTool instances representing the available GitHub operations.

        Returns:
            List of FunctionTool instances
        """

        return [
            FunctionTool(self.github_set_owner_and_repo),
            FunctionTool(self.github_list_branches),
            FunctionTool(self.github_create_branch),
            FunctionTool(self.github_list_issues),
            FunctionTool(self.github_create_issue),
            FunctionTool(self.github_get_issue),
            FunctionTool(self.github_list_pull_requests),
            FunctionTool(self.github_create_pull_request),
            FunctionTool(self.github_get_pull_request)
        ]
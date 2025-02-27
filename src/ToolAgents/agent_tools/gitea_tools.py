import requests
import json
import os
import base64
from typing import Dict, List, Optional, Union, Any, Tuple

from ToolAgents import FunctionTool

class GiteaTools:
    """
    A class to interact with Gitea API for common Gitea operations.
    This class complements git command-line tools or libraries by providing
    Gitea-specific functionality not available in standard git.
    """

    def __init__(self, base_url: str, owner: str, repo: str, token: Optional[str] = None):
        """
        Initialize GiteaTools with Gitea API token and repository information.

        Args:
            base_url: Base URL of your Gitea instance (e.g., "https://gitea.yourdomain.com/api/v1")
            token: Gitea personal access token. If None, tries to get from GITEA_TOKEN env var.
            owner: Repository owner (username or organization) - required
            repo: Repository name - required
        """
        self.token = token or os.environ.get('GITEA_TOKEN')
        if not self.token:
            raise ValueError("Gitea token is required. Provide as parameter or set GITEA_TOKEN env var.")

        self.owner = owner
        self.repo = repo
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/api/v1'):
            self.base_url = f"{self.base_url}/api/v1"
            
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        Make a request to the Gitea API.

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
            raise Exception(f"Gitea API error ({response.status_code}): {error_message}")

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

    def gitea_set_owner_and_repo(self, owner: str, repo: str):
        """
        Change the current owner and repo settings for Gitea API.

        Args:
            owner: Gitea username.
            repo: Gitea repo name.

        Returns:
            Confirmation message
        """
        self.owner = owner
        self.repo = repo
        self._ensure_repo_info()
        return f"Repository context set to {owner}/{repo}"

    # Repository operations
    def gitea_list_repositories(self, username: Optional[str] = None) -> str:
        """
        List repositories for a user or the authenticated user.

        Args:
            username: Gitea username. If None, lists authenticated user's repositories.

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
                f"Stars: {repo['stars_count']}",
                f"Forks: {repo['forks_count']}",
                "---"
            ]
            repos_info.append("\n".join(repo_info))

        if not repos_info:
            return "No repositories found."

        return "\n".join(repos_info)

    def gitea_create_repository(self, name: str, description: str = "", private: bool = False) -> str:
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

    def gitea_delete_repository(self) -> str:
        """
        Delete a repository.

        Returns:
            Success message
        """
        self._ensure_repo_info()
        self._request("DELETE", f"repos/{self.owner}/{self.repo}")
        return f"Repository {self.owner}/{self.repo} has been deleted."

    # Branch operations
    def gitea_list_branches(self) -> str:
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
                f"SHA: {branch['commit']['id']}",
                "---"
            ]
            branches_info.append("\n".join(branch_info))

        return "\n".join(branches_info)

    def gitea_create_branch(self, branch_name: str, from_branch: str = "main") -> str:
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
        sha = base_branch["commit"]["id"]

        # Create the new reference
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": sha
        }

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/git/refs", data)

        # Format the response
        return f"Branch '{branch_name}' created successfully from '{from_branch}' (SHA: {sha[:7]})"

    def gitea_delete_branch(self, branch_name: str) -> str:
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
    def gitea_get_file_content(self, path: str, ref: str = "main") -> str:
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

        if isinstance(response, list):
            raise ValueError(f"Path points to a directory, not a file: {path}")

        content = base64.b64decode(response["content"]).decode("utf-8")
        return content

    def gitea_create_or_update_file(self, path: str, content: str, message: str,
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

    def gitea_delete_file(self, path: str, message: str, sha: str, branch: str = "main") -> str:
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
    def gitea_list_issues(self, state: str = "open") -> str:
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
            if issue.get("pull_request", None) is not None:
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

    def gitea_create_issue(self, title: str, body: str = "",
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

    def gitea_get_issue(self, issue_number: int) -> str:
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
        is_pr = response.get("pull_request", None) is not None

        # Format the response
        issue_info = [
            f"{'Pull Request' if is_pr else 'Issue'} #{response['number']}: {response['title']}",
            f"State: {response['state']}",
            f"Created by: {response['user']['username']} on {response['created_at']}",
            f"URL: {response['html_url']}"
        ]

        # Add assignees if any
        if response.get('assignees'):
            assignee_names = [assignee['username'] for assignee in response['assignees']]
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

    def gitea_create_pull_request(self, title: str, head: str, base: str,
                            body: str = "", draft: bool = False) -> str:
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
            "body": body
        }
        
        # Gitea might handle draft PRs differently than GitHub
        if draft:
            # Some Gitea versions support draft PRs via title prefix
            data["title"] = f"Draft: {title}"

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/pulls", data)

        return (f"Pull request created successfully!\n"
                f"Title: {response['title']}\n"
                f"Number: #{response['number']}\n"
                f"URL: {response['html_url']}")

    def gitea_merge_pull_request(self, pr_number: int, merge_message: Optional[str] = None) -> str:
        """
        Merge a pull request.

        Args:
            pr_number: Pull request number
            merge_message: Custom merge commit message

        Returns:
            Success message
        """
        self._ensure_repo_info()

        data = {
            "Do": "merge"
        }
        if merge_message:
            data["MergeMessageField"] = merge_message

        self._request("POST", f"repos/{self.owner}/{self.repo}/pulls/{pr_number}/merge", data)
        return f"Pull request #{pr_number} has been merged."

    def gitea_list_pull_requests(self, state: str = "open", sort: str = "created", 
                               direction: str = "desc", limit: Optional[int] = None) -> str:
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
                f"  Created by: {pr['user']['username']} on {created_at}\n"
                f"  Status: {pr['state']}\n"
                f"  URL: {pr['html_url']}\n\n"
            )

        return result.strip()

    def gitea_get_pull_request(self, pr_number: int) -> str:
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
        pr_info = [
            f"Pull Request #{response['number']}: {response['title']}",
            f"State: {response['state']}",
            f"Created by: {response['user']['username']} on {response['created_at']}",
            f"URL: {response['html_url']}",
            f"Branches: {response['head']['ref']} → {response['base']['ref']}"
        ]

        # Add mergeable status if available
        if 'mergeable' in response:
            pr_info.append(f"Mergeable: {response['mergeable']}")

        # Add draft status if supported
        if 'draft' in response:
            pr_info.append(f"Draft: {response['draft']}")
        elif response['title'].lower().startswith('draft:'):
            pr_info.append("Draft: Yes (via title prefix)")

        # Add merged status
        if response.get('merged', False):
            pr_info.append(f"Merged: Yes")
            if 'merged_at' in response:
                pr_info.append(f"Merged at: {response['merged_at']}")
        else:
            pr_info.append("Merged: No")

        # Add body/description
        pr_info.append("\nDescription:")
        pr_info.append(response.get('body', 'No description provided.'))

        return "\n".join(pr_info)

    # Collaborators
    def gitea_list_collaborators(self) -> str:
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
                f"Username: {collaborator['username']}",
                f"Profile: {collaborator['html_url']}",
                f"Permissions: {collaborator.get('permissions', {})}",
                "---"
            ]
            collaborators_info.append("\n".join(collab_info))

        return "\n".join(collaborators_info)

    def gitea_add_collaborator(self, username: str, permission: str = "write") -> str:
        """
        Add a collaborator to a repository.

        Args:
            username: Username to add as collaborator
            permission: Permission level (read, write, admin)

        Returns:
            Success message
        """
        self._ensure_repo_info()

        data = {
            "permission": permission
        }

        response = self._request("PUT", f"repos/{self.owner}/{self.repo}/collaborators/{username}", data)

        return f"{username} added as a collaborator to {self.owner}/{self.repo}."

    def gitea_remove_collaborator(self, username: str) -> str:
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

    # Releases
    def gitea_list_releases(self) -> str:
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
                f"Name: {release.get('name', 'No name')}",
                f"Tag: {release['tag_name']}",
                f"Created: {release['created_at']}",
                f"URL: {release['html_url']}",
                f"Draft: {release.get('draft', False)}",
                f"Prerelease: {release.get('prerelease', False)}",
                "---"
            ]
            releases_info.append("\n".join(release_info))

        return "\n".join(releases_info)

    def gitea_create_release(self, tag_name: str, name: str = "", body: str = "",
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
            "name": name or tag_name,
            "body": body,
            "draft": draft,
            "prerelease": prerelease
        }

        response = self._request("POST", f"repos/{self.owner}/{self.repo}/releases", data)

        return (f"Release created successfully!\n"
                f"Name: {response.get('name', tag_name)}\n"
                f"Tag: {response['tag_name']}\n"
                f"URL: {response['html_url']}")

    def get_tools(self):
        """
        Return a list of function names for the available Gitea operations.

        Returns:
            List of method names
        """
        return [
            FunctionTool(self.gitea_set_owner_and_repo),
            FunctionTool(self.gitea_list_repositories),
            FunctionTool(self.gitea_create_repository),
            FunctionTool(self.gitea_delete_repository),
            FunctionTool(self.gitea_list_branches),
            FunctionTool(self.gitea_create_branch),
            FunctionTool(self.gitea_delete_branch),
            FunctionTool(self.gitea_get_file_content),
            FunctionTool(self.gitea_create_or_update_file),
            FunctionTool(self.gitea_delete_file),
            FunctionTool(self.gitea_list_issues),
            FunctionTool(self.gitea_create_issue),
            FunctionTool(self.gitea_get_issue),
            FunctionTool(self.gitea_create_pull_request),
            FunctionTool(self.gitea_merge_pull_request),
            FunctionTool(self.gitea_list_pull_requests),
            FunctionTool(self.gitea_get_pull_request),
            FunctionTool(self.gitea_list_collaborators),
            FunctionTool(self.gitea_add_collaborator),
            FunctionTool(self.gitea_remove_collaborator),
            FunctionTool(self.gitea_list_releases),
            FunctionTool(self.gitea_create_release)
        ]
"""Version-history operations for NavigableMemory."""

from __future__ import annotations

import difflib
from typing import List, Optional

from .models import DocumentVersion


class NavigableVersionMixin:
    def list_versions(self, path: str) -> List[DocumentVersion]:
        """List all historical versions of a document, newest first."""
        if not self._has_versioning_support():
            return []
        return self.backend.list_versions(path)  # type: ignore[attr-defined]

    def list_history(
        self, path: str, include_current: bool = False,
    ) -> List[DocumentVersion]:
        """List old versions of a document, newest first.

        ``list_versions`` exposes the backend's raw snapshot list, which
        includes the current version for versioned backends in this package.
        This helper is aimed at UI/tooling use-cases where "history" means
        previous content only.
        """
        versions = self.list_versions(path)
        if include_current:
            return versions

        current = self.read(path)
        if current is None:
            return versions
        return [v for v in versions if v.version != current.version]

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        """Read a specific historical version."""
        if not self._has_versioning_support():
            return None
        return self.backend.get_version(path, version)  # type: ignore[attr-defined]

    def format_version(self, path: str, version: int) -> str:
        """Render a saved version for display."""
        ver = self.get_version(path, version)
        if ver is None:
            return f"Version {version} of '{path}' not found."
        return self._format_document_version(ver)

    def compare_versions(
        self, path: str, from_version: int, to_version: Optional[int] = None,
        context_lines: int = 3,
    ) -> str:
        """Return a unified diff between two versions or a version and current.

        Args:
            path: Document path.
            from_version: Baseline version number.
            to_version: Target version number. If omitted, compare to the
                current document content.
            context_lines: Number of unchanged context lines around each diff
                hunk.
        """
        before = self.get_version(path, from_version)
        if before is None:
            return f"Version {from_version} of '{path}' not found."

        after_label: str
        if to_version is None:
            current = self.read(path)
            if current is None:
                return f"Current document not found: '{path}'"
            after_title = current.title
            after_content = current.content
            after_binary = current.is_binary
            after_mime = current.mime_type
            after_size = current.human_size
            after_label = f"{path} (current v{current.version})"
        else:
            after = self.get_version(path, to_version)
            if after is None:
                return f"Version {to_version} of '{path}' not found."
            after_title = after.title
            after_content = after.content
            after_binary = after.is_binary
            after_mime = after.mime_type
            after_size = after.human_size
            after_label = f"{path} (v{after.version})"

        before_label = f"{path} (v{before.version})"
        if before.is_binary or after_binary:
            lines = [
                f"Binary comparison for '{path}':",
                f"  {before_label}: {before.mime_type}, {before.human_size}",
                f"  {after_label}: {after_mime}, {after_size}",
            ]
            if before.title != after_title:
                lines.append(f"  title: {before.title!r} -> {after_title!r}")
            if before.content != after_content:
                lines.append("  caption changed")
            return "\n".join(lines)

        diff = difflib.unified_diff(
            before.content.splitlines(),
            after_content.splitlines(),
            fromfile=before_label,
            tofile=after_label,
            lineterm="",
            n=max(0, context_lines),
        )
        body = "\n".join(diff)
        return body or f"No content changes between {before_label} and {after_label}."

    def build_version_context(
        self, path: Optional[str] = None, max_versions: int = 3,
        include_content: bool = False, max_chars: int = 800,
    ) -> str:
        """Build display-ready context for old versions of a document.

        This is useful as a PromptComposer module or a quick UI panel when
        the agent should reason about previous content without rolling back.
        """
        target = path or self.location.current_path
        if not target:
            return ""

        history = self.list_history(target)[:max(0, max_versions)]
        if not history:
            return f"## Previous versions\nNo older versions for '{target}'."

        lines = [f"## Previous versions of '{target}'"]
        for ver in history:
            note = f" — {ver.change_note}" if ver.change_note else ""
            author = f" by {ver.author}" if ver.author else ""
            lines.append(f"- v{ver.version} ({ver.created_at}){author}{note}")
            if include_content:
                content = ver.content
                if len(content) > max_chars:
                    content = content[:max_chars].rstrip() + "..."
                if ver.is_binary:
                    lines.append(
                        f"  [{ver.mime_type}, {ver.human_size}]"
                        + (f" Caption: {content}" if content else "")
                    )
                else:
                    snippet = content.replace("\n", " ")
                    lines.append(f"  {snippet}")
        return "\n".join(lines)

    def rollback(self, path: str, version: int, author: str = "",
                 change_note: str = "") -> bool:
        """Restore a document to a previous version (creating a new version)."""
        if not self._has_versioning_support():
            return False
        ok = self.backend.rollback(  # type: ignore[attr-defined]
            path, version, author=author, change_note=change_note,
        )
        # Refresh current content if we rolled back the active location
        if ok and self.location.current_path == path:
            doc = self.backend.read(path)
            if doc is not None:
                self.location.current_title = doc.title
                self.location.current_content = doc.content
        if ok:
            self._refresh_semantic_index(path)
        return ok

    def prune_versions(self, path: str, keep_last_n: int) -> int:
        """Drop old versions, keeping only the most recent N."""
        if not self._has_versioning_support():
            return 0
        return self.backend.prune_versions(path, keep_last_n)  # type: ignore[attr-defined]

    def _format_document_version(self, ver: DocumentVersion) -> str:
        """Render a version snapshot with metadata and content."""
        header = (
            f"## {ver.title} (v{ver.version})\n"
            f"Saved: {ver.created_at}"
        )
        if ver.author:
            header += f" by {ver.author}"
        if ver.change_note:
            header += f"\nNote: {ver.change_note}"
        if ver.is_binary:
            return (
                f"{header}\n\n[binary {ver.mime_type}, {ver.human_size}]"
                + (f"\nCaption: {ver.content}" if ver.content else "")
            )
        return f"{header}\n\n{ver.content}"

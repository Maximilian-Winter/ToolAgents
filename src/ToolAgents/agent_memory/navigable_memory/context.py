"""Navigation and context rendering mixin for NavigableMemory."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from .models import DepartureRecord, Document

logger = logging.getLogger(__name__)


class NavigableContextMixin:
    def navigate(self, path: str) -> str:
        """Navigate to a document path. Loads its content as current context.

        If already at a location, fires the on_depart callback with the
        old location's data before moving.

        Args:
            path: Full document path (e.g., "projects/vr/status.md").

        Returns:
            Status message describing what happened.
        """
        doc = self.backend.read(path)
        if doc is None:
            return f"Location not found: '{path}'"

        # Depart from current location
        if self.location.has_location:
            departure = DepartureRecord(
                path=self.location.current_path,
                title=self.location.current_title,
                content=self.location.current_content,
                timestamp=datetime.now().isoformat(),
            )
            self._departure_history.append(departure)
            if len(self._departure_history) > self.context_window * 2:
                self._departure_history = self._departure_history[-(self.context_window * 2):]

            if self.on_depart:
                try:
                    self.on_depart(departure)
                except Exception as e:
                    logger.error("Error in on_depart callback: %s", e)

        # Arrive at new location
        self.location.move_to(path, doc.title, doc.content)

        visit = self.location.visit_count
        visit_note = f" (visit #{visit})" if visit > 1 else ""
        logger.info("Navigated to: %s%s", doc.title, visit_note)
        return f"Navigated to: {doc.title}{visit_note}"

    def navigate_up(self) -> str:
        """Navigate to the parent directory's overview document.

        Looks for an overview.md in the parent path. If not found,
        lists what's available at the parent level.
        """
        if not self.location.has_location:
            return "No current location."

        parent = self._get_parent_prefix(self.location.current_path)
        if not parent:
            return "Already at root level."

        overview_path = f"{parent}overview.md"
        doc = self.backend.read(overview_path)
        if doc:
            return self.navigate(overview_path)

        # No overview — list what's available
        items = self.backend.list(parent)
        if items:
            listing = "\n".join(f"  - {d.title} ({d.path})" for d in items[:10])
            return f"No overview at '{parent}'. Available:\n{listing}"
        return f"Nothing found at '{parent}'."

    # ── Context Building ──────────────────────────────────────────

    def build_context(self) -> str:
        """Build the full context string for the current location.

        This is designed to be used as a PromptComposer content_fn:
            composer.add_module("context", content_fn=memory.build_context)

        Returns:
            Assembled context string with current location, parent overview,
            and sibling listings.
        """
        if not self.location.has_location:
            return "No location loaded. Use navigate to move to a knowledge path."

        parts = []

        # Parent overview for broader context
        if self.include_parent:
            parent_content = self._get_parent_overview()
            if parent_content:
                parts.append(parent_content)

        # Current location (full detail)
        # Re-read so we get mime_type/binary metadata + version freshness
        current_doc = self.backend.read(self.location.current_path)
        parts.append(self._format_current_doc(current_doc))

        # Outgoing references (graph edges)
        if self._has_references_support():
            outgoing = self.backend.list_references_from(self.location.current_path)
            if outgoing:
                lines = ["## References from here:"]
                for r in outgoing:
                    target = self.backend.read(r.to_path)
                    label = target.title if target else r.to_path
                    note = f" — {r.note}" if r.note else ""
                    lines.append(f"  → [{r.ref_type}] {label} ({r.to_path}){note}")
                parts.append("\n".join(lines))

        # Sibling documents (what else is nearby)
        if self.include_siblings:
            siblings = self._get_siblings()
            if siblings:
                listing = "\n".join(
                    f"  - {d.title} ({d.path})" for d in siblings
                )
                parts.append(f"## Nearby:\n{listing}")

        return "\n\n---\n\n".join(parts)

    def _format_current_doc(self, doc: Optional[Document]) -> str:
        """Render the current location, handling binary documents gracefully."""
        if doc is None:
            # Backend doesn't return a doc — fall back to cached state
            return (
                f"## Current: {self.location.current_title}\n"
                f"Path: {self.location.current_path}\n\n"
                f"{self.location.current_content}"
            )

        version_tag = f" (v{doc.version})" if doc.version > 1 else ""
        if doc.is_binary:
            kind = "image" if doc.is_image else ("audio" if doc.is_audio else "binary")
            header = (
                f"## Current: {doc.title}{version_tag}\n"
                f"Path: {doc.path}\n"
                f"Type: {doc.mime_type} ({doc.human_size})\n\n"
                f"[📎 {kind} attachment — bytes available via read_binary('{doc.path}')]"
            )
            if doc.content:
                header += f"\n\nCaption: {doc.content}"
            return header

        return (
            f"## Current: {doc.title}{version_tag}\n"
            f"Path: {doc.path}\n\n"
            f"{doc.content}"
        )

    def build_history_context(self) -> str:
        """Build a summary of recently visited locations.

        Useful as a secondary PromptComposer module showing
        where the agent has been.
        """
        if not self._departure_history:
            return ""

        recent = self._departure_history[-self.context_window:]
        lines = ["## Recent locations:"]
        for dep in reversed(recent):
            snippet = dep.content[:150].replace("\n", " ")
            lines.append(f"  - {dep.title} ({dep.path}): {snippet}...")
        return "\n".join(lines)

    def _get_parent_prefix(self, path: str) -> Optional[str]:
        """Get the parent directory prefix for a path."""
        parts = path.rsplit("/", 1)
        if len(parts) > 1:
            return parts[0] + "/"
        return None

    def _get_parent_overview(self) -> Optional[str]:
        """Load the parent directory's overview document."""
        if not self.location.current_path:
            return None
        parent = self._get_parent_prefix(self.location.current_path)
        if not parent:
            return None

        overview_path = f"{parent}overview.md"
        if overview_path == self.location.current_path:
            return None

        doc = self.backend.read(overview_path)
        if doc:
            return f"## Area: {doc.title}\n{doc.content}"
        return None

    def _get_siblings(self) -> List[Document]:
        """Get sibling documents in the same directory."""
        if not self.location.current_path:
            return []
        parent = self._get_parent_prefix(self.location.current_path)
        if not parent:
            return []

        docs = self.backend.list(parent)
        return [
            d for d in docs
            if d.path != self.location.current_path
            and not d.path.endswith("overview.md")
        ]

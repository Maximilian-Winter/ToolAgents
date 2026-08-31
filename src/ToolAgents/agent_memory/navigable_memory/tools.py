"""Pydantic tool models for NavigableMemory."""

from __future__ import annotations

from typing import List, Optional


def create_navigable_memory_tools(nav_memory) -> list:
    """Create FunctionTool-compatible Pydantic models for the LLM.

    Returns a list of Pydantic model classes with run() methods
    that can be wrapped with FunctionTool().

    Usage:
        from ToolAgents import FunctionTool
        tools = [FunctionTool(t) for t in memory.create_tools()]
        harness.add_tools(tools)
    """
    from pydantic import BaseModel, Field

    class Navigate(BaseModel):
        """Navigate to a location in the knowledge space.
        This loads the document's content into the active context.
        Use list_locations first to discover available paths."""
        path: str = Field(
            ..., description="Full document path (e.g., 'projects/vr/status.md')."
        )

        def run(self) -> str:
            return nav_memory.navigate(self.path)

    class NavigateUp(BaseModel):
        """Navigate up to the parent area. Shows the overview
        or lists available documents at the parent level."""

        def run(self) -> str:
            return nav_memory.navigate_up()

    class ListLocations(BaseModel):
        """List available documents under a path prefix.
        Use to discover what knowledge is available."""
        prefix: str = Field(
            "", description="Path prefix with trailing slash (e.g., 'projects/')."
        )

        def run(self) -> str:
            docs = nav_memory.list_at(self.prefix)
            if not docs:
                return f"No documents under '{self.prefix}'."
            lines = [f"Documents under '{self.prefix}':"]
            for d in docs:
                lines.append(f"  - {d.title} ({d.path})")
            return "\n".join(lines)

    class SearchKnowledge(BaseModel):
        """Search the knowledge base for a term or topic."""
        query: str = Field(
            ..., description="Search term."
        )

        def run(self) -> str:
            results = nav_memory.search(self.query)
            if not results:
                return f"No results for '{self.query}'."
            lines = [f"Search results for '{self.query}':"]
            for d in results[:8]:
                snippet = d.content[:120].replace("\n", " ")
                lines.append(f"  - {d.title} ({d.path}): {snippet}...")
            return "\n".join(lines)

    class SemanticSearchKnowledge(BaseModel):
        """Search the optional semantic index for conceptually related text.
        Results include source paths so you can navigate to the full document."""
        query: str = Field(..., description="Natural-language search query.")
        k: int = Field(8, description="Maximum number of chunk results.")
        path_prefix: Optional[str] = Field(
            None, description="Optional path prefix filter."
        )
        tags: Optional[List[str]] = Field(
            None, description="Optional tags that all results must have."
        )

        def run(self) -> str:
            results = nav_memory.semantic_search(
                self.query, k=self.k,
                path_prefix=self.path_prefix, tags=self.tags,
            )
            if not results:
                return f"No semantic results for '{self.query}'."
            lines = [f"Semantic results for '{self.query}':"]
            for idx, hit in enumerate(results, start=1):
                tag_text = f" tags={hit.tags}" if hit.tags else ""
                lines.append(
                    f"  {idx}. {hit.title} ({hit.path}) "
                    f"score={hit.score:.3f} v{hit.version} "
                    f"chunk={hit.chunk_index}{tag_text}"
                )
                lines.append(f"     {hit.snippet(160)}")
                lines.append(f"     Navigate to: {hit.path}")
            return "\n".join(lines)

    class HybridSearchKnowledge(BaseModel):
        """Search using semantic chunks plus lexical, tag, and reference boosts."""
        query: str = Field(..., description="Natural-language search query.")
        k: int = Field(8, description="Maximum number of document results.")
        path_prefix: Optional[str] = Field(
            None, description="Optional path prefix filter."
        )
        tags: Optional[List[str]] = Field(
            None, description="Optional tags that all results must have."
        )
        include_references: bool = Field(
            True,
            description="Boost direct references around the current location.",
        )

        def run(self) -> str:
            results = nav_memory.hybrid_search(
                self.query, k=self.k,
                path_prefix=self.path_prefix, tags=self.tags,
                include_references=self.include_references,
            )
            if not results:
                return f"No hybrid results for '{self.query}'."
            lines = [f"Hybrid results for '{self.query}':"]
            for idx, hit in enumerate(results, start=1):
                sources = hit.metadata.get("sources") or []
                source_text = f" sources={sources}" if sources else ""
                tag_text = f" tags={hit.tags}" if hit.tags else ""
                lines.append(
                    f"  {idx}. {hit.title} ({hit.path}) "
                    f"score={hit.score:.3f} v{hit.version}{source_text}{tag_text}"
                )
                lines.append(f"     {hit.snippet(160)}")
                lines.append(f"     Navigate to: {hit.path}")
            return "\n".join(lines)

    class ReadDocument(BaseModel):
        """Read a specific document without navigating to it.
        Use when you need to check content elsewhere without
        changing the current context."""
        path: str = Field(
            ..., description="Full document path."
        )

        def run(self) -> str:
            doc = nav_memory.read(self.path)
            if doc is None:
                return f"Not found: '{self.path}'"
            return f"## {doc.title}\n\n{doc.content}"

    class WriteDocument(BaseModel):
        """Write or update a document in the knowledge base.
        Creates the document if it doesn't exist, overwrites if it does."""
        path: str = Field(
            ..., description="Full document path."
        )
        title: str = Field(
            ..., description="Document title."
        )
        content: str = Field(
            ..., description="Document content (markdown)."
        )

        def run(self) -> str:
            ok = nav_memory.write(self.path, self.title, self.content)
            return f"Written: '{self.title}'" if ok else f"Failed to write '{self.path}'."

    class AppendToDocument(BaseModel):
        """Append content to an existing document.
        Adds a timestamped log entry. Useful for event logs and notes."""
        path: str = Field(
            ..., description="Full document path."
        )
        content: str = Field(
            ..., description="Content to append."
        )

        def run(self) -> str:
            return nav_memory.append(self.path, self.content)

    # ── Versioning Tools ─────────────────────────────────────

    class ListVersions(BaseModel):
        """List previous versions of a document, newest first.
        Use to inspect change history before reading or rolling back."""
        path: str = Field(..., description="Full document path.")
        include_current: bool = Field(
            False,
            description="Include the current version snapshot as well.",
        )

        def run(self) -> str:
            versions = nav_memory.list_history(
                self.path, include_current=self.include_current,
            )
            if not versions:
                return (
                    f"No previous versions for '{self.path}' "
                    "(or backend does not support versioning)."
                )
            lines = [f"Versions of '{self.path}':"]
            for v in versions:
                note = f" — {v.change_note}" if v.change_note else ""
                author = f" by {v.author}" if v.author else ""
                lines.append(
                    f"  v{v.version} ({v.created_at}){author}{note}"
                )
            return "\n".join(lines)

    class ReadVersion(BaseModel):
        """Read a specific historical version of a document.
        Does not navigate or change current location."""
        path: str = Field(..., description="Full document path.")
        version: int = Field(..., description="Version number to read.")

        def run(self) -> str:
            return nav_memory.format_version(self.path, self.version)

    class CompareVersions(BaseModel):
        """Compare an old document version with another version or current.
        Returns a unified text diff for text documents."""
        path: str = Field(..., description="Full document path.")
        from_version: int = Field(
            ..., description="Older/baseline version number."
        )
        to_version: Optional[int] = Field(
            None,
            description="Target version number. Defaults to current content.",
        )
        context_lines: int = Field(
            3, description="Unchanged lines to show around each diff hunk.",
        )

        def run(self) -> str:
            return nav_memory.compare_versions(
                self.path, self.from_version, self.to_version,
                self.context_lines,
            )

    class ShowVersionContext(BaseModel):
        """Show a compact context block of previous document versions."""
        path: Optional[str] = Field(
            None,
            description="Document path. Defaults to current location.",
        )
        max_versions: int = Field(
            3, description="Maximum number of previous versions to show.",
        )
        include_content: bool = Field(
            False, description="Include content snippets for each version.",
        )

        def run(self) -> str:
            return nav_memory.build_version_context(
                self.path, self.max_versions, self.include_content,
            )

    class RollbackToVersion(BaseModel):
        """Restore a document to a previous version.
        Creates a NEW version on top — history is never lost."""
        path: str = Field(..., description="Full document path.")
        version: int = Field(..., description="Version number to restore.")
        reason: str = Field(
            "", description="Optional change note explaining the rollback."
        )

        def run(self) -> str:
            ok = nav_memory.rollback(
                self.path, self.version,
                change_note=self.reason or f"rolled back to v{self.version}",
            )
            if not ok:
                return (
                    f"Could not rollback '{self.path}' to v{self.version} "
                    "(version not found or backend lacks versioning)."
                )
            return f"Restored '{self.path}' to v{self.version}."

    # ── Reference Tools ──────────────────────────────────────

    class AddReference(BaseModel):
        """Create a directed reference between two documents.
        Use to capture relationships: links, dependencies, supersedes, etc."""
        from_path: str = Field(..., description="Source document path.")
        to_path: str = Field(..., description="Target document path.")
        ref_type: str = Field(
            "links_to",
            description=(
                "Relationship kind. Common values: 'links_to', 'depends_on', "
                "'supersedes', 'see_also', 'embeds', 'replies_to', 'derived_from'."
            ),
        )
        note: str = Field(
            "", description="Optional annotation explaining the link."
        )

        def run(self) -> str:
            ok = nav_memory.add_reference(
                self.from_path, self.to_path, self.ref_type, self.note,
            )
            if not ok:
                return (
                    f"Reference '{self.from_path}' →[{self.ref_type}]→ "
                    f"'{self.to_path}' already exists or backend lacks support."
                )
            return (
                f"Linked: '{self.from_path}' →[{self.ref_type}]→ "
                f"'{self.to_path}'"
            )

    class RemoveReference(BaseModel):
        """Remove a reference between two documents."""
        from_path: str = Field(..., description="Source document path.")
        to_path: str = Field(..., description="Target document path.")
        ref_type: Optional[str] = Field(
            None,
            description=(
                "If given, only remove references of this type. "
                "Otherwise remove all edges between the two paths."
            ),
        )

        def run(self) -> str:
            n = nav_memory.remove_reference(
                self.from_path, self.to_path, self.ref_type,
            )
            return f"Removed {n} reference(s)."

    class ListReferences(BaseModel):
        """List references for a document.
        Direction: 'from' = outgoing links; 'to' = backlinks; 'both' = both."""
        path: str = Field(..., description="Full document path.")
        direction: str = Field(
            "both",
            description="Direction: 'from', 'to', or 'both'. Default 'both'.",
        )

        def run(self) -> str:
            direction = self.direction.lower().strip()
            lines: List[str] = []
            if direction in ("from", "both"):
                out = nav_memory.references_from(self.path)
                if out:
                    lines.append(f"Outgoing from '{self.path}':")
                    for r in out:
                        note = f" — {r.note}" if r.note else ""
                        lines.append(
                            f"  → [{r.ref_type}] {r.to_path}{note}"
                        )
            if direction in ("to", "both"):
                incoming = nav_memory.references_to(self.path)
                if incoming:
                    lines.append(f"Backlinks to '{self.path}':")
                    for r in incoming:
                        note = f" — {r.note}" if r.note else ""
                        lines.append(
                            f"  ← [{r.ref_type}] {r.from_path}{note}"
                        )
            if not lines:
                return f"No references found for '{self.path}'."
            return "\n".join(lines)

    # ── Tag Tools ────────────────────────────────────────────

    class ListTags(BaseModel):
        """List every unique tag used across the knowledge base."""

        def run(self) -> str:
            tags = nav_memory.list_tags()
            if not tags:
                return "No tags in use."
            return "Tags in use:\n" + "\n".join(f"  - {t}" for t in tags)

    class FindByTag(BaseModel):
        """Find all documents that carry a specific tag."""
        tag: str = Field(..., description="The tag to search for.")

        def run(self) -> str:
            docs = nav_memory.list_by_tag(self.tag)
            if not docs:
                return f"No documents tagged '{self.tag}'."
            lines = [f"Documents tagged '{self.tag}':"]
            for d in docs:
                lines.append(f"  - {d.title} ({d.path})")
            return "\n".join(lines)

    class FindByTags(BaseModel):
        """Find documents by combining multiple tags.
        Mode 'any' = at least one tag matches (OR).
        Mode 'all' = all tags must match (AND).
        Mode 'none' = none of these tags (exclusion)."""
        tags: List[str] = Field(..., description="Tags to match.")
        mode: str = Field(
            "any", description="'any' (OR), 'all' (AND), or 'none'.",
        )

        def run(self) -> str:
            docs = nav_memory.find_by_tags(self.tags, self.mode)
            if not docs:
                return f"No documents match tags={self.tags} mode={self.mode}."
            lines = [f"Documents matching {self.tags} ({self.mode}):"]
            for d in docs:
                tag_list = ", ".join(d.tags) if d.tags else "(none)"
                lines.append(f"  - {d.title} ({d.path}) [tags: {tag_list}]")
            return "\n".join(lines)

    class AddTags(BaseModel):
        """Add tags to a document (existing tags are preserved)."""
        path: str = Field(..., description="Document path.")
        tags: List[str] = Field(..., description="Tags to add.")

        def run(self) -> str:
            if not nav_memory.add_tags(self.path, *self.tags):
                return f"Document not found: '{self.path}'"
            doc = nav_memory.read(self.path)
            current = ", ".join(doc.tags) if doc and doc.tags else "(none)"
            return f"Tags added to '{self.path}'. Current: [{current}]"

    class RemoveTags(BaseModel):
        """Remove specific tags from a document."""
        path: str = Field(..., description="Document path.")
        tags: List[str] = Field(..., description="Tags to remove.")

        def run(self) -> str:
            if not nav_memory.remove_tags(self.path, *self.tags):
                return f"Document not found: '{self.path}'"
            doc = nav_memory.read(self.path)
            current = ", ".join(doc.tags) if doc and doc.tags else "(none)"
            return f"Tags removed from '{self.path}'. Current: [{current}]"

    class SetTags(BaseModel):
        """Replace a document's entire tag list with the given tags."""
        path: str = Field(..., description="Document path.")
        tags: List[str] = Field(..., description="New tag list (replaces existing).")

        def run(self) -> str:
            if not nav_memory.set_tags(self.path, self.tags):
                return f"Document not found: '{self.path}'"
            return f"Tags on '{self.path}' set to: [{', '.join(self.tags)}]"

    # ── Reference Walking Tool ───────────────────────────────

    class FollowReferences(BaseModel):
        """Walk the reference graph from a starting document.
        Returns a tree showing connected docs up to max_depth hops away.
        Useful for exploring how knowledge is interconnected."""
        path: Optional[str] = Field(
            None,
            description=(
                "Starting document path. Defaults to the current location."
            ),
        )
        direction: str = Field(
            "outgoing",
            description=(
                "'outgoing' (follow links from this doc), "
                "'incoming' (follow backlinks to this doc), "
                "or 'both'. Default 'outgoing'."
            ),
        )
        max_depth: int = Field(
            2, description="How many hops to traverse. Default 2.",
        )
        ref_types: Optional[List[str]] = Field(
            None,
            description=(
                "Optional filter by ref_type, e.g. ['embeds', 'depends_on']. "
                "If omitted, all edge types are followed."
            ),
        )
        max_nodes: int = Field(
            25, description="Maximum nodes to visit before truncating.",
        )

        def run(self) -> str:
            start = self.path or nav_memory.current_path
            if not start:
                return (
                    "No starting path: provide 'path' or navigate to a "
                    "document first."
                )
            walk = nav_memory.walk_references(
                start_path=start,
                direction=self.direction,
                max_depth=self.max_depth,
                ref_types=self.ref_types,
                max_nodes=self.max_nodes,
            )
            tree = nav_memory.render_reference_walk(walk)
            summary = (
                f"Reference walk from '{start}' "
                f"(direction={self.direction}, depth={self.max_depth}, "
                f"visited {len(walk['nodes'])} nodes):"
            )
            return f"{summary}\n{tree}"

    # ── Binary Tools ─────────────────────────────────────────

    class DescribeBinary(BaseModel):
        """Inspect a binary document's metadata (mime type, size, caption).
        Useful before deciding whether to retrieve raw bytes."""
        path: str = Field(..., description="Full document path.")

        def run(self) -> str:
            doc = nav_memory.read(self.path)
            if doc is None:
                return f"Not found: '{self.path}'"
            if not doc.is_binary:
                return f"'{self.path}' is a text document ({doc.mime_type})."
            lines = [
                f"## {doc.title}",
                f"Path: {doc.path}",
                f"Type: {doc.mime_type}",
                f"Size: {doc.human_size}",
                f"Version: {doc.version}",
            ]
            if doc.content:
                lines.append(f"Caption: {doc.content}")
            if doc.tags:
                lines.append(f"Tags: {', '.join(doc.tags)}")
            return "\n".join(lines)

    tools = [
        Navigate, NavigateUp, ListLocations, SearchKnowledge,
        ReadDocument, WriteDocument, AppendToDocument,
        # Tag operations always work (NavigableMemory has fallbacks)
        ListTags, FindByTag, FindByTags, AddTags, RemoveTags, SetTags,
    ]
    if nav_memory._has_semantic_support():
        tools.extend([SemanticSearchKnowledge, HybridSearchKnowledge])
    if nav_memory._has_versioning_support():
        tools.extend([
            ListVersions, ReadVersion, CompareVersions,
            ShowVersionContext, RollbackToVersion,
        ])
    if nav_memory._has_references_support():
        tools.extend([AddReference, RemoveReference, ListReferences,
                      FollowReferences])
    if nav_memory._has_binary_support():
        tools.append(DescribeBinary)
    return tools

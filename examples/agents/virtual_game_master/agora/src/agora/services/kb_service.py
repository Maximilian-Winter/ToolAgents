"""Knowledge base service — section extraction, tree building, FTS sync."""

import re
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.models.kb_document import KBDocument

# Note: The spec mentions SQLAlchemy event listeners for FTS sync, but async
# sessions do not support synchronous event hooks. FTS sync is done via
# explicit async calls (fts_insert/fts_update/fts_delete) in the route handlers.


# ── Section Extraction ──────────────────────────────────────────


def extract_section(content: str, section_name: str) -> Optional[str]:
    """Extract a section from markdown content by header text.

    Returns content from the matched header down to the next header
    of equal or higher level. Case-insensitive match.
    Returns None if no matching section found.
    """
    lines = content.split("\n")
    result_lines: list[str] = []
    capturing = False
    capture_level = 0

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            if capturing:
                # Stop at same or higher level header
                if level <= capture_level:
                    break
            elif header_text.lower() == section_name.lower():
                capturing = True
                capture_level = level
                result_lines.append(line)
                continue
        if capturing:
            result_lines.append(line)

    if not result_lines:
        return None
    return "\n".join(result_lines).strip()


# ── Tree Building ────────────────────────────────────────────────


def build_tree(docs: list[dict]) -> list[dict]:
    """Build a nested tree structure from flat document list.

    Input: [{"path": "a/b.md", "title": "B"}, ...]
    Output: [{"name": "a", "children": [{"name": "b.md", "path": "a/b.md", "title": "B"}]}]
    """
    root: dict = {}

    for doc in docs:
        parts = doc["path"].split("/")
        node = root
        for i, part in enumerate(parts):
            if part not in node:
                node[part] = {}
            if i < len(parts) - 1:
                if "__children__" not in node[part]:
                    node[part]["__children__"] = {}
                node = node[part]["__children__"]
            else:
                node[part]["__leaf__"] = doc

    def _to_list(node: dict) -> list[dict]:
        result = []
        for name, value in sorted(node.items()):
            if "__leaf__" in value:
                leaf = value["__leaf__"]
                result.append({"name": name, "path": leaf["path"], "title": leaf["title"]})
            elif "__children__" in value:
                result.append({"name": name, "children": _to_list(value["__children__"])})
            else:
                result.append({"name": name, "children": _to_list(value)})
        return result

    return _to_list(root)


# ── Tag Matching ─────────────────────────────────────────────────


def tags_contain(tags_csv: Optional[str], tag: str) -> bool:
    """Check if a comma-separated tags string contains an exact tag match."""
    if not tags_csv:
        return False
    return tag.strip().lower() in [t.strip().lower() for t in tags_csv.split(",")]


# ── FTS5 Sync ────────────────────────────────────────────────────


async def create_fts_table(db: AsyncSession) -> None:
    """Create FTS5 virtual table if it doesn't exist."""
    await db.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS kb_document_fts USING fts5(
            title, content, tags,
            content='kb_documents',
            content_rowid='id'
        )
    """))
    await db.commit()


async def fts_insert(db: AsyncSession, doc_id: int, title: str, content: str, tags: str) -> None:
    """Insert a document into the FTS index."""
    await db.execute(
        text("INSERT INTO kb_document_fts(rowid, title, content, tags) VALUES (:id, :title, :content, :tags)"),
        {"id": doc_id, "title": title, "content": content, "tags": tags or ""},
    )


async def fts_delete(db: AsyncSession, doc_id: int, title: str, content: str, tags: str) -> None:
    """Delete a document from the FTS index."""
    await db.execute(
        text("INSERT INTO kb_document_fts(kb_document_fts, rowid, title, content, tags) VALUES('delete', :id, :title, :content, :tags)"),
        {"id": doc_id, "title": title, "content": content, "tags": tags or ""},
    )


async def fts_update(db: AsyncSession, doc_id: int, old_title: str, old_content: str, old_tags: str, new_title: str, new_content: str, new_tags: str) -> None:
    """Update a document in the FTS index (delete old + insert new)."""
    await fts_delete(db, doc_id, old_title, old_content, old_tags)
    await fts_insert(db, doc_id, new_title, new_content, new_tags)

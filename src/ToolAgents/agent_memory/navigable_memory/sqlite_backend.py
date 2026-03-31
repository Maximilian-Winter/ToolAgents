"""
SQLite Storage Backend for NavigableMemory.

Persistent document storage with full-text search (FTS5).
Single-file database, no external dependencies beyond Python stdlib.

Usage:
    from navigable_memory import NavigableMemory
    from sqlite_backend import SQLiteBackend

    backend = SQLiteBackend("my_knowledge.db")
    memory = NavigableMemory(backend)

    memory.write("projects/api.md", "API Design", "# API Design\n...")
    memory.navigate("projects/api.md")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .navigable_memory import Document

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    path       TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'Untitled',
    content    TEXT NOT NULL DEFAULT '',
    tags       TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, title, content, tags,
    content='documents',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, path, title, content, tags)
    VALUES (new.rowid, new.path, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, path, title, content, tags)
    VALUES ('delete', old.rowid, old.path, old.title, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, path, title, content, tags)
    VALUES ('delete', old.rowid, old.path, old.title, old.content, old.tags);
    INSERT INTO documents_fts(rowid, path, title, content, tags)
    VALUES (new.rowid, new.path, new.title, new.content, new.tags);
END;
"""


class SQLiteBackend:
    """StorageBackend implementation using SQLite with FTS5 full-text search.

    Features:
    - Single-file persistent storage
    - Full-text search via SQLite FTS5
    - Automatic schema creation
    - Tags stored as comma-separated string, queried via FTS
    - Metadata stored as JSON blob
    - Thread-safe with check_same_thread=False

    Args:
        db_path: Path to the SQLite database file.
            Use ":memory:" for an in-memory database (testing).
        wal_mode: Enable WAL journal mode for better concurrent reads.
            Defaults to True for file-based databases.
    """

    def __init__(self, db_path: str = "knowledge.db", wal_mode: bool = True):
        self.db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if wal_mode and db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")

        self._init_schema()

    def _init_schema(self):
        """Create tables and FTS index if they don't exist."""
        self._conn.executescript(_SCHEMA)

    # ── StorageBackend Protocol ───────────────────────────────────

    def read(self, path: str) -> Optional[Document]:
        row = self._conn.execute(
            "SELECT path, title, content, tags, metadata, updated_at "
            "FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        now = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)

        try:
            existing = self._conn.execute(
                "SELECT created_at FROM documents WHERE path = ?", (path,)
            ).fetchone()

            if existing:
                self._conn.execute(
                    "UPDATE documents SET title=?, content=?, tags=?, metadata=?, updated_at=? "
                    "WHERE path=?",
                    (title, content, tags_str, meta_str, now, path),
                )
            else:
                self._conn.execute(
                    "INSERT INTO documents (path, title, content, tags, metadata, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (path, title, content, tags_str, meta_str, now, now),
                )
            return True
        except sqlite3.Error as e:
            logger.error("SQLite write error for '%s': %s", path, e)
            return False

    def list(self, prefix: str = "") -> List[Document]:
        if prefix:
            rows = self._conn.execute(
                "SELECT path, title, content, tags, metadata, updated_at "
                "FROM documents WHERE path LIKE ? ORDER BY path",
                (prefix + "%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT path, title, content, tags, metadata, updated_at "
                "FROM documents ORDER BY path"
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def search(self, query: str) -> List[Document]:
        """Full-text search using FTS5.

        Supports FTS5 query syntax:
        - Simple terms: "websocket"
        - Phrases: '"api design"'
        - AND/OR/NOT: "rust AND async"
        - Prefix: "web*"
        """
        try:
            # Use a safe query — escape user input for basic searches
            safe_query = self._escape_fts_query(query)
            rows = self._conn.execute(
                "SELECT d.path, d.title, d.content, d.tags, d.metadata, d.updated_at "
                "FROM documents d "
                "JOIN documents_fts fts ON d.rowid = fts.rowid "
                "WHERE documents_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT 20",
                (safe_query,),
            ).fetchall()
            return [self._row_to_document(r) for r in rows]
        except sqlite3.OperationalError as e:
            # Fallback to LIKE search if FTS query syntax is invalid
            logger.debug("FTS query failed ('%s'), falling back to LIKE: %s", query, e)
            return self._search_like(query)

    def delete(self, path: str) -> bool:
        try:
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE path = ?", (path,)
            )
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error("SQLite delete error for '%s': %s", path, e)
            return False

    # ── Additional Operations ─────────────────────────────────────

    def move(self, old_path: str, new_path: str) -> bool:
        """Move/rename a document."""
        try:
            cursor = self._conn.execute(
                "UPDATE documents SET path = ?, updated_at = ? WHERE path = ?",
                (new_path, datetime.now().isoformat(), old_path),
            )
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error("SQLite move error '%s' → '%s': %s", old_path, new_path, e)
            return False

    def list_tags(self) -> List[str]:
        """Get all unique tags across all documents."""
        rows = self._conn.execute(
            "SELECT DISTINCT tags FROM documents WHERE tags != ''"
        ).fetchall()
        all_tags = set()
        for row in rows:
            for tag in row["tags"].split(","):
                tag = tag.strip()
                if tag:
                    all_tags.add(tag)
        return sorted(all_tags)

    def list_by_tag(self, tag: str) -> List[Document]:
        """List documents that have a specific tag."""
        # Use comma-bounded matching to avoid partial tag matches
        rows = self._conn.execute(
            "SELECT path, title, content, tags, metadata, updated_at "
            "FROM documents "
            "WHERE ',' || tags || ',' LIKE ? "
            "ORDER BY path",
            (f"%,{tag},%",),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def count(self) -> int:
        """Total number of documents."""
        row = self._conn.execute("SELECT COUNT(*) as n FROM documents").fetchone()
        return row["n"]

    def tree(self, prefix: str = "") -> Dict:
        """Build a nested directory tree of all document paths.

        Returns a dict like:
            {"projects": {"api.md": None, "game": {"design.md": None}}}
        Where None means leaf (document), dict means directory.
        """
        docs = self.list(prefix)
        tree = {}
        for doc in docs:
            parts = doc.path.split("/")
            node = tree
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = {"title": doc.title, "path": doc.path}
        return tree

    def stats(self) -> Dict[str, Any]:
        """Database statistics."""
        doc_count = self.count()
        tags = self.list_tags()
        size = os.path.getsize(self.db_path) if self.db_path != ":memory:" else 0
        return {
            "documents": doc_count,
            "tags": len(tags),
            "db_size_bytes": size,
            "db_path": self.db_path,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        tags_str = row["tags"]
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        meta_str = row["metadata"]
        try:
            metadata = json.loads(meta_str) if meta_str and meta_str != "{}" else {}
        except json.JSONDecodeError:
            metadata = {}
        return Document(
            path=row["path"],
            title=row["title"],
            content=row["content"],
            tags=tags,
            metadata=metadata,
            updated_at=row["updated_at"],
        )

    def _escape_fts_query(self, query: str) -> str:
        """Escape a user query for safe FTS5 usage.

        Wraps each word in double quotes to treat them as literals,
        avoiding FTS5 syntax errors from special characters.
        """
        words = query.strip().split()
        if not words:
            return '""'
        # Each word quoted and joined with implicit AND
        return " ".join(f'"{w}"' for w in words)

    def _search_like(self, query: str) -> List[Document]:
        """Fallback search using LIKE when FTS fails."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT path, title, content, tags, metadata, updated_at "
            "FROM documents "
            "WHERE content LIKE ? OR title LIKE ? OR tags LIKE ? "
            "ORDER BY path LIMIT 20",
            (pattern, pattern, pattern),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def close(self):
        """Close the database connection."""
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"SQLiteBackend('{self.db_path}', {self.count()} docs)"

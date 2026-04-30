"""
SQLite Storage Backend for NavigableMemory.

Persistent document storage with full-text search (FTS5),
binary blob support, automatic version history, and a
reference graph between documents.

Single-file database, no external dependencies beyond Python stdlib.

Usage:
    from navigable_memory import NavigableMemory
    from sqlite_backend import SQLiteBackend

    backend = SQLiteBackend("my_knowledge.db")
    memory = NavigableMemory(backend)

    memory.write("projects/api.md", "API Design", "# API Design\n...")
    memory.navigate("projects/api.md")

    # Binary
    with open("diagram.png", "rb") as f:
        memory.write_binary(
            "projects/diagram.png", "Architecture diagram",
            "image/png", f.read(), caption="System overview",
        )

    # Versioning
    versions = memory.list_versions("projects/api.md")
    memory.rollback("projects/api.md", version=2)

    # References
    memory.add_reference(
        "projects/api.md", "projects/diagram.png",
        ref_type="embeds", note="API gateway diagram",
    )
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .navigable_memory import (
    Document, DocumentVersion, Reference, RefType,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    path        TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Untitled',
    content     TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}',
    mime_type   TEXT NOT NULL DEFAULT 'text/markdown',
    binary_data BLOB,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, title, content, tags,
    content='documents',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync (text columns only — binary skipped)
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

-- Version history
CREATE TABLE IF NOT EXISTS document_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    title       TEXT NOT NULL DEFAULT 'Untitled',
    content     TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}',
    mime_type   TEXT NOT NULL DEFAULT 'text/markdown',
    binary_data BLOB,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    author      TEXT NOT NULL DEFAULT '',
    change_note TEXT NOT NULL DEFAULT '',
    UNIQUE(path, version)
);

CREATE INDEX IF NOT EXISTS idx_versions_path
    ON document_versions(path, version DESC);

-- Inter-document references (graph edges)
CREATE TABLE IF NOT EXISTS document_references (
    from_path  TEXT NOT NULL,
    to_path    TEXT NOT NULL,
    ref_type   TEXT NOT NULL DEFAULT 'links_to',
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (from_path, to_path, ref_type)
);

CREATE INDEX IF NOT EXISTS idx_refs_from ON document_references(from_path);
CREATE INDEX IF NOT EXISTS idx_refs_to   ON document_references(to_path);
"""


class SQLiteBackend:
    """StorageBackend implementation using SQLite with FTS5 full-text search.

    Implements the optional BinaryStorage, VersionedStorage, and
    ReferenceStorage protocols.

    Features:
    - Single-file persistent storage
    - Full-text search via SQLite FTS5
    - Binary blobs (images, audio, PDFs, ...)
    - Automatic version history on every write
    - Directed reference graph between documents
    - Forward-compatible schema migration

    Args:
        db_path: Path to the SQLite database file.
            Use ":memory:" for an in-memory database (testing).
        wal_mode: Enable WAL journal mode for better concurrent reads.
            Defaults to True for file-based databases.
        track_versions: If False, writes do not create version snapshots.
            Default True.
    """

    def __init__(
        self,
        db_path: str = "knowledge.db",
        wal_mode: bool = True,
        track_versions: bool = True,
    ):
        self.db_path = db_path
        self.track_versions = track_versions
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
        """Create tables and FTS index, then run forward-compatible migrations."""
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self):
        """Add columns if they're missing (databases created before
        binary/versioning support was added)."""
        cols = {
            row["name"] for row in self._conn.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }
        added = []
        if "mime_type" not in cols:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN mime_type TEXT NOT NULL "
                "DEFAULT 'text/markdown'"
            )
            added.append("mime_type")
        if "binary_data" not in cols:
            self._conn.execute("ALTER TABLE documents ADD COLUMN binary_data BLOB")
            added.append("binary_data")
        if "size_bytes" not in cols:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN size_bytes INTEGER NOT NULL "
                "DEFAULT 0"
            )
            added.append("size_bytes")
        if "version" not in cols:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN version INTEGER NOT NULL "
                "DEFAULT 1"
            )
            added.append("version")
        if added:
            logger.info("Migrated documents table: added %s", ", ".join(added))

    # ── StorageBackend Protocol ───────────────────────────────────

    def read(self, path: str) -> Optional[Document]:
        row = self._conn.execute(
            "SELECT path, title, content, tags, metadata, mime_type, "
            "binary_data, size_bytes, version, updated_at "
            "FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              author: str = "", change_note: str = "") -> bool:
        """Write or update a textual document.

        Automatically creates a version snapshot if track_versions is True.
        """
        now = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        size = len(content.encode("utf-8"))

        try:
            existing = self._conn.execute(
                "SELECT version, created_at FROM documents WHERE path = ?",
                (path,),
            ).fetchone()

            if existing:
                next_version = existing["version"] + 1
                self._conn.execute(
                    "UPDATE documents SET title=?, content=?, tags=?, "
                    "metadata=?, mime_type=?, binary_data=NULL, "
                    "size_bytes=?, version=?, updated_at=? "
                    "WHERE path=?",
                    (title, content, tags_str, meta_str, "text/markdown",
                     size, next_version, now, path),
                )
            else:
                next_version = 1
                self._conn.execute(
                    "INSERT INTO documents "
                    "(path, title, content, tags, metadata, mime_type, "
                    "binary_data, size_bytes, version, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                    (path, title, content, tags_str, meta_str,
                     "text/markdown", size, next_version, now, now),
                )

            if self.track_versions:
                self._snapshot_version(
                    path=path, version=next_version, title=title,
                    content=content, tags_str=tags_str, meta_str=meta_str,
                    mime_type="text/markdown", binary_data=None,
                    size_bytes=size, created_at=now,
                    author=author, change_note=change_note,
                )
            return True
        except sqlite3.Error as e:
            logger.error("SQLite write error for '%s': %s", path, e)
            return False

    def list(self, prefix: str = "") -> List[Document]:
        if prefix:
            rows = self._conn.execute(
                "SELECT path, title, content, tags, metadata, mime_type, "
                "binary_data, size_bytes, version, updated_at "
                "FROM documents WHERE path LIKE ? ORDER BY path",
                (prefix + "%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT path, title, content, tags, metadata, mime_type, "
                "binary_data, size_bytes, version, updated_at "
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
            safe_query = self._escape_fts_query(query)
            rows = self._conn.execute(
                "SELECT d.path, d.title, d.content, d.tags, d.metadata, "
                "d.mime_type, d.binary_data, d.size_bytes, d.version, d.updated_at "
                "FROM documents d "
                "JOIN documents_fts fts ON d.rowid = fts.rowid "
                "WHERE documents_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT 20",
                (safe_query,),
            ).fetchall()
            return [self._row_to_document(r) for r in rows]
        except sqlite3.OperationalError as e:
            logger.debug("FTS query failed ('%s'), falling back to LIKE: %s", query, e)
            return self._search_like(query)

    def delete(self, path: str) -> bool:
        try:
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE path = ?", (path,)
            )
            # Also clean up history and references
            self._conn.execute(
                "DELETE FROM document_versions WHERE path = ?", (path,)
            )
            self._conn.execute(
                "DELETE FROM document_references "
                "WHERE from_path = ? OR to_path = ?",
                (path, path),
            )
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error("SQLite delete error for '%s': %s", path, e)
            return False

    # ── BinaryStorage Protocol ────────────────────────────────────

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     author: str = "", change_note: str = "") -> bool:
        """Write or update a binary document (image, audio, PDF, ...).

        The caption is stored in the content column for context display
        and full-text search. The actual bytes go into binary_data.
        """
        now = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        blob = sqlite3.Binary(data)
        size = len(data)

        try:
            existing = self._conn.execute(
                "SELECT version FROM documents WHERE path = ?", (path,)
            ).fetchone()

            if existing:
                next_version = existing["version"] + 1
                self._conn.execute(
                    "UPDATE documents SET title=?, content=?, tags=?, "
                    "metadata=?, mime_type=?, binary_data=?, "
                    "size_bytes=?, version=?, updated_at=? "
                    "WHERE path=?",
                    (title, caption, tags_str, meta_str, mime_type,
                     blob, size, next_version, now, path),
                )
            else:
                next_version = 1
                self._conn.execute(
                    "INSERT INTO documents "
                    "(path, title, content, tags, metadata, mime_type, "
                    "binary_data, size_bytes, version, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (path, title, caption, tags_str, meta_str, mime_type,
                     blob, size, next_version, now, now),
                )

            if self.track_versions:
                self._snapshot_version(
                    path=path, version=next_version, title=title,
                    content=caption, tags_str=tags_str, meta_str=meta_str,
                    mime_type=mime_type, binary_data=blob,
                    size_bytes=size, created_at=now,
                    author=author, change_note=change_note,
                )
            return True
        except sqlite3.Error as e:
            logger.error("SQLite write_binary error for '%s': %s", path, e)
            return False

    def read_binary(self, path: str) -> Optional[bytes]:
        row = self._conn.execute(
            "SELECT binary_data FROM documents WHERE path = ?", (path,)
        ).fetchone()
        if row is None or row["binary_data"] is None:
            return None
        return bytes(row["binary_data"])

    # ── VersionedStorage Protocol ─────────────────────────────────

    def list_versions(self, path: str) -> List[DocumentVersion]:
        rows = self._conn.execute(
            "SELECT path, version, title, content, tags, metadata, "
            "mime_type, binary_data, size_bytes, created_at, author, change_note "
            "FROM document_versions WHERE path = ? "
            "ORDER BY version DESC",
            (path,),
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        row = self._conn.execute(
            "SELECT path, version, title, content, tags, metadata, "
            "mime_type, binary_data, size_bytes, created_at, author, change_note "
            "FROM document_versions WHERE path = ? AND version = ?",
            (path, version),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def rollback(self, path: str, version: int, author: str = "",
                 change_note: str = "") -> bool:
        target = self.get_version(path, version)
        if target is None:
            return False
        note = change_note or f"Rolled back to v{version}"
        if target.binary_data is not None:
            return self.write_binary(
                path=path, title=target.title, mime_type=target.mime_type,
                data=target.binary_data, caption=target.content,
                tags=list(target.tags), metadata=dict(target.metadata),
                author=author, change_note=note,
            )
        return self.write(
            path=path, title=target.title, content=target.content,
            tags=list(target.tags), metadata=dict(target.metadata),
            author=author, change_note=note,
        )

    def prune_versions(self, path: str, keep_last_n: int) -> int:
        """Drop versions older than the most recent N.

        Returns the number of rows removed.
        """
        if keep_last_n < 1:
            keep_last_n = 1
        # Find the (keep_last_n+1)-th newest version's number, if any
        row = self._conn.execute(
            "SELECT version FROM document_versions WHERE path = ? "
            "ORDER BY version DESC LIMIT 1 OFFSET ?",
            (path, keep_last_n),
        ).fetchone()
        if row is None:
            return 0
        cutoff = row["version"]
        cursor = self._conn.execute(
            "DELETE FROM document_versions WHERE path = ? AND version <= ?",
            (path, cutoff),
        )
        return cursor.rowcount

    # ── ReferenceStorage Protocol ─────────────────────────────────

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        try:
            self._conn.execute(
                "INSERT INTO document_references "
                "(from_path, to_path, ref_type, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (from_path, to_path, ref_type, note, datetime.now().isoformat()),
            )
            return True
        except sqlite3.IntegrityError:
            # Edge already exists — idempotent no-op
            return False
        except sqlite3.Error as e:
            logger.error("SQLite add_reference error: %s", e)
            return False

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        try:
            if ref_type is None:
                cursor = self._conn.execute(
                    "DELETE FROM document_references "
                    "WHERE from_path = ? AND to_path = ?",
                    (from_path, to_path),
                )
            else:
                cursor = self._conn.execute(
                    "DELETE FROM document_references "
                    "WHERE from_path = ? AND to_path = ? AND ref_type = ?",
                    (from_path, to_path, ref_type),
                )
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error("SQLite remove_reference error: %s", e)
            return 0

    def list_references_from(self, path: str) -> List[Reference]:
        rows = self._conn.execute(
            "SELECT from_path, to_path, ref_type, note, created_at "
            "FROM document_references WHERE from_path = ? "
            "ORDER BY created_at",
            (path,),
        ).fetchall()
        return [self._row_to_reference(r) for r in rows]

    def list_references_to(self, path: str) -> List[Reference]:
        rows = self._conn.execute(
            "SELECT from_path, to_path, ref_type, note, created_at "
            "FROM document_references WHERE to_path = ? "
            "ORDER BY created_at",
            (path,),
        ).fetchall()
        return [self._row_to_reference(r) for r in rows]

    def list_all_references(self) -> List[Reference]:
        rows = self._conn.execute(
            "SELECT from_path, to_path, ref_type, note, created_at "
            "FROM document_references ORDER BY created_at"
        ).fetchall()
        return [self._row_to_reference(r) for r in rows]

    # ── Additional Operations ─────────────────────────────────────

    def move(self, old_path: str, new_path: str) -> bool:
        """Move/rename a document. Updates references and version history paths."""
        try:
            cursor = self._conn.execute(
                "UPDATE documents SET path = ?, updated_at = ? WHERE path = ?",
                (new_path, datetime.now().isoformat(), old_path),
            )
            if cursor.rowcount == 0:
                return False
            self._conn.execute(
                "UPDATE document_versions SET path = ? WHERE path = ?",
                (new_path, old_path),
            )
            self._conn.execute(
                "UPDATE document_references SET from_path = ? WHERE from_path = ?",
                (new_path, old_path),
            )
            self._conn.execute(
                "UPDATE document_references SET to_path = ? WHERE to_path = ?",
                (new_path, old_path),
            )
            return True
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
        rows = self._conn.execute(
            "SELECT path, title, content, tags, metadata, mime_type, "
            "binary_data, size_bytes, version, updated_at "
            "FROM documents "
            "WHERE ',' || tags || ',' LIKE ? "
            "ORDER BY path",
            (f"%,{tag},%",),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def list_by_mime_type(self, mime_prefix: str) -> List[Document]:
        """List documents whose mime_type starts with the given prefix.

        e.g. ``list_by_mime_type("image/")`` returns all images.
        """
        rows = self._conn.execute(
            "SELECT path, title, content, tags, metadata, mime_type, "
            "binary_data, size_bytes, version, updated_at "
            "FROM documents "
            "WHERE mime_type LIKE ? "
            "ORDER BY path",
            (mime_prefix + "%",),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def count(self) -> int:
        """Total number of documents."""
        row = self._conn.execute("SELECT COUNT(*) as n FROM documents").fetchone()
        return row["n"]

    def tree(self, prefix: str = "") -> Dict:
        """Build a nested directory tree of all document paths."""
        docs = self.list(prefix)
        tree: Dict[str, Any] = {}
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
        version_count = self._conn.execute(
            "SELECT COUNT(*) as n FROM document_versions"
        ).fetchone()["n"]
        ref_count = self._conn.execute(
            "SELECT COUNT(*) as n FROM document_references"
        ).fetchone()["n"]
        binary_count = self._conn.execute(
            "SELECT COUNT(*) as n FROM documents WHERE binary_data IS NOT NULL"
        ).fetchone()["n"]
        return {
            "documents": doc_count,
            "binary_documents": binary_count,
            "versions": version_count,
            "references": ref_count,
            "tags": len(tags),
            "db_size_bytes": size,
            "db_path": self.db_path,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _snapshot_version(
        self, path: str, version: int, title: str, content: str,
        tags_str: str, meta_str: str, mime_type: str,
        binary_data: Optional[Any], size_bytes: int,
        created_at: str, author: str, change_note: str,
    ) -> None:
        """Insert a row into document_versions."""
        try:
            self._conn.execute(
                "INSERT INTO document_versions "
                "(path, version, title, content, tags, metadata, mime_type, "
                "binary_data, size_bytes, created_at, author, change_note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (path, version, title, content, tags_str, meta_str, mime_type,
                 binary_data, size_bytes, created_at, author, change_note),
            )
        except sqlite3.IntegrityError:
            # (path, version) already snapshotted — keep first writer's record
            logger.debug("Version v%d for '%s' already exists, skipping snapshot",
                         version, path)
        except sqlite3.Error as e:
            logger.error("Snapshot error for '%s' v%d: %s", path, version, e)

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        keys = row.keys()
        tags_str = row["tags"]
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        meta_str = row["metadata"]
        try:
            metadata = json.loads(meta_str) if meta_str and meta_str != "{}" else {}
        except json.JSONDecodeError:
            metadata = {}
        binary_data = None
        if "binary_data" in keys and row["binary_data"] is not None:
            binary_data = bytes(row["binary_data"])
        return Document(
            path=row["path"],
            title=row["title"],
            content=row["content"],
            tags=tags,
            metadata=metadata,
            updated_at=row["updated_at"],
            mime_type=row["mime_type"] if "mime_type" in keys else "text/markdown",
            binary_data=binary_data,
            size_bytes=row["size_bytes"] if "size_bytes" in keys else 0,
            version=row["version"] if "version" in keys else 1,
        )

    def _row_to_version(self, row: sqlite3.Row) -> DocumentVersion:
        keys = row.keys()
        tags_str = row["tags"]
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        meta_str = row["metadata"]
        try:
            metadata = json.loads(meta_str) if meta_str and meta_str != "{}" else {}
        except json.JSONDecodeError:
            metadata = {}
        binary_data = None
        if "binary_data" in keys and row["binary_data"] is not None:
            binary_data = bytes(row["binary_data"])
        return DocumentVersion(
            path=row["path"],
            version=row["version"],
            title=row["title"],
            content=row["content"],
            tags=tags,
            metadata=metadata,
            mime_type=row["mime_type"],
            binary_data=binary_data,
            size_bytes=row["size_bytes"],
            created_at=row["created_at"],
            author=row["author"],
            change_note=row["change_note"],
        )

    def _row_to_reference(self, row: sqlite3.Row) -> Reference:
        return Reference(
            from_path=row["from_path"],
            to_path=row["to_path"],
            ref_type=row["ref_type"],
            note=row["note"],
            created_at=row["created_at"],
        )

    def _escape_fts_query(self, query: str) -> str:
        """Escape a user query for safe FTS5 usage.

        Wraps each word in double quotes to treat them as literals,
        avoiding FTS5 syntax errors from special characters.
        """
        words = query.strip().split()
        if not words:
            return '""'
        return " ".join(f'"{w}"' for w in words)

    def _search_like(self, query: str) -> List[Document]:
        """Fallback search using LIKE when FTS fails."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT path, title, content, tags, metadata, mime_type, "
            "binary_data, size_bytes, version, updated_at "
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

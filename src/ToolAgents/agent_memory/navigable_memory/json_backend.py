"""
JSON Storage Backend for NavigableMemory.

A single-file, human-inspectable, git-friendly knowledge store with no
external dependencies beyond the Python stdlib.

Implements:
    - StorageBackend (read/write/list/search/delete)
    - BinaryStorage (write_binary/read_binary, base64-encoded inline)
    - VersionedStorage (full document_versions history)
    - ReferenceStorage (directed graph between documents)

Layout of the JSON file:
    {
        "schema_version": 1,
        "documents": {
            "<path>": {
                "title": ..., "content": ..., "tags": [...],
                "metadata": {...}, "mime_type": "...",
                "binary_data_b64": "..."  | null,
                "size_bytes": N, "version": N,
                "created_at": "...", "updated_at": "..."
            }
        },
        "versions": {
            "<path>": [ { ...same shape as document, plus author/change_note... } ]
        },
        "references": [
            { "from_path": "...", "to_path": "...",
              "ref_type": "...", "note": "...", "created_at": "..." }
        ]
    }

Trade-offs vs SQLite:
    - Pros: human-readable, gitable, zero deps, easy to inspect/diff/merge.
    - Cons: O(n) search (no FTS), entire file rewritten on every save,
      not safe for high-concurrency multi-process writes.

Best for: single-agent memory at modest scale (hundreds to low thousands
of documents). For larger or concurrent workloads, prefer SQLiteBackend.

Usage:
    from navigable_memory import NavigableMemory
    from json_backend import JSONBackend

    backend = JSONBackend("memory.json")
    memory = NavigableMemory(backend)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from .navigable_memory import (
    Document, DocumentVersion, Reference, RefType,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class JSONBackend:
    """Single-file JSON storage backend.

    Args:
        file_path: Path to the JSON file. Created if it doesn't exist.
            Use ":memory:" for an ephemeral, in-process store with no disk I/O.
        track_versions: If False, writes do not append to version history.
            Default True.
        autosave: If True (default), every mutation flushes to disk.
            Set False for batched updates; call ``save()`` manually.
        indent: JSON indent level for pretty printing. Default 2 (gitable).
            Set to None for compact output.
    """

    def __init__(
        self,
        file_path: str = "memory.json",
        track_versions: bool = True,
        autosave: bool = True,
        indent: Optional[int] = 2,
    ):
        self.file_path = file_path
        self.track_versions = track_versions
        self.autosave = autosave
        self.indent = indent
        self._lock = threading.RLock()

        self._documents: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._references: List[Dict[str, Any]] = []

        self._load()

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        """Load state from disk. Creates an empty store if file is missing."""
        if self.file_path == ":memory:":
            return
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load '%s': %s", self.file_path, e)
            return

        if not isinstance(data, dict):
            logger.warning("'%s' is not a JSON object; ignoring.", self.file_path)
            return

        self._documents = data.get("documents", {}) or {}
        self._versions = data.get("versions", {}) or {}
        self._references = data.get("references", []) or []
        # Future schema migrations would dispatch on data.get("schema_version") here.

    def save(self) -> None:
        """Persist state to disk via atomic temp-file rename."""
        if self.file_path == ":memory:":
            return
        with self._lock:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "documents": self._documents,
                "versions": self._versions,
                "references": self._references,
            }
            parent = os.path.dirname(os.path.abspath(self.file_path))
            os.makedirs(parent, exist_ok=True)
            tmp_path = f"{self.file_path}.tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False,
                              indent=self.indent, sort_keys=True)
                os.replace(tmp_path, self.file_path)
            except OSError as e:
                logger.error("Failed to save '%s': %s", self.file_path, e)
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _encode_binary(data: Optional[bytes]) -> Optional[str]:
        if data is None:
            return None
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _decode_binary(b64: Optional[str]) -> Optional[bytes]:
        if not b64:
            return None
        return base64.b64decode(b64)

    def _doc_dict_to_document(self, path: str, raw: Dict[str, Any]) -> Document:
        return Document(
            path=path,
            title=raw.get("title", "Untitled"),
            content=raw.get("content", ""),
            tags=list(raw.get("tags") or []),
            metadata=dict(raw.get("metadata") or {}),
            updated_at=raw.get("updated_at"),
            mime_type=raw.get("mime_type", "text/markdown"),
            binary_data=self._decode_binary(raw.get("binary_data_b64")),
            size_bytes=int(raw.get("size_bytes", 0)),
            version=int(raw.get("version", 1)),
        )

    def _version_dict_to_object(self, raw: Dict[str, Any]) -> DocumentVersion:
        return DocumentVersion(
            path=raw["path"],
            version=int(raw["version"]),
            title=raw.get("title", "Untitled"),
            content=raw.get("content", ""),
            tags=list(raw.get("tags") or []),
            metadata=dict(raw.get("metadata") or {}),
            mime_type=raw.get("mime_type", "text/markdown"),
            binary_data=self._decode_binary(raw.get("binary_data_b64")),
            size_bytes=int(raw.get("size_bytes", 0)),
            created_at=raw.get("created_at", ""),
            author=raw.get("author", ""),
            change_note=raw.get("change_note", ""),
        )

    def _ref_dict_to_object(self, raw: Dict[str, Any]) -> Reference:
        return Reference(
            from_path=raw["from_path"],
            to_path=raw["to_path"],
            ref_type=raw.get("ref_type", RefType.LINKS_TO),
            note=raw.get("note", ""),
            created_at=raw.get("created_at", ""),
        )

    def _record_version(
        self, path: str, raw_doc: Dict[str, Any],
        author: str, change_note: str,
    ) -> None:
        if not self.track_versions:
            return
        snapshot = dict(raw_doc)  # shallow copy is fine — values are JSON-safe
        snapshot["path"] = path
        snapshot["author"] = author
        snapshot["change_note"] = change_note
        # Use updated_at as the version's created_at
        snapshot["created_at"] = raw_doc.get("updated_at", datetime.now().isoformat())
        self._versions.setdefault(path, []).append(snapshot)

    # ── StorageBackend Protocol ───────────────────────────────────

    def read(self, path: str) -> Optional[Document]:
        with self._lock:
            raw = self._documents.get(path)
            if raw is None:
                return None
            return self._doc_dict_to_document(path, raw)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              author: str = "", change_note: str = "") -> bool:
        now = datetime.now().isoformat()
        with self._lock:
            existing = self._documents.get(path)
            next_version = (existing.get("version", 0) + 1) if existing else 1
            created_at = existing.get("created_at", now) if existing else now
            raw = {
                "title": title,
                "content": content,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "mime_type": "text/markdown",
                "binary_data_b64": None,
                "size_bytes": len(content.encode("utf-8")),
                "version": next_version,
                "created_at": created_at,
                "updated_at": now,
            }
            self._documents[path] = raw
            self._record_version(path, raw, author=author, change_note=change_note)
            self._maybe_save()
            return True

    def list(self, prefix: str = "") -> List[Document]:
        with self._lock:
            return [
                self._doc_dict_to_document(p, raw)
                for p, raw in sorted(self._documents.items())
                if p.startswith(prefix)
            ]

    def search(self, query: str) -> List[Document]:
        """Substring search across title, content, and tags (case-insensitive)."""
        q = query.lower().strip()
        if not q:
            return []
        with self._lock:
            results: List[Document] = []
            for path, raw in sorted(self._documents.items()):
                title = raw.get("title", "").lower()
                content = raw.get("content", "").lower()
                tags = [t.lower() for t in (raw.get("tags") or [])]
                if q in title or q in content or any(q in t for t in tags):
                    results.append(self._doc_dict_to_document(path, raw))
            return results[:20]

    def delete(self, path: str) -> bool:
        with self._lock:
            if path not in self._documents:
                return False
            del self._documents[path]
            self._versions.pop(path, None)
            self._references = [
                r for r in self._references
                if r.get("from_path") != path and r.get("to_path") != path
            ]
            self._maybe_save()
            return True

    # ── BinaryStorage Protocol ────────────────────────────────────

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     author: str = "", change_note: str = "") -> bool:
        now = datetime.now().isoformat()
        with self._lock:
            existing = self._documents.get(path)
            next_version = (existing.get("version", 0) + 1) if existing else 1
            created_at = existing.get("created_at", now) if existing else now
            raw = {
                "title": title,
                "content": caption,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "mime_type": mime_type,
                "binary_data_b64": self._encode_binary(data),
                "size_bytes": len(data),
                "version": next_version,
                "created_at": created_at,
                "updated_at": now,
            }
            self._documents[path] = raw
            self._record_version(path, raw, author=author, change_note=change_note)
            self._maybe_save()
            return True

    def read_binary(self, path: str) -> Optional[bytes]:
        with self._lock:
            raw = self._documents.get(path)
            if raw is None:
                return None
            return self._decode_binary(raw.get("binary_data_b64"))

    # ── VersionedStorage Protocol ─────────────────────────────────

    def list_versions(self, path: str) -> List[DocumentVersion]:
        with self._lock:
            versions = self._versions.get(path, [])
            return [
                self._version_dict_to_object(v)
                for v in sorted(
                    versions, key=lambda v: int(v.get("version", 0)), reverse=True,
                )
            ]

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        with self._lock:
            for v in self._versions.get(path, []):
                if int(v.get("version", -1)) == version:
                    return self._version_dict_to_object(v)
            return None

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
        if keep_last_n < 1:
            keep_last_n = 1
        with self._lock:
            versions = self._versions.get(path, [])
            if len(versions) <= keep_last_n:
                return 0
            ordered = sorted(versions, key=lambda v: int(v.get("version", 0)))
            removed = len(ordered) - keep_last_n
            self._versions[path] = ordered[-keep_last_n:]
            self._maybe_save()
            return removed

    # ── ReferenceStorage Protocol ─────────────────────────────────

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        with self._lock:
            for r in self._references:
                if (r.get("from_path") == from_path
                        and r.get("to_path") == to_path
                        and r.get("ref_type") == ref_type):
                    return False  # idempotent: edge already exists
            self._references.append({
                "from_path": from_path,
                "to_path": to_path,
                "ref_type": ref_type,
                "note": note,
                "created_at": datetime.now().isoformat(),
            })
            self._maybe_save()
            return True

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        with self._lock:
            before = len(self._references)
            self._references = [
                r for r in self._references
                if not (r.get("from_path") == from_path
                        and r.get("to_path") == to_path
                        and (ref_type is None or r.get("ref_type") == ref_type))
            ]
            removed = before - len(self._references)
            if removed:
                self._maybe_save()
            return removed

    def list_references_from(self, path: str) -> List[Reference]:
        with self._lock:
            return [
                self._ref_dict_to_object(r)
                for r in self._references
                if r.get("from_path") == path
            ]

    def list_references_to(self, path: str) -> List[Reference]:
        with self._lock:
            return [
                self._ref_dict_to_object(r)
                for r in self._references
                if r.get("to_path") == path
            ]

    def list_all_references(self) -> List[Reference]:
        with self._lock:
            return [self._ref_dict_to_object(r) for r in self._references]

    # ── Convenience / Inspection ──────────────────────────────────

    def count(self) -> int:
        with self._lock:
            return len(self._documents)

    def list_tags(self) -> List[str]:
        with self._lock:
            tags = set()
            for raw in self._documents.values():
                for t in raw.get("tags") or []:
                    if t:
                        tags.add(t.strip())
            return sorted(tags)

    def list_by_tag(self, tag: str) -> List[Document]:
        with self._lock:
            return [
                self._doc_dict_to_document(p, raw)
                for p, raw in sorted(self._documents.items())
                if tag in (raw.get("tags") or [])
            ]

    def list_by_mime_type(self, mime_prefix: str) -> List[Document]:
        with self._lock:
            return [
                self._doc_dict_to_document(p, raw)
                for p, raw in sorted(self._documents.items())
                if raw.get("mime_type", "").startswith(mime_prefix)
            ]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            binary_count = sum(
                1 for raw in self._documents.values()
                if raw.get("binary_data_b64")
            )
            version_count = sum(len(v) for v in self._versions.values())
            size = (
                os.path.getsize(self.file_path)
                if self.file_path != ":memory:" and os.path.exists(self.file_path)
                else 0
            )
            return {
                "documents": len(self._documents),
                "binary_documents": binary_count,
                "versions": version_count,
                "references": len(self._references),
                "tags": len(self.list_tags()),
                "file_size_bytes": size,
                "file_path": self.file_path,
            }

    def tree(self, prefix: str = "") -> Dict[str, Any]:
        """Build a nested directory tree of all document paths."""
        with self._lock:
            tree: Dict[str, Any] = {}
            for path in sorted(self._documents.keys()):
                if not path.startswith(prefix):
                    continue
                raw = self._documents[path]
                parts = path.split("/")
                node = tree
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = {"title": raw.get("title", ""), "path": path}
            return tree

    def reload(self) -> None:
        """Discard in-memory state and re-read from disk."""
        with self._lock:
            self._documents = {}
            self._versions = {}
            self._references = []
            self._load()

    def close(self) -> None:
        """Flush any pending writes. Safe to call multiple times."""
        if self.autosave:
            return  # Already on disk
        try:
            self.save()
        except OSError:
            pass

    def __repr__(self) -> str:
        return f"JSONBackend('{self.file_path}', {self.count()} docs)"

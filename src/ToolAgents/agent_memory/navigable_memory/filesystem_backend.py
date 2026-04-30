"""
Filesystem Storage Backend for NavigableMemory.

Stores each document as a real file at its logical path so humans can
open, edit, view, and ``git diff`` the agent's memory directly.

Layout under the root directory::

    root/
    ├── docs/
    │   ├── intro.md                  ← pure markdown content
    │   └── architecture.md
    ├── assets/
    │   └── diagrams/
    │       └── arch.png              ← raw bytes (opens in any image viewer)
    └── .navmem/                      ← all metadata machinery lives here
        ├── meta/
        │   ├── docs/intro.md.json    ← per-doc metadata sidecar
        │   └── assets/diagrams/arch.png.json
        ├── versions/
        │   └── docs/intro.md.json    ← array of historical versions
        └── references.json            ← graph edges

Implements StorageBackend, BinaryStorage, VersionedStorage, and
ReferenceStorage. The backend is the sole writer; manual edits to
the document files won't auto-sync into metadata. Use ``rescan()``
to forcibly reload from disk.

Usage::

    backend = FilesystemBackend("./agent_memory")
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

NAVMEM_DIR = ".navmem"
META_SUBDIR = "meta"
VERSIONS_SUBDIR = "versions"
REFERENCES_FILE = "references.json"


class FilesystemBackend:
    """Filesystem-backed storage where each document is a real file.

    Args:
        root_dir: Directory under which all documents and metadata live.
            Created if missing.
        track_versions: If False, writes do not append to version history.
            Default True.
        indent: JSON indent for sidecar/version/reference files. Default 2.
    """

    def __init__(
        self,
        root_dir: str = "agent_memory",
        track_versions: bool = True,
        indent: Optional[int] = 2,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.track_versions = track_versions
        self.indent = indent
        self._lock = threading.RLock()
        self._navmem = os.path.join(self.root_dir, NAVMEM_DIR)
        os.makedirs(os.path.join(self._navmem, META_SUBDIR), exist_ok=True)
        os.makedirs(os.path.join(self._navmem, VERSIONS_SUBDIR), exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────

    def _doc_fs_path(self, path: str) -> str:
        """Filesystem path where the document's payload lives."""
        return os.path.join(self.root_dir, *path.split("/"))

    def _meta_fs_path(self, path: str) -> str:
        """Filesystem path of the metadata sidecar."""
        parts = path.split("/")
        parts[-1] = parts[-1] + ".json"
        return os.path.join(self._navmem, META_SUBDIR, *parts)

    def _versions_fs_path(self, path: str) -> str:
        """Filesystem path of the versions list for a document."""
        parts = path.split("/")
        parts[-1] = parts[-1] + ".json"
        return os.path.join(self._navmem, VERSIONS_SUBDIR, *parts)

    def _references_fs_path(self) -> str:
        return os.path.join(self._navmem, REFERENCES_FILE)

    def _is_under_navmem(self, fs_path: str) -> bool:
        rel = os.path.relpath(fs_path, self.root_dir)
        return rel == NAVMEM_DIR or rel.startswith(NAVMEM_DIR + os.sep)

    def _logical_path(self, fs_path: str) -> str:
        """Convert an absolute filesystem path to a logical doc path."""
        rel = os.path.relpath(fs_path, self.root_dir)
        return rel.replace(os.sep, "/")

    @staticmethod
    def _atomic_write_text(fs_path: str, text: str) -> None:
        os.makedirs(os.path.dirname(fs_path), exist_ok=True)
        tmp = fs_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, fs_path)

    @staticmethod
    def _atomic_write_bytes(fs_path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(fs_path), exist_ok=True)
        tmp = fs_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, fs_path)

    def _read_json(self, fs_path: str) -> Optional[Any]:
        if not os.path.exists(fs_path):
            return None
        try:
            with open(fs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read JSON '%s': %s", fs_path, e)
            return None

    def _write_json(self, fs_path: str, data: Any) -> None:
        text = json.dumps(data, ensure_ascii=False,
                          indent=self.indent, sort_keys=True)
        self._atomic_write_text(fs_path, text)

    # ── Construction helpers ─────────────────────────────────────

    @staticmethod
    def _encode_b64(data: Optional[bytes]) -> Optional[str]:
        return None if data is None else base64.b64encode(data).decode("ascii")

    @staticmethod
    def _decode_b64(s: Optional[str]) -> Optional[bytes]:
        return None if not s else base64.b64decode(s)

    def _build_document(
        self, path: str, meta: Dict[str, Any],
    ) -> Document:
        binary_data: Optional[bytes] = None
        content: str = ""
        mime_type = meta.get("mime_type", "text/markdown")
        fs_path = self._doc_fs_path(path)
        if mime_type.startswith("text/"):
            try:
                with open(fs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError as e:
                logger.error("Failed to read text doc '%s': %s", fs_path, e)
                content = ""
        else:
            # Binary: load bytes lazily? For simplicity load now (small enough
            # for typical agent memory). Caption lives in meta["caption"].
            try:
                with open(fs_path, "rb") as f:
                    binary_data = f.read()
            except OSError as e:
                logger.error("Failed to read binary doc '%s': %s", fs_path, e)
                binary_data = None
            content = meta.get("caption", "")
        return Document(
            path=path,
            title=meta.get("title", "Untitled"),
            content=content,
            tags=list(meta.get("tags") or []),
            metadata=dict(meta.get("metadata") or {}),
            updated_at=meta.get("updated_at"),
            mime_type=mime_type,
            binary_data=binary_data,
            size_bytes=int(meta.get("size_bytes", 0)),
            version=int(meta.get("version", 1)),
        )

    # ── StorageBackend Protocol ───────────────────────────────────

    def read(self, path: str) -> Optional[Document]:
        with self._lock:
            meta = self._read_json(self._meta_fs_path(path))
            if meta is None:
                return None
            return self._build_document(path, meta)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              author: str = "", change_note: str = "") -> bool:
        now = datetime.now().isoformat()
        with self._lock:
            meta_path = self._meta_fs_path(path)
            existing_meta = self._read_json(meta_path) or {}
            next_version = int(existing_meta.get("version", 0)) + 1
            created_at = existing_meta.get("created_at", now)
            size = len(content.encode("utf-8"))
            meta = {
                "title": title,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "mime_type": "text/markdown",
                "size_bytes": size,
                "version": next_version,
                "created_at": created_at,
                "updated_at": now,
            }
            try:
                self._atomic_write_text(self._doc_fs_path(path), content)
                self._write_json(meta_path, meta)
            except OSError as e:
                logger.error("Failed to write text doc '%s': %s", path, e)
                return False
            self._snapshot_version(
                path, version=next_version, title=title, content=content,
                tags=tags or [], metadata=metadata or {},
                mime_type="text/markdown", binary_data=None,
                size_bytes=size, created_at=now,
                author=author, change_note=change_note,
            )
            return True

    def list(self, prefix: str = "") -> List[Document]:
        """Walk the metadata tree and return matching documents."""
        with self._lock:
            results: List[Document] = []
            meta_root = os.path.join(self._navmem, META_SUBDIR)
            if not os.path.isdir(meta_root):
                return []
            for dirpath, _dirnames, filenames in os.walk(meta_root):
                for fname in filenames:
                    if not fname.endswith(".json"):
                        continue
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, meta_root)
                    # rel ends with .json — strip it to recover doc path
                    rel = rel[:-len(".json")]
                    path = rel.replace(os.sep, "/")
                    if not path.startswith(prefix):
                        continue
                    meta = self._read_json(full)
                    if meta is None:
                        continue
                    results.append(self._build_document(path, meta))
            results.sort(key=lambda d: d.path)
            return results

    def search(self, query: str) -> List[Document]:
        q = query.lower().strip()
        if not q:
            return []
        results: List[Document] = []
        for doc in self.list(""):
            haystack = doc.title.lower()
            if not doc.is_binary:
                haystack += "\n" + doc.content.lower()
            if any(q in t.lower() for t in doc.tags) or q in haystack:
                results.append(doc)
            if len(results) >= 20:
                break
        return results

    def delete(self, path: str) -> bool:
        with self._lock:
            meta_path = self._meta_fs_path(path)
            if not os.path.exists(meta_path):
                return False
            doc_path = self._doc_fs_path(path)
            versions_path = self._versions_fs_path(path)
            for p in (doc_path, meta_path, versions_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError as e:
                    logger.warning("Failed to remove '%s': %s", p, e)
            self._cleanup_empty_dirs(os.path.dirname(doc_path))
            self._cleanup_empty_dirs(os.path.dirname(meta_path))
            self._cleanup_empty_dirs(os.path.dirname(versions_path))
            # Drop any references mentioning this path
            refs = self._read_json(self._references_fs_path()) or []
            new_refs = [
                r for r in refs
                if r.get("from_path") != path and r.get("to_path") != path
            ]
            if len(new_refs) != len(refs):
                self._write_json(self._references_fs_path(), new_refs)
            return True

    def _cleanup_empty_dirs(self, start: str) -> None:
        """Remove empty directories upward to (but not including) the root."""
        try:
            current = os.path.abspath(start)
            while current.startswith(self.root_dir) and current != self.root_dir:
                if not os.path.isdir(current):
                    break
                if os.listdir(current):
                    break
                os.rmdir(current)
                current = os.path.dirname(current)
        except OSError:
            pass

    # ── BinaryStorage Protocol ────────────────────────────────────

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     author: str = "", change_note: str = "") -> bool:
        now = datetime.now().isoformat()
        with self._lock:
            meta_path = self._meta_fs_path(path)
            existing_meta = self._read_json(meta_path) or {}
            next_version = int(existing_meta.get("version", 0)) + 1
            created_at = existing_meta.get("created_at", now)
            size = len(data)
            meta = {
                "title": title,
                "caption": caption,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "mime_type": mime_type,
                "size_bytes": size,
                "version": next_version,
                "created_at": created_at,
                "updated_at": now,
            }
            try:
                self._atomic_write_bytes(self._doc_fs_path(path), data)
                self._write_json(meta_path, meta)
            except OSError as e:
                logger.error("Failed to write binary doc '%s': %s", path, e)
                return False
            self._snapshot_version(
                path, version=next_version, title=title, content=caption,
                tags=tags or [], metadata=metadata or {},
                mime_type=mime_type, binary_data=data,
                size_bytes=size, created_at=now,
                author=author, change_note=change_note,
            )
            return True

    def read_binary(self, path: str) -> Optional[bytes]:
        with self._lock:
            meta = self._read_json(self._meta_fs_path(path))
            if meta is None:
                return None
            mime_type = meta.get("mime_type", "text/markdown")
            if mime_type.startswith("text/"):
                return None
            try:
                with open(self._doc_fs_path(path), "rb") as f:
                    return f.read()
            except OSError as e:
                logger.error("Failed to read binary '%s': %s", path, e)
                return None

    # ── VersionedStorage Protocol ─────────────────────────────────

    def _snapshot_version(
        self, path: str, version: int, title: str, content: str,
        tags: List[str], metadata: Dict[str, Any], mime_type: str,
        binary_data: Optional[bytes], size_bytes: int,
        created_at: str, author: str, change_note: str,
    ) -> None:
        if not self.track_versions:
            return
        snap = {
            "path": path,
            "version": version,
            "title": title,
            "content": content,
            "tags": list(tags),
            "metadata": dict(metadata),
            "mime_type": mime_type,
            "binary_data_b64": self._encode_b64(binary_data),
            "size_bytes": size_bytes,
            "created_at": created_at,
            "author": author,
            "change_note": change_note,
        }
        versions_path = self._versions_fs_path(path)
        existing = self._read_json(versions_path) or []
        if not isinstance(existing, list):
            existing = []
        existing.append(snap)
        try:
            self._write_json(versions_path, existing)
        except OSError as e:
            logger.error("Failed to snapshot '%s' v%d: %s", path, version, e)

    def _version_dict_to_object(self, raw: Dict[str, Any]) -> DocumentVersion:
        return DocumentVersion(
            path=raw["path"],
            version=int(raw["version"]),
            title=raw.get("title", "Untitled"),
            content=raw.get("content", ""),
            tags=list(raw.get("tags") or []),
            metadata=dict(raw.get("metadata") or {}),
            mime_type=raw.get("mime_type", "text/markdown"),
            binary_data=self._decode_b64(raw.get("binary_data_b64")),
            size_bytes=int(raw.get("size_bytes", 0)),
            created_at=raw.get("created_at", ""),
            author=raw.get("author", ""),
            change_note=raw.get("change_note", ""),
        )

    def list_versions(self, path: str) -> List[DocumentVersion]:
        with self._lock:
            data = self._read_json(self._versions_fs_path(path)) or []
            if not isinstance(data, list):
                return []
            return [
                self._version_dict_to_object(v)
                for v in sorted(
                    data, key=lambda v: int(v.get("version", 0)), reverse=True,
                )
            ]

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        with self._lock:
            data = self._read_json(self._versions_fs_path(path)) or []
            if not isinstance(data, list):
                return None
            for v in data:
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
            versions_path = self._versions_fs_path(path)
            data = self._read_json(versions_path) or []
            if not isinstance(data, list) or len(data) <= keep_last_n:
                return 0
            ordered = sorted(data, key=lambda v: int(v.get("version", 0)))
            kept = ordered[-keep_last_n:]
            removed = len(ordered) - len(kept)
            try:
                self._write_json(versions_path, kept)
            except OSError as e:
                logger.error("prune_versions '%s' failed: %s", path, e)
                return 0
            return removed

    # ── ReferenceStorage Protocol ─────────────────────────────────

    def _load_references(self) -> List[Dict[str, Any]]:
        data = self._read_json(self._references_fs_path()) or []
        return data if isinstance(data, list) else []

    def _save_references(self, refs: List[Dict[str, Any]]) -> None:
        try:
            self._write_json(self._references_fs_path(), refs)
        except OSError as e:
            logger.error("Failed to write references: %s", e)

    def _ref_dict_to_object(self, raw: Dict[str, Any]) -> Reference:
        return Reference(
            from_path=raw["from_path"],
            to_path=raw["to_path"],
            ref_type=raw.get("ref_type", RefType.LINKS_TO),
            note=raw.get("note", ""),
            created_at=raw.get("created_at", ""),
        )

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        with self._lock:
            refs = self._load_references()
            for r in refs:
                if (r.get("from_path") == from_path
                        and r.get("to_path") == to_path
                        and r.get("ref_type") == ref_type):
                    return False
            refs.append({
                "from_path": from_path,
                "to_path": to_path,
                "ref_type": ref_type,
                "note": note,
                "created_at": datetime.now().isoformat(),
            })
            self._save_references(refs)
            return True

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        with self._lock:
            refs = self._load_references()
            before = len(refs)
            refs = [
                r for r in refs
                if not (r.get("from_path") == from_path
                        and r.get("to_path") == to_path
                        and (ref_type is None or r.get("ref_type") == ref_type))
            ]
            removed = before - len(refs)
            if removed:
                self._save_references(refs)
            return removed

    def list_references_from(self, path: str) -> List[Reference]:
        with self._lock:
            return [
                self._ref_dict_to_object(r)
                for r in self._load_references()
                if r.get("from_path") == path
            ]

    def list_references_to(self, path: str) -> List[Reference]:
        with self._lock:
            return [
                self._ref_dict_to_object(r)
                for r in self._load_references()
                if r.get("to_path") == path
            ]

    def list_all_references(self) -> List[Reference]:
        with self._lock:
            return [self._ref_dict_to_object(r) for r in self._load_references()]

    # ── Convenience / Inspection ──────────────────────────────────

    def count(self) -> int:
        return len(self.list(""))

    def list_tags(self) -> List[str]:
        tags = set()
        for doc in self.list(""):
            for t in doc.tags:
                if t:
                    tags.add(t)
        return sorted(tags)

    def list_by_tag(self, tag: str) -> List[Document]:
        return [d for d in self.list("") if tag in d.tags]

    def list_by_mime_type(self, mime_prefix: str) -> List[Document]:
        return [d for d in self.list("") if d.mime_type.startswith(mime_prefix)]

    def stats(self) -> Dict[str, Any]:
        docs = self.list("")
        binary_count = sum(1 for d in docs if d.is_binary)
        total_versions = 0
        versions_root = os.path.join(self._navmem, VERSIONS_SUBDIR)
        if os.path.isdir(versions_root):
            for dirpath, _dirs, files in os.walk(versions_root):
                for fname in files:
                    if not fname.endswith(".json"):
                        continue
                    data = self._read_json(os.path.join(dirpath, fname)) or []
                    if isinstance(data, list):
                        total_versions += len(data)
        ref_count = len(self._load_references())
        size = self._dir_size(self.root_dir)
        return {
            "documents": len(docs),
            "binary_documents": binary_count,
            "versions": total_versions,
            "references": ref_count,
            "tags": len(self.list_tags()),
            "root_size_bytes": size,
            "root_dir": self.root_dir,
        }

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        try:
            for dirpath, _dirs, files in os.walk(path):
                for f in files:
                    full = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(full)
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def rescan(self) -> None:
        """Currently a no-op (no in-memory cache). Reserved for future caching layer."""
        return None

    def __repr__(self) -> str:
        return f"FilesystemBackend('{self.root_dir}', {self.count()} docs)"

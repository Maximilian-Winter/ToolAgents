"""In-memory NavigableMemory backend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Document, DocumentVersion, Reference, RefType

class InMemoryBackend:
    """Simple dict-based storage backend for testing.

    Documents are stored in memory and lost on restart.
    Implements StorageBackend, BinaryStorage, VersionedStorage,
    and ReferenceStorage protocols. Useful for unit tests and
    quick prototyping.
    """

    def __init__(self, track_versions: bool = True):
        self._docs: Dict[str, Document] = {}
        self._versions: Dict[str, List[DocumentVersion]] = {}
        self._refs: List[Reference] = []
        self._track_versions = track_versions

    # ── Internal helpers ─────────────────────────────────────────

    def _record_version(self, doc: Document, author: str = "",
                        change_note: str = "") -> None:
        if not self._track_versions:
            return
        ver = DocumentVersion(
            path=doc.path, version=doc.version, title=doc.title,
            content=doc.content, tags=list(doc.tags),
            metadata=dict(doc.metadata), mime_type=doc.mime_type,
            binary_data=doc.binary_data, size_bytes=doc.size_bytes,
            created_at=doc.updated_at or datetime.now().isoformat(),
            author=author, change_note=change_note,
        )
        self._versions.setdefault(doc.path, []).append(ver)

    # ── StorageBackend ───────────────────────────────────────────

    def read(self, path: str) -> Optional[Document]:
        return self._docs.get(path)

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              author: str = "", change_note: str = "") -> bool:
        existing = self._docs.get(path)
        next_version = (existing.version + 1) if existing else 1
        size = len(content.encode("utf-8"))
        doc = Document(
            path=path, title=title, content=content,
            tags=tags or [], metadata=metadata or {},
            updated_at=datetime.now().isoformat(),
            mime_type="text/markdown",
            binary_data=None,
            size_bytes=size,
            version=next_version,
        )
        self._docs[path] = doc
        self._record_version(doc, author=author, change_note=change_note)
        return True

    def list(self, prefix: str = "") -> List[Document]:
        return [
            doc for path, doc in sorted(self._docs.items())
            if path.startswith(prefix)
        ]

    def search(self, query: str) -> List[Document]:
        query_lower = query.lower()
        results = []
        for doc in self._docs.values():
            if (query_lower in doc.content.lower()
                    or query_lower in doc.title.lower()
                    or any(query_lower in t.lower() for t in doc.tags)):
                results.append(doc)
        return results

    def delete(self, path: str) -> bool:
        if path in self._docs:
            del self._docs[path]
            self._versions.pop(path, None)
            self._refs = [
                r for r in self._refs
                if r.from_path != path and r.to_path != path
            ]
            return True
        return False

    # ── BinaryStorage ────────────────────────────────────────────

    def write_binary(self, path: str, title: str, mime_type: str,
                     data: bytes, caption: str = "",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     author: str = "", change_note: str = "") -> bool:
        existing = self._docs.get(path)
        next_version = (existing.version + 1) if existing else 1
        doc = Document(
            path=path, title=title, content=caption,
            tags=tags or [], metadata=metadata or {},
            updated_at=datetime.now().isoformat(),
            mime_type=mime_type,
            binary_data=bytes(data),
            size_bytes=len(data),
            version=next_version,
        )
        self._docs[path] = doc
        self._record_version(doc, author=author, change_note=change_note)
        return True

    def read_binary(self, path: str) -> Optional[bytes]:
        doc = self._docs.get(path)
        return doc.binary_data if doc else None

    # ── VersionedStorage ─────────────────────────────────────────

    def list_versions(self, path: str) -> List[DocumentVersion]:
        return list(reversed(self._versions.get(path, [])))

    def get_version(self, path: str, version: int) -> Optional[DocumentVersion]:
        for ver in self._versions.get(path, []):
            if ver.version == version:
                return ver
        return None

    def rollback(self, path: str, version: int, author: str = "",
                 change_note: str = "") -> bool:
        target = self.get_version(path, version)
        if target is None:
            return False
        existing = self._docs.get(path)
        next_version = (existing.version + 1) if existing else 1
        note = change_note or f"Rolled back to v{version}"
        doc = Document(
            path=path, title=target.title, content=target.content,
            tags=list(target.tags), metadata=dict(target.metadata),
            updated_at=datetime.now().isoformat(),
            mime_type=target.mime_type,
            binary_data=target.binary_data,
            size_bytes=target.size_bytes,
            version=next_version,
        )
        self._docs[path] = doc
        self._record_version(doc, author=author, change_note=note)
        return True

    def prune_versions(self, path: str, keep_last_n: int) -> int:
        versions = self._versions.get(path, [])
        if len(versions) <= keep_last_n:
            return 0
        removed = len(versions) - keep_last_n
        self._versions[path] = versions[-keep_last_n:]
        return removed

    # ── ReferenceStorage ─────────────────────────────────────────

    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        # Idempotent: skip duplicate triples
        for r in self._refs:
            if (r.from_path == from_path and r.to_path == to_path
                    and r.ref_type == ref_type):
                return False
        self._refs.append(Reference(
            from_path=from_path, to_path=to_path,
            ref_type=ref_type, note=note,
            created_at=datetime.now().isoformat(),
        ))
        return True

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        before = len(self._refs)
        self._refs = [
            r for r in self._refs
            if not (r.from_path == from_path and r.to_path == to_path
                    and (ref_type is None or r.ref_type == ref_type))
        ]
        return before - len(self._refs)

    def list_references_from(self, path: str) -> List[Reference]:
        return [r for r in self._refs if r.from_path == path]

    def list_references_to(self, path: str) -> List[Reference]:
        return [r for r in self._refs if r.to_path == path]

    def list_all_references(self) -> List[Reference]:
        return list(self._refs)

    # ── TagStorage ───────────────────────────────────────────────

    def list_tags(self) -> List[str]:
        all_tags: set = set()
        for doc in self._docs.values():
            for t in doc.tags:
                if t:
                    all_tags.add(t)
        return sorted(all_tags)

    def list_by_tag(self, tag: str) -> List[Document]:
        return sorted(
            (d for d in self._docs.values() if tag in d.tags),
            key=lambda d: d.path,
        )

    def find_by_tags(self, tags: List[str], mode: str = "any") -> List[Document]:
        if not tags:
            return []
        target = set(tags)
        results: List[Document] = []
        for doc in self._docs.values():
            doc_tags = set(doc.tags)
            if mode == "all" and target.issubset(doc_tags):
                results.append(doc)
            elif mode == "none" and target.isdisjoint(doc_tags):
                results.append(doc)
            elif mode == "any" and target & doc_tags:
                results.append(doc)
        results.sort(key=lambda d: d.path)
        return results

    def set_tags(self, path: str, tags: List[str]) -> bool:
        existing = self._docs.get(path)
        if existing is None:
            return False
        # Document is frozen — replace with a copy
        self._docs[path] = Document(
            path=existing.path,
            title=existing.title,
            content=existing.content,
            tags=list(tags),
            metadata=dict(existing.metadata),
            updated_at=datetime.now().isoformat(),
            mime_type=existing.mime_type,
            binary_data=existing.binary_data,
            size_bytes=existing.size_bytes,
            version=existing.version,
        )
        return True

    @property
    def document_count(self) -> int:
        return len(self._docs)

"""
Cross-backend migration for NavigableMemory.

Move documents, version histories, binary blobs, and reference graphs
between any pair of backends — InMemory, JSON, SQLite, Filesystem, or
any custom backend that implements the relevant protocols.

Typical use::

    from navigable_memory import (
        JSONBackend, SQLiteBackend, migrate, MigrationReport,
    )

    src = JSONBackend("agent_memory.json")
    dst = SQLiteBackend("agent_memory.db")
    report = migrate(src, dst)
    print(report)
    # MigrationReport(documents=12, binaries=3, versions=27, references=5, ...)

Capability detection is automatic: if the destination doesn't implement
``BinaryStorage`` then binary docs become text docs with their caption
as content. If versioning isn't supported on the destination, only the
current state is copied. Same for references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .navigable_memory import (
    BinaryStorage, Document, ReferenceStorage, StorageBackend,
    VersionedStorage,
)

logger = logging.getLogger(__name__)


@dataclass
class MigrationReport:
    """Summary of a migration run."""
    documents: int = 0
    binaries: int = 0
    versions: int = 0
    references: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        return (
            f"MigrationReport(documents={self.documents}, "
            f"binaries={self.binaries}, versions={self.versions}, "
            f"references={self.references}, skipped={self.skipped}, "
            f"errors={len(self.errors)})"
        )


def migrate(
    src: StorageBackend,
    dst: StorageBackend,
    *,
    include_versions: bool = True,
    include_references: bool = True,
    include_binaries: bool = True,
    overwrite: bool = False,
    prefix: str = "",
) -> MigrationReport:
    """Copy a backend's contents into another backend.

    Args:
        src: Source backend (where data is read from).
        dst: Destination backend (where data is written to).
        include_versions: If True and both backends support versioning,
            replay the source's version history on the destination so
            chronological order is preserved. Default True.
        include_references: If True and both support references, copy
            the reference graph. Default True.
        include_binaries: If True and both support binaries, copy
            binary documents as binaries; otherwise as text fallback.
            Default True.
        overwrite: If True, delete each destination doc (and its history)
            before re-importing. If False, skip docs that already exist
            on the destination. Default False.
        prefix: Only migrate documents whose path starts with this prefix.
            Default "" (everything).

    Returns:
        A ``MigrationReport`` with counts and any error messages.
    """
    report = MigrationReport()

    src_supports_versions = isinstance(src, VersionedStorage)
    dst_supports_versions = isinstance(dst, VersionedStorage)
    src_supports_binary = isinstance(src, BinaryStorage)
    dst_supports_binary = isinstance(dst, BinaryStorage)
    src_supports_refs = isinstance(src, ReferenceStorage)
    dst_supports_refs = isinstance(dst, ReferenceStorage)

    do_versions = include_versions and src_supports_versions and dst_supports_versions
    do_binaries = include_binaries and src_supports_binary and dst_supports_binary
    do_references = include_references and src_supports_refs and dst_supports_refs

    logger.info(
        "Migration plan: versions=%s, binaries=%s, references=%s",
        do_versions, do_binaries, do_references,
    )

    docs = src.list(prefix)
    for doc in docs:
        try:
            _migrate_document(
                doc=doc, src=src, dst=dst,
                do_versions=do_versions,
                do_binaries=do_binaries,
                src_supports_binary=src_supports_binary,
                src_supports_versions=src_supports_versions,
                overwrite=overwrite, report=report,
            )
        except Exception as e:  # pragma: no cover - defensive
            msg = f"document '{doc.path}': {e}"
            logger.exception("Migration error for %s", doc.path)
            report.errors.append(msg)

    if do_references:
        try:
            _migrate_references(src, dst, prefix=prefix, report=report)
        except Exception as e:  # pragma: no cover
            msg = f"references: {e}"
            logger.exception("Migration error for references")
            report.errors.append(msg)

    return report


# ── Internal helpers ─────────────────────────────────────────────


def _migrate_document(
    *,
    doc: Document,
    src: StorageBackend,
    dst: StorageBackend,
    do_versions: bool,
    do_binaries: bool,
    src_supports_binary: bool,
    src_supports_versions: bool,
    overwrite: bool,
    report: MigrationReport,
) -> None:
    existing = dst.read(doc.path)
    if existing is not None:
        if not overwrite:
            report.skipped += 1
            return
        dst.delete(doc.path)

    if do_versions and src_supports_versions:
        # Replay history chronologically (oldest first).
        versions = list(reversed(src.list_versions(doc.path)))  # type: ignore[attr-defined]
        if not versions:
            _write_one(doc=doc, src=src, dst=dst,
                       do_binaries=do_binaries,
                       src_supports_binary=src_supports_binary,
                       report=report)
            return
        for ver in versions:
            _write_version_to_dst(
                ver=ver, dst=dst,
                do_binaries=do_binaries, report=report,
            )
        # Tally the document write itself separately from the bulk version count
        if versions[-1].binary_data is not None and do_binaries:
            report.binaries += 1
        else:
            report.documents += 1
        # Versions counted include all replayed snapshots
        report.versions += len(versions)
    else:
        _write_one(doc=doc, src=src, dst=dst,
                   do_binaries=do_binaries,
                   src_supports_binary=src_supports_binary,
                   report=report)


def _write_one(
    *,
    doc: Document,
    src: StorageBackend,
    dst: StorageBackend,
    do_binaries: bool,
    src_supports_binary: bool,
    report: MigrationReport,
) -> None:
    """Write a single document (latest state only) to dst."""
    if doc.is_binary and do_binaries:
        # Source supports binary too — get raw bytes
        data: Optional[bytes] = doc.binary_data
        if data is None and src_supports_binary:
            data = src.read_binary(doc.path)  # type: ignore[attr-defined]
        if data is None:
            report.errors.append(
                f"binary missing for '{doc.path}' (source returned no bytes)"
            )
            return
        dst.write_binary(  # type: ignore[attr-defined]
            path=doc.path, title=doc.title, mime_type=doc.mime_type,
            data=data, caption=doc.content,
            tags=list(doc.tags), metadata=dict(doc.metadata),
            change_note="migrated",
        )
        report.binaries += 1
    elif doc.is_binary and not do_binaries:
        # Fallback: write as text doc with placeholder body
        body = doc.content or f"[binary {doc.mime_type}, {doc.size_bytes} bytes — not migrated]"
        _safe_write(dst, path=doc.path, title=doc.title, content=body,
                    tags=list(doc.tags), metadata=dict(doc.metadata),
                    change_note="migrated (binary downgraded to text)")
        report.documents += 1
    else:
        _safe_write(dst, path=doc.path, title=doc.title, content=doc.content,
                    tags=list(doc.tags), metadata=dict(doc.metadata),
                    change_note="migrated")
        report.documents += 1


def _write_version_to_dst(
    *, ver, dst: StorageBackend, do_binaries: bool, report: MigrationReport,
) -> None:
    """Replay a single historical version onto dst (creates a new version row)."""
    if ver.binary_data is not None and do_binaries:
        dst.write_binary(  # type: ignore[attr-defined]
            path=ver.path, title=ver.title, mime_type=ver.mime_type,
            data=ver.binary_data, caption=ver.content,
            tags=list(ver.tags), metadata=dict(ver.metadata),
            author=ver.author,
            change_note=(ver.change_note or f"migrated v{ver.version}"),
        )
    elif ver.binary_data is not None and not do_binaries:
        body = ver.content or f"[binary {ver.mime_type}, {ver.size_bytes} bytes]"
        _safe_write(dst, path=ver.path, title=ver.title, content=body,
                    tags=list(ver.tags), metadata=dict(ver.metadata),
                    author=ver.author,
                    change_note=(ver.change_note or f"migrated v{ver.version}"))
    else:
        _safe_write(dst, path=ver.path, title=ver.title, content=ver.content,
                    tags=list(ver.tags), metadata=dict(ver.metadata),
                    author=ver.author,
                    change_note=(ver.change_note or f"migrated v{ver.version}"))


def _safe_write(dst: StorageBackend, **kwargs: Any) -> bool:
    """Call dst.write with extended kwargs; fall back if backend doesn't accept them."""
    try:
        return dst.write(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        # Backend uses the minimal Protocol signature
        return dst.write(
            path=kwargs["path"], title=kwargs["title"],
            content=kwargs["content"],
            tags=kwargs.get("tags"), metadata=kwargs.get("metadata"),
        )


def _migrate_references(
    src: StorageBackend, dst: StorageBackend,
    *, prefix: str, report: MigrationReport,
) -> None:
    refs = src.list_all_references()  # type: ignore[attr-defined]
    for r in refs:
        if prefix and not (r.from_path.startswith(prefix)
                           or r.to_path.startswith(prefix)):
            continue
        ok = dst.add_reference(  # type: ignore[attr-defined]
            r.from_path, r.to_path, r.ref_type, r.note,
        )
        if ok:
            report.references += 1
        else:
            # Idempotent no-op (already exists) — count as skipped
            report.skipped += 1

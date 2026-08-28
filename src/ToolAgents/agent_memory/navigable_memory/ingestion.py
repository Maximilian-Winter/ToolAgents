"""File and directory ingestion helpers for NavigableMemory."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from .navigable_memory import NavigableMemory


DEFAULT_TEXT_EXTENSIONS: Tuple[str, ...] = (
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".xml",
    ".sql",
)


@dataclass
class FileIngestionConfig:
    """Configuration for text-file ingestion."""

    path_prefix: str = ""
    extensions: Optional[Tuple[str, ...]] = DEFAULT_TEXT_EXTENSIONS
    recursive: bool = True
    include_hidden: bool = False
    follow_symlinks: bool = False
    overwrite: bool = True
    encoding: str = "utf-8"
    errors: str = "replace"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    title_from_heading: bool = True
    max_file_size_bytes: Optional[int] = 2_000_000
    excluded_dir_names: Tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
    )


@dataclass(frozen=True)
class IngestionSource:
    """Text ready to be written into NavigableMemory."""

    source_path: Path
    memory_path: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    """Result for one ingested or skipped file."""

    source_path: str
    memory_path: str
    title: str = ""
    status: str = "skipped"
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"written", "skipped"}


@dataclass
class IngestionReport:
    """Aggregate ingestion result."""

    scanned: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[IngestionResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def add(self, result: IngestionResult) -> None:
        self.results.append(result)
        self.scanned += 1
        if result.status == "written":
            self.written += 1
        elif result.status == "failed":
            self.failed += 1
        else:
            self.skipped += 1

    def summary(self) -> str:
        return (
            f"scanned={self.scanned}, written={self.written}, "
            f"skipped={self.skipped}, failed={self.failed}"
        )


IngestionTransformer = Callable[[IngestionSource], IngestionSource | Dict[str, Any] | str]


def ingest_file(
    memory: "NavigableMemory",
    file_path: str | Path,
    *,
    memory_path: Optional[str] = None,
    title: Optional[str] = None,
    config: Optional[FileIngestionConfig] = None,
    transform: Optional[IngestionTransformer] = None,
    author: str = "",
    change_note: str = "ingested from file",
) -> IngestionResult:
    """Ingest one text file into NavigableMemory."""
    config = config or FileIngestionConfig()
    source_path = Path(file_path)
    if not source_path.is_file():
        return IngestionResult(str(source_path), memory_path or "", status="failed", error="not a file")
    if not _is_allowed_file(source_path, config):
        return IngestionResult(str(source_path), memory_path or "", status="skipped", error="extension or visibility filter")

    target_path = normalize_memory_path(
        memory_path or source_path.name,
        config.path_prefix,
    )
    return _ingest_one(
        memory,
        source_path,
        target_path,
        title=title,
        config=config,
        transform=transform,
        author=author,
        change_note=change_note,
    )


def ingest_directory(
    memory: "NavigableMemory",
    directory_path: str | Path,
    *,
    config: Optional[FileIngestionConfig] = None,
    transform: Optional[IngestionTransformer] = None,
    author: str = "",
    change_note: str = "ingested from directory",
) -> IngestionReport:
    """Ingest text files from a directory into NavigableMemory."""
    config = config or FileIngestionConfig()
    root = Path(directory_path)
    report = IngestionReport()
    if not root.is_dir():
        report.add(IngestionResult(str(root), "", status="failed", error="not a directory"))
        return report

    for source_path in _iter_text_files(root, config):
        rel_path = source_path.relative_to(root).as_posix()
        memory_path = normalize_memory_path(rel_path, config.path_prefix)
        report.add(
            _ingest_one(
                memory,
                source_path,
                memory_path,
                title=None,
                config=config,
                transform=transform,
                author=author,
                change_note=change_note,
            )
        )
    return report


def create_llm_ingestion_transformer(
    agent: Any,
    *,
    settings: Any = None,
    prompt_template: Optional[str] = None,
) -> IngestionTransformer:
    """Create a transformer that asks an LLM agent to rewrite file content.

    The transformer keeps the original memory path, title, tags, and metadata.
    The model response becomes the stored document content.
    """
    template = prompt_template or (
        "Prepare this file for a navigable memory knowledge base.\n\n"
        "Path: {memory_path}\n"
        "Title: {title}\n\n"
        "Keep important details, preserve useful structure, and remove only "
        "noise that would not help future retrieval.\n\n"
        "<file_content>\n{content}\n</file_content>"
    )

    def transform(source: IngestionSource) -> IngestionSource:
        from ToolAgents.data_models.messages import ChatMessage

        prompt = template.format(
            source_path=source.source_path,
            memory_path=source.memory_path,
            title=source.title,
            content=source.content,
        )
        response = agent.get_response(
            [ChatMessage.create_user_message(prompt)],
            settings=settings,
        )
        content = getattr(response, "response", str(response))
        return dataclasses.replace(source, content=content)

    return transform


def create_ingestion_tools(
    memory: "NavigableMemory",
    *,
    config: Optional[FileIngestionConfig] = None,
    transform: Optional[IngestionTransformer] = None,
    allowed_root: Optional[str | Path] = None,
) -> list:
    """Create opt-in Pydantic tools for file/directory ingestion.

    ``allowed_root`` is strongly recommended when exposing these tools to an
    agent. If provided, all input paths must resolve inside that directory.
    """
    from pydantic import BaseModel, Field

    base_config = config or FileIngestionConfig()
    root = Path(allowed_root).resolve() if allowed_root is not None else None

    class IngestTextFile(BaseModel):
        """Ingest one local text file into navigable memory."""

        file_path: str = Field(..., description="Local file path to ingest.")
        memory_path: Optional[str] = Field(
            None,
            description="Optional target memory path. Defaults to the file name.",
        )
        title: Optional[str] = Field(
            None,
            description="Optional document title. Defaults to heading or file stem.",
        )
        tags: List[str] = Field(
            default_factory=list,
            description="Additional tags to add to this document.",
        )
        overwrite: Optional[bool] = Field(
            None,
            description="Override whether existing memory paths may be overwritten.",
        )

        def run(self) -> str:
            try:
                source_path = _resolve_allowed_path(self.file_path, root)
            except ValueError as exc:
                return str(exc)
            cfg = _copy_config(
                base_config,
                overwrite=self.overwrite,
                tags=list(dict.fromkeys(base_config.tags + self.tags)),
            )
            result = ingest_file(
                memory,
                source_path,
                memory_path=self.memory_path,
                title=self.title,
                config=cfg,
                transform=transform,
            )
            return _format_single_result(result)

    class IngestTextDirectory(BaseModel):
        """Ingest local text files from a directory into navigable memory."""

        directory_path: str = Field(..., description="Local directory path to ingest.")
        memory_prefix: Optional[str] = Field(
            None,
            description="Optional memory path prefix for imported files.",
        )
        recursive: Optional[bool] = Field(
            None,
            description="Override recursive directory traversal.",
        )
        extensions: Optional[List[str]] = Field(
            None,
            description="Optional extension allow-list, e.g. ['.md', '.txt'].",
        )
        tags: List[str] = Field(
            default_factory=list,
            description="Additional tags to add to imported documents.",
        )
        overwrite: Optional[bool] = Field(
            None,
            description="Override whether existing memory paths may be overwritten.",
        )

        def run(self) -> str:
            try:
                directory = _resolve_allowed_path(self.directory_path, root)
            except ValueError as exc:
                return str(exc)
            cfg = _copy_config(
                base_config,
                path_prefix=self.memory_prefix,
                recursive=self.recursive,
                extensions=tuple(self.extensions) if self.extensions is not None else None,
                overwrite=self.overwrite,
                tags=list(dict.fromkeys(base_config.tags + self.tags)),
            )
            report = ingest_directory(
                memory,
                directory,
                config=cfg,
                transform=transform,
            )
            return _format_report(report)

    class RebuildNavigableSemanticIndex(BaseModel):
        """Rebuild the attached semantic index from current memory documents."""

        path_prefix: str = Field(
            "",
            description="Only rebuild documents under this memory path prefix.",
        )

        def run(self) -> str:
            index = getattr(memory, "semantic_index", None)
            if index is None:
                return "No semantic index is attached."
            indexed = index.rebuild(memory=memory, prefix=self.path_prefix)
            return f"Semantic index rebuilt: {indexed} document(s) indexed."

    return [IngestTextFile, IngestTextDirectory, RebuildNavigableSemanticIndex]


def build_navigable_memory_skill_prompt(
    *,
    include_ingestion: bool = False,
    include_semantic: bool = True,
    include_versions: bool = True,
    include_references: bool = True,
) -> str:
    """Return concise system-prompt guidance for navigable-memory tools."""
    lines = [
        "You have access to a navigable memory knowledge base.",
        "Use list/search tools to discover relevant paths, then navigate or read source documents for evidence.",
        "Treat document paths as stable source identifiers and cite paths when answering from memory.",
        "Prefer navigation when a location's surrounding context, nearby documents, or references may matter.",
    ]
    if include_semantic:
        lines.append(
            "Use semantic or hybrid search for conceptual queries; use lexical search for exact names, terms, and paths."
        )
    if include_references:
        lines.append(
            "Follow references and backlinks when dependencies, related work, provenance, or blockers matter."
        )
    if include_versions:
        lines.append(
            "Use version tools before rollback or when the user asks what changed or what an older document said."
        )
    if include_ingestion:
        lines.append(
            "Use ingestion tools only for user-approved local files or directories, and respect any allowed-root restriction."
        )
    lines.append(
        "Do not invent unseen documents; if search/navigation cannot find evidence, say what was checked."
    )
    return "\n".join(f"- {line}" for line in lines)


def normalize_memory_path(path: str, path_prefix: str = "") -> str:
    """Normalize a local-style path into a slash-separated memory path."""
    normalized = str(path).replace("\\", "/").strip("/")
    prefix = str(path_prefix or "").replace("\\", "/").strip("/")
    if prefix:
        return f"{prefix}/{normalized}" if normalized else prefix
    return normalized


def _ingest_one(
    memory: "NavigableMemory",
    source_path: Path,
    memory_path: str,
    *,
    title: Optional[str],
    config: FileIngestionConfig,
    transform: Optional[IngestionTransformer],
    author: str,
    change_note: str,
) -> IngestionResult:
    if not config.overwrite and memory.read(memory_path) is not None:
        return IngestionResult(
            str(source_path),
            memory_path,
            title or _default_title(source_path, ""),
            status="skipped",
            error="already exists",
        )
    try:
        content = source_path.read_text(encoding=config.encoding, errors=config.errors)
        doc_title = title or _default_title(source_path, content, config.title_from_heading)
        metadata = dict(config.metadata)
        metadata.update({
            "source_path": str(source_path),
            "source_name": source_path.name,
            "source_suffix": source_path.suffix,
        })
        source = IngestionSource(
            source_path=source_path,
            memory_path=memory_path,
            title=doc_title,
            content=content,
            tags=list(config.tags),
            metadata=metadata,
        )
        source = _apply_transform(source, transform)
        ok = memory.write(
            source.memory_path,
            source.title,
            source.content,
            tags=source.tags,
            metadata=source.metadata,
            author=author,
            change_note=change_note,
        )
        if not ok:
            return IngestionResult(
                str(source_path), source.memory_path, source.title,
                status="failed", error="memory write returned false",
            )
        return IngestionResult(
            str(source_path), source.memory_path, source.title, status="written",
        )
    except Exception as exc:
        return IngestionResult(
            str(source_path), memory_path, title or "", status="failed", error=str(exc),
        )


def _apply_transform(
    source: IngestionSource,
    transform: Optional[IngestionTransformer],
) -> IngestionSource:
    if transform is None:
        return source
    transformed = transform(source)
    if isinstance(transformed, IngestionSource):
        return transformed
    if isinstance(transformed, str):
        return dataclasses.replace(source, content=transformed)
    if isinstance(transformed, dict):
        allowed = {"source_path", "memory_path", "title", "content", "tags", "metadata"}
        updates = {key: value for key, value in transformed.items() if key in allowed}
        return dataclasses.replace(source, **updates)
    raise TypeError("ingestion transform must return IngestionSource, dict, or str")


def _iter_text_files(root: Path, config: FileIngestionConfig) -> Iterable[Path]:
    if config.recursive:
        walker = os.walk(root, followlinks=config.follow_symlinks)
    else:
        walker = [(root, [], [path.name for path in root.iterdir() if path.is_file()])]

    for current_root, dir_names, file_names in walker:
        current = Path(current_root)
        dir_names[:] = sorted(
            name for name in dir_names
            if name not in config.excluded_dir_names
            and (config.include_hidden or not name.startswith("."))
        )
        for file_name in sorted(file_names):
            path = current / file_name
            if not config.follow_symlinks and path.is_symlink():
                continue
            if _is_in_excluded_dir(path, root, config):
                continue
            if _is_allowed_file(path, config):
                yield path


def _is_allowed_file(path: Path, config: FileIngestionConfig) -> bool:
    if not config.include_hidden and path.name.startswith("."):
        return False
    if not config.follow_symlinks and path.is_symlink():
        return False
    if config.extensions is not None and path.suffix.lower() not in {
        ext.lower() for ext in config.extensions
    }:
        return False
    if config.max_file_size_bytes is not None:
        try:
            if path.stat().st_size > config.max_file_size_bytes:
                return False
        except OSError:
            return False
    return True


def _is_in_excluded_dir(path: Path, root: Path, config: FileIngestionConfig) -> bool:
    try:
        rel_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return False
    return any(part in config.excluded_dir_names for part in rel_parts)


def _default_title(
    path: Path,
    content: str,
    title_from_heading: bool = True,
) -> str:
    if title_from_heading:
        for line in content.splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                if title:
                    return title
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name


def _copy_config(config: FileIngestionConfig, **overrides: Any) -> FileIngestionConfig:
    updates = {
        key: value for key, value in overrides.items()
        if value is not None
    }
    return dataclasses.replace(config, **updates)


def _resolve_allowed_path(path: str | Path, allowed_root: Optional[Path]) -> Path:
    resolved = Path(path).resolve()
    if allowed_root is None:
        return resolved
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Path '{resolved}' is outside allowed root '{allowed_root}'.") from exc
    return resolved


def _format_single_result(result: IngestionResult) -> str:
    if result.status == "written":
        return f"Ingested '{result.source_path}' as '{result.memory_path}' ({result.title})."
    if result.status == "skipped":
        return f"Skipped '{result.source_path}' ({result.error})."
    return f"Failed to ingest '{result.source_path}': {result.error}"


def _format_report(report: IngestionReport) -> str:
    lines = [f"Ingestion complete: {report.summary()}"]
    for result in report.results[:12]:
        lines.append(f"  - {result.status}: {result.memory_path or result.source_path}")
    if len(report.results) > 12:
        lines.append(f"  ... {len(report.results) - 12} more")
    return "\n".join(lines)

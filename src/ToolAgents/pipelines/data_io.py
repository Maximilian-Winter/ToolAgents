"""Sources and sinks: where a pipeline's input comes from and its output goes.

A source loads data — a file, a list of files, a folder — optionally splits it
with a :mod:`~ToolAgents.pipelines.chunking` splitter, and writes the result
into the ``inputs`` section. A sink takes a value out of the results and emits
it: to a file, to a stream, or to an HTTP endpoint.

Both are ordinary processes, so they compose with everything else. Writing one
file per item is a sink inside a
:class:`~ToolAgents.pipelines.flow.MapProcess`; writing only on success is a
sink inside a :class:`~ToolAgents.pipelines.flow.ConditionalProcess`. Neither
needed new machinery.

Trust
-----

Reading is enabled by default; **writing files and making HTTP requests are
not**. A pipeline document that reads is a privacy question, but one that
writes destroys data and one that POSTs exfiltrates it, so those require
``allow_writes=True`` at load time.

The gate is on *loading a document that writes*, not on writing Python that
writes: a sink constructed in code defaults to permitted, because writing the
code is the intent. Only :meth:`Pipeline.from_dict` defaults to denied.
"""

from __future__ import annotations

import abc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ToolAgents.agents.base_llm_agent import BaseToolAgent
from ToolAgents.pipelines.chunking import build_splitter, splitter_config_to_dict
from ToolAgents.pipelines.pipeline import (
    PipelineExecutionError,
    PipelineLoadContext,
    PipelineSerializationError,
    PipelineToolRegistry,
    Process,
    register_process_type,
)
from ToolAgents.pipelines.results import PipelineResults
from ToolAgents.utilities.message_template import MessageTemplate

__all__ = [
    "FileSink",
    "FileSource",
    "FilesSink",
    "FilesSource",
    "FolderSource",
    "HttpSink",
    "Sink",
    "SinkProcess",
    "Source",
    "SourceProcess",
    "StreamSink",
    "TextSource",
    "register_sink_type",
    "register_source_type",
    "sink_from_config",
    "source_from_config",
]

#: Guard against a mistyped glob pulling in a whole filesystem.
DEFAULT_MAX_FILES = 1000


def normalize_placeholders(template: str) -> str:
    r"""Repair placeholders whose separator was rewritten by ``pathlib``.

    ``Path("out") / "{vars/index}.md"`` becomes ``out\{vars\index}.md`` on
    Windows: joining a path normalizes every ``/``, including the one inside
    the placeholder, which then no longer resolves. Since no section or key
    name contains a backslash, one inside braces is always this accident.
    Only the inside of ``{...}`` is touched, so real Windows paths around it
    are left alone.
    """

    return re.sub(
        r"\{([^{}]*)\}",
        lambda match: "{" + match.group(1).replace("\\", "/") + "}",
        template,
    )


def render_path(template: str, results: Mapping[str, Any]) -> str:
    """Fill ``{section/key}`` placeholders in a path or URL.

    Unresolved placeholders are an error rather than a silent gap: a path with
    a hole in it would write to the wrong place, which is exactly the mistake
    worth being loud about.
    """

    template = normalize_placeholders(template)
    rendered = MessageTemplate.from_string(template).generate_message_content(
        results, remove_empty_template_field=False
    )
    if "{" in rendered and "}" in rendered:
        raise PipelineExecutionError(
            f"Path {template!r} still contains an unresolved placeholder after "
            f"substitution: {rendered!r}. Refusing to use it."
        )
    return rendered


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class Source(abc.ABC):
    """Loads data into a pipeline."""

    #: Value written to, and dispatched on, the JSON ``type`` field.
    source_type: str = ""

    #: Whether this source yields a single string (rather than a list of
    #: records). Splitting a single-value source produces a list of chunks.
    yields_text: bool = False

    @abc.abstractmethod
    def load(self, results: PipelineResults) -> Any:
        """Return the loaded value."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Source":
        """Restore a source of this kind from JSON."""


_SOURCE_TYPES: dict[str, type[Source]] = {}


def register_source_type(source_cls: type[Source]) -> type[Source]:
    """Register a ``Source`` subclass so pipeline JSON can name it."""

    if not getattr(source_cls, "source_type", ""):
        raise ValueError(
            f"{source_cls.__name__} must define a non-empty 'source_type'."
        )
    _SOURCE_TYPES[source_cls.source_type] = source_cls
    return source_cls


def source_from_config(config: Source | Mapping[str, Any]) -> Source:
    """Build a source from its JSON representation."""

    if isinstance(config, Source):
        return config
    if not isinstance(config, Mapping):
        raise PipelineSerializationError(
            f"Source config must be an object, got {type(config).__name__}."
        )
    source_type = str(config.get("type", ""))
    source_cls = _SOURCE_TYPES.get(source_type)
    if source_cls is None:
        known = ", ".join(sorted(_SOURCE_TYPES)) or "<none>"
        raise PipelineSerializationError(
            f"Unknown source type: '{source_type}'. Known types: {known}."
        )
    return source_cls.from_dict(config)


@register_source_type
class TextSource(Source):
    """Inline text, useful for tests and small fixed prompts."""

    source_type = "text"
    yields_text = True

    def __init__(self, text: str) -> None:
        self.text = text

    def load(self, results: PipelineResults) -> Any:
        return render_path(self.text, results) if "{" in self.text else self.text

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.source_type, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TextSource":
        if "text" not in data:
            raise PipelineSerializationError("Text source requires 'text'.")
        return cls(str(data["text"]))


@register_source_type
class FileSource(Source):
    """One file, read as text.

    The path may contain placeholders: ``{inputs/report_path}``.
    """

    source_type = "file"
    yields_text = True

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def load(self, results: PipelineResults) -> Any:
        path = Path(render_path(self.path, results))
        if not path.is_file():
            raise PipelineExecutionError(f"Source file does not exist: {path}")
        return path.read_text(encoding=self.encoding)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.source_type, "path": self.path}
        if self.encoding != "utf-8":
            data["encoding"] = self.encoding
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileSource":
        if "path" not in data:
            raise PipelineSerializationError("File source requires 'path'.")
        return cls(str(data["path"]), str(data.get("encoding", "utf-8")))


def _read_records(
    paths: Sequence[Path],
    encoding: str,
    max_files: int,
) -> list[dict[str, Any]]:
    if len(paths) > max_files:
        raise PipelineExecutionError(
            f"Source matched {len(paths)} files, over the limit of {max_files}. "
            "Narrow the pattern, or raise 'max_files'."
        )
    records = []
    for path in paths:
        records.append(
            {
                "path": str(path),
                "name": path.name,
                "content": path.read_text(encoding=encoding),
            }
        )
    return records


@register_source_type
class FilesSource(Source):
    """An explicit list of files, loaded as records."""

    source_type = "files"

    def __init__(
        self,
        paths: Sequence[str],
        encoding: str = "utf-8",
        max_files: int = DEFAULT_MAX_FILES,
    ) -> None:
        self.paths = list(paths)
        self.encoding = encoding
        self.max_files = max_files

    def load(self, results: PipelineResults) -> Any:
        resolved = [Path(render_path(p, results)) for p in self.paths]
        missing = [str(p) for p in resolved if not p.is_file()]
        if missing:
            raise PipelineExecutionError(
                f"Source files do not exist: {', '.join(missing)}"
            )
        return _read_records(resolved, self.encoding, self.max_files)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.source_type, "paths": list(self.paths)}
        if self.encoding != "utf-8":
            data["encoding"] = self.encoding
        if self.max_files != DEFAULT_MAX_FILES:
            data["max_files"] = self.max_files
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FilesSource":
        paths = data.get("paths")
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
            raise PipelineSerializationError(
                "Files source requires 'paths' as a list."
            )
        return cls(
            [str(p) for p in paths],
            str(data.get("encoding", "utf-8")),
            int(data.get("max_files", DEFAULT_MAX_FILES)),
        )


@register_source_type
class FolderSource(Source):
    """Every file in a folder matching a glob, loaded as records.

    Results are sorted by path so a run is reproducible.
    """

    source_type = "folder"

    def __init__(
        self,
        path: str,
        glob: str = "*",
        encoding: str = "utf-8",
        max_files: int = DEFAULT_MAX_FILES,
    ) -> None:
        self.path = path
        self.glob = glob
        self.encoding = encoding
        self.max_files = max_files

    def load(self, results: PipelineResults) -> Any:
        folder = Path(render_path(self.path, results))
        if not folder.is_dir():
            raise PipelineExecutionError(f"Source folder does not exist: {folder}")
        matches = sorted(p for p in folder.glob(self.glob) if p.is_file())
        return _read_records(matches, self.encoding, self.max_files)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.source_type, "path": self.path}
        if self.glob != "*":
            data["glob"] = self.glob
        if self.encoding != "utf-8":
            data["encoding"] = self.encoding
        if self.max_files != DEFAULT_MAX_FILES:
            data["max_files"] = self.max_files
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FolderSource":
        if "path" not in data:
            raise PipelineSerializationError("Folder source requires 'path'.")
        return cls(
            str(data["path"]),
            str(data.get("glob", "*")),
            str(data.get("encoding", "utf-8")),
            int(data.get("max_files", DEFAULT_MAX_FILES)),
        )


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class Sink(abc.ABC):
    """Emits a value out of a pipeline."""

    #: Value written to, and dispatched on, the JSON ``type`` field.
    sink_type: str = ""

    #: Whether this sink leaves the process (writes a file, calls a network).
    #: Sinks that only print are not gated.
    writes: bool = True

    @abc.abstractmethod
    def emit(self, value: Any, results: PipelineResults) -> Any:
        """Emit ``value``. Returns anything worth recording, or ``None``."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sink":
        """Restore a sink of this kind from JSON."""


_SINK_TYPES: dict[str, type[Sink]] = {}


def register_sink_type(sink_cls: type[Sink]) -> type[Sink]:
    """Register a ``Sink`` subclass so pipeline JSON can name it."""

    if not getattr(sink_cls, "sink_type", ""):
        raise ValueError(f"{sink_cls.__name__} must define a non-empty 'sink_type'.")
    _SINK_TYPES[sink_cls.sink_type] = sink_cls
    return sink_cls


def sink_from_config(config: Sink | Mapping[str, Any] | str) -> Sink:
    """Build a sink from its JSON representation, or a bare type name."""

    if isinstance(config, Sink):
        return config
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, Mapping):
        raise PipelineSerializationError(
            f"Sink config must be an object, got {type(config).__name__}."
        )
    sink_type = str(config.get("type", ""))
    sink_cls = _SINK_TYPES.get(sink_type)
    if sink_cls is None:
        known = ", ".join(sorted(_SINK_TYPES)) or "<none>"
        raise PipelineSerializationError(
            f"Unknown sink type: '{sink_type}'. Known types: {known}."
        )
    return sink_cls.from_dict(config)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


@register_sink_type
class StreamSink(Sink):
    """Print to stdout or stderr. Not gated: printing writes nothing."""

    sink_type = "stream"
    writes = False

    STREAMS = ("stdout", "stderr")

    def __init__(self, stream: str = "stdout", prefix: str = "") -> None:
        if stream not in self.STREAMS:
            raise ValueError(f"stream must be one of {self.STREAMS}, got {stream!r}.")
        self.stream = stream
        self.prefix = prefix

    def emit(self, value: Any, results: PipelineResults) -> Any:
        target = sys.stdout if self.stream == "stdout" else sys.stderr
        prefix = render_path(self.prefix, results) if self.prefix else ""
        print(f"{prefix}{_as_text(value)}", file=target)
        return None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.sink_type}
        if self.stream != "stdout":
            data["stream"] = self.stream
        if self.prefix:
            data["prefix"] = self.prefix
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StreamSink":
        return cls(str(data.get("stream", "stdout")), str(data.get("prefix", "")))


@register_sink_type
class FileSink(Sink):
    """Write the value to one file. The path may contain placeholders."""

    sink_type = "file"

    MODES = ("write", "append")

    def __init__(
        self,
        path: str,
        mode: str = "write",
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")
        self.path = path
        self.mode = mode
        self.encoding = encoding
        self.create_parents = create_parents

    def emit(self, value: Any, results: PipelineResults) -> Any:
        path = Path(render_path(self.path, results))
        if self.create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if self.mode == "append" else "w", encoding=self.encoding) as handle:
            handle.write(_as_text(value))
        return str(path)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.sink_type, "path": self.path}
        if self.mode != "write":
            data["mode"] = self.mode
        if self.encoding != "utf-8":
            data["encoding"] = self.encoding
        if not self.create_parents:
            data["create_parents"] = False
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileSink":
        if "path" not in data:
            raise PipelineSerializationError("File sink requires 'path'.")
        return cls(
            str(data["path"]),
            str(data.get("mode", "write")),
            str(data.get("encoding", "utf-8")),
            bool(data.get("create_parents", True)),
        )


@register_sink_type
class FilesSink(Sink):
    """Write one file per item of a list.

    The path template sees ``{index}`` and, when the item is a mapping, its
    own keys — so ``"out/{name}.md"`` works over records from a folder source.
    """

    sink_type = "files"

    def __init__(
        self,
        path: str,
        content_key: str | None = None,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> None:
        self.path = path
        self.content_key = content_key
        self.encoding = encoding
        self.create_parents = create_parents

    def emit(self, value: Any, results: PipelineResults) -> Any:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise PipelineExecutionError(
                f"Files sink expected a list, got {type(value).__name__}."
            )

        written = []
        for index, item in enumerate(value):
            scope = results.copy()
            scope.section("item", create=True).update(
                item if isinstance(item, Mapping) else {"value": item}
            )
            scope.section("item", create=True)["index"] = index
            # The item's own fields shadow nothing: they live in their own
            # section, and bare {name} still resolves through it.
            fields = dict(scope.flat())
            if isinstance(item, Mapping):
                fields.update(item)
            fields["index"] = index

            rendered = MessageTemplate.from_string(
                normalize_placeholders(self.path)
            ).generate_message_content(fields, remove_empty_template_field=False)
            if "{" in rendered and "}" in rendered:
                raise PipelineExecutionError(
                    f"Files sink path {self.path!r} has an unresolved placeholder "
                    f"for item {index}: {rendered!r}."
                )

            content = item
            if self.content_key is not None:
                if not isinstance(item, Mapping) or self.content_key not in item:
                    raise PipelineExecutionError(
                        f"Files sink content_key '{self.content_key}' is missing "
                        f"from item {index}."
                    )
                content = item[self.content_key]

            path = Path(rendered)
            if self.create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_as_text(content), encoding=self.encoding)
            written.append(str(path))
        return written

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.sink_type, "path": self.path}
        if self.content_key is not None:
            data["content_key"] = self.content_key
        if self.encoding != "utf-8":
            data["encoding"] = self.encoding
        if not self.create_parents:
            data["create_parents"] = False
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FilesSink":
        if "path" not in data:
            raise PipelineSerializationError("Files sink requires 'path'.")
        content_key = data.get("content_key")
        return cls(
            str(data["path"]),
            None if content_key is None else str(content_key),
            str(data.get("encoding", "utf-8")),
            bool(data.get("create_parents", True)),
        )


@register_sink_type
class HttpSink(Sink):
    """Send the value to an HTTP endpoint.

    Secrets are never serialized: ``headers_from_env`` maps a header name to
    the *environment variable* holding its value, the same rule provider
    configs follow.
    """

    sink_type = "http"

    METHODS = ("POST", "PUT", "PATCH")

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        headers_from_env: Mapping[str, str] | None = None,
        as_json: bool = True,
        field: str = "content",
        timeout: float = 30.0,
    ) -> None:
        method = method.upper()
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}.")
        self.url = url
        self.method = method
        self.headers = dict(headers or {})
        self.headers_from_env = dict(headers_from_env or {})
        self.as_json = as_json
        self.field = field
        self.timeout = timeout

    def resolve_headers(self) -> dict[str, str]:
        """Return the headers, reading secret values from the environment."""

        headers = dict(self.headers)
        for name, env_var in self.headers_from_env.items():
            value = os.environ.get(env_var)
            if not value:
                raise PipelineExecutionError(
                    f"Environment variable '{env_var}' is not set, so header "
                    f"'{name}' cannot be built for {self.url}."
                )
            headers[name] = value
        return headers

    def emit(self, value: Any, results: PipelineResults) -> Any:
        import requests

        url = render_path(self.url, results)
        headers = self.resolve_headers()

        if self.as_json:
            payload = value if isinstance(value, (dict, list)) else {self.field: value}
            response = requests.request(
                self.method, url, json=payload, headers=headers, timeout=self.timeout
            )
        else:
            response = requests.request(
                self.method,
                url,
                data=_as_text(value).encode("utf-8"),
                headers={"Content-Type": "text/plain", **headers},
                timeout=self.timeout,
            )

        if response.status_code >= 400:
            raise PipelineExecutionError(
                f"HTTP sink {self.method} {url} failed with "
                f"{response.status_code}: {response.text[:200]}"
            )
        return response.status_code

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.sink_type, "url": self.url}
        if self.method != "POST":
            data["method"] = self.method
        if self.headers:
            data["headers"] = dict(self.headers)
        if self.headers_from_env:
            data["headers_from_env"] = dict(self.headers_from_env)
        if not self.as_json:
            data["as_json"] = False
        if self.field != "content":
            data["field"] = self.field
        if self.timeout != 30.0:
            data["timeout"] = self.timeout
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HttpSink":
        if "url" not in data:
            raise PipelineSerializationError("HTTP sink requires 'url'.")
        for forbidden in ("auth", "token", "api_key"):
            if forbidden in data:
                raise PipelineSerializationError(
                    f"HTTP sink must not contain a literal '{forbidden}'. Use "
                    "'headers_from_env' to name the environment variable "
                    "holding it."
                )
        return cls(
            str(data["url"]),
            str(data.get("method", "POST")),
            data.get("headers"),
            data.get("headers_from_env"),
            bool(data.get("as_json", True)),
            str(data.get("field", "content")),
            float(data.get("timeout", 30.0)),
        )


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


@register_process_type
class SourceProcess(Process):
    """Load data into the pipeline.

    The loaded value lands in ``inputs`` by default, because it is input --
    ``outputs`` stays for what the model produced.

    JSON::

        {
          "process_type": "source",
          "process_name": "load",
          "source": {"type": "folder", "path": "./notes", "glob": "*.md"},
          "splitter": {"type": "recursive_character", "chunk_size": 800},
          "result_key": "documents"
        }
    """

    process_type = "source"

    def __init__(
        self,
        source: Source | Mapping[str, Any],
        result_key: str = "documents",
        splitter: Mapping[str, Any] | str | None = None,
        section: str = "inputs",
        process_name: str = "SourceProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
    ):
        """Initialize a source process.

        Args:
            source: Source object or config describing what to load.
            result_key: Key the loaded value is written to.
            splitter: Optional text splitter config. A source yielding text
                becomes a list of chunks; a source yielding records becomes
                more records, each with a ``chunk_index``.
            section: Results section to write into. Defaults to ``inputs``.
            process_name: Name identifier for the process.
            agent: Unused; sources call no model. Accepted so a source can sit
                anywhere a process can.
            agent_name: Name of a declared agent, for JSON round-tripping.
        """
        super().__init__(process_name, agent, agent_name)
        self.source = source_from_config(source)
        self.result_key = result_key
        self.splitter_config = splitter_config_to_dict(splitter)
        self.section = section

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """Load the source and write it into the configured section."""

        value = self.source.load(results)
        splitter = build_splitter(self.splitter_config)

        if splitter is not None:
            if self.source.yields_text:
                value = list(splitter.get_chunks(value))
            else:
                chunked: list[dict[str, Any]] = []
                for record in value:
                    for index, chunk in enumerate(splitter.get_chunks(record["content"])):
                        chunked.append({**record, "content": chunk, "chunk_index": index})
                value = chunked

        results.section(self.section, create=True)[self.result_key] = value
        return results

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this source process."""

        data: dict[str, Any] = {
            "process_type": self.process_type,
            "process_name": self.process_name,
            "source": self.source.to_dict(),
            "result_key": self.result_key,
        }
        if self.splitter_config is not None:
            data["splitter"] = dict(self.splitter_config)
        if self.section != "inputs":
            data["section"] = self.section
        if self.agent_name:
            data["agent"] = self.agent_name
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "SourceProcess":
        """Restore a source process from JSON."""

        process_name = str(data.get("process_name", data.get("name", "SourceProcess")))
        if "source" not in data:
            raise PipelineSerializationError(
                f"Source process '{process_name}' is missing 'source'."
            )
        agent_name = data.get("agent")
        return cls(
            source=data["source"],
            result_key=str(data.get("result_key", "documents")),
            splitter=data.get("splitter"),
            section=str(data.get("section", "inputs")),
            process_name=process_name,
            agent=None,
            agent_name=str(agent_name) if agent_name else None,
        )


@register_process_type
class SinkProcess(Process):
    """Send a value out of the pipeline.

    JSON::

        {
          "process_type": "sink",
          "process_name": "save",
          "sink": {"type": "file", "path": "out/{inputs/name}.md"},
          "from": "outputs/draft"
        }

    Sinks that write a file or make a request are refused unless the pipeline
    was loaded with ``allow_writes=True``. A sink built in Python is permitted
    by default; the gate is on loading a document that writes.
    """

    process_type = "sink"

    def __init__(
        self,
        sink: Sink | Mapping[str, Any] | str,
        source_key: str = "outputs",
        process_name: str = "SinkProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
        record_as: str | None = None,
        allow_writes: bool = True,
    ):
        """Initialize a sink process.

        Args:
            sink: Sink object or config describing where the value goes.
            source_key: Results path to read, such as ``outputs/draft``.
                Defaults to the whole ``outputs`` section.
            process_name: Name identifier for the process.
            agent: Unused; sinks call no model.
            agent_name: Name of a declared agent, for JSON round-tripping.
            record_as: Optional output key recording what the sink returned --
                the path written, or the HTTP status.
            allow_writes: Whether this sink may touch the filesystem or the
                network. Defaults to true in Python and false when loaded from
                JSON without ``allow_writes=True``.
        """
        super().__init__(process_name, agent, agent_name)
        self.sink = sink_from_config(sink)
        self.source_key = source_key
        self.record_as = record_as
        self.allow_writes = allow_writes

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """Read the configured value and emit it."""

        if self.sink.writes and not self.allow_writes:
            raise PipelineExecutionError(
                f"Sink '{self.process_name}' uses a '{self.sink.sink_type}' sink, "
                "which writes outside the process, but this pipeline was loaded "
                "without write permission. Pass allow_writes=True to enable it."
            )

        found, value = results.resolve_path(self.source_key)
        if not found:
            raise PipelineExecutionError(
                f"Sink '{self.process_name}' reads '{self.source_key}', which "
                f"does not exist. Available outputs: "
                f"{', '.join(sorted(results.outputs)) or '<none>'}."
            )

        emitted = self.sink.emit(value, results)
        if self.record_as:
            results.outputs[self.record_as] = emitted
        return results

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this sink process."""

        data: dict[str, Any] = {
            "process_type": self.process_type,
            "process_name": self.process_name,
            "sink": self.sink.to_dict(),
        }
        if self.source_key != "outputs":
            data["from"] = self.source_key
        if self.record_as:
            data["record_as"] = self.record_as
        if self.agent_name:
            data["agent"] = self.agent_name
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "SinkProcess":
        """Restore a sink process from JSON."""

        process_name = str(data.get("process_name", data.get("name", "SinkProcess")))
        if "sink" not in data:
            raise PipelineSerializationError(
                f"Sink process '{process_name}' is missing 'sink'."
            )
        agent_name = data.get("agent")
        record_as = data.get("record_as")
        return cls(
            sink=data["sink"],
            source_key=str(data.get("from", data.get("source_key", "outputs"))),
            process_name=process_name,
            agent=None,
            agent_name=str(agent_name) if agent_name else None,
            record_as=str(record_as) if record_as else None,
            allow_writes=context.allow_writes,
        )

"""Serializable text splitters for pipeline sources.

ToolAgents already ships text splitters under
:mod:`ToolAgents.knowledge.text_processing.text_splitter`. This module gives
them names and a JSON config shape so a pipeline file can ask for one::

    {"type": "recursive_character", "chunk_size": 1000, "chunk_overlap": 100}

Register your own with :func:`register_splitter_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

__all__ = [
    "SPLITTER_SPECS",
    "SplitterSpec",
    "SplitterConfigurationError",
    "build_splitter",
    "register_splitter_spec",
    "splitter_config_to_dict",
]


class SplitterConfigurationError(ValueError):
    """Raised when a text splitter cannot be built from configuration."""


@dataclass(frozen=True)
class SplitterSpec:
    """How to construct one kind of text splitter from configuration."""

    #: Value used in the JSON ``type`` field.
    name: str
    #: Builds the splitter from the config's remaining keys.
    factory: Callable[..., Any]
    #: Keys this splitter accepts, so a typo is reported rather than ignored.
    fields: tuple[str, ...] = ()


SPLITTER_SPECS: dict[str, SplitterSpec] = {}


def register_splitter_spec(spec: SplitterSpec) -> SplitterSpec:
    """Register a splitter kind so pipeline JSON can name it."""

    SPLITTER_SPECS[spec.name] = spec
    return spec


def _build_none(**_: Any) -> Any:
    from ToolAgents.knowledge.text_processing.text_splitter import NonTextSplitter

    return NonTextSplitter()


def _build_simple(chunk_size: int = 1000, overlap: int = 0) -> Any:
    from ToolAgents.knowledge.text_processing.text_splitter import SimpleTextSplitter

    return SimpleTextSplitter(chunk_size=int(chunk_size), overlap=int(overlap))


def _build_recursive(
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
    keep_separator: bool = False,
) -> Any:
    from ToolAgents.knowledge.text_processing.text_splitter import (
        RecursiveCharacterTextSplitter,
    )

    return RecursiveCharacterTextSplitter(
        separators=list(separators) if separators else ["\n\n", "\n", ". ", " ", ""],
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        keep_separator=bool(keep_separator),
    )


for _spec in (
    SplitterSpec(name="none", factory=_build_none),
    SplitterSpec(name="simple", factory=_build_simple, fields=("chunk_size", "overlap")),
    SplitterSpec(
        name="recursive_character",
        factory=_build_recursive,
        fields=("chunk_size", "chunk_overlap", "separators", "keep_separator"),
    ),
):
    register_splitter_spec(_spec)


def build_splitter(config: Mapping[str, Any] | str | None) -> Any:
    """Build a text splitter from a config mapping, or a bare type name.

    Returns ``None`` when ``config`` is ``None``, meaning "do not split".
    """

    if config is None:
        return None
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, Mapping):
        raise SplitterConfigurationError(
            f"Splitter config must be an object or a type name, got "
            f"{type(config).__name__}."
        )

    splitter_type = str(config.get("type", "recursive_character"))
    spec = SPLITTER_SPECS.get(splitter_type)
    if spec is None:
        known = ", ".join(sorted(SPLITTER_SPECS))
        raise SplitterConfigurationError(
            f"Unknown splitter type: '{splitter_type}'. Known types: {known}."
        )

    options = {key: value for key, value in config.items() if key != "type"}
    unknown = sorted(set(options) - set(spec.fields))
    if unknown:
        allowed = ", ".join(spec.fields) or "<none>"
        raise SplitterConfigurationError(
            f"Splitter '{splitter_type}' does not accept: {', '.join(unknown)}. "
            f"Accepted options: {allowed}."
        )

    try:
        return spec.factory(**options)
    except SplitterConfigurationError:
        raise
    except Exception as exc:
        raise SplitterConfigurationError(
            f"Could not build splitter '{splitter_type}': {exc}"
        ) from exc


def splitter_config_to_dict(
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any] | None:
    """Normalize a splitter config for serialization."""

    if config is None:
        return None
    if isinstance(config, str):
        return {"type": config}
    return dict(config)

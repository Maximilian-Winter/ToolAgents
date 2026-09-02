"""Sectioned pipeline results.

A pipeline used to carry one flat dictionary in which the caller's arguments,
every step's output, and flow-control scratch state all shared a namespace.
That is why a step named ``sum`` could shadow a condition helper, why a loop's
``iteration`` counter leaked into the caller's results, and why a map had to
*infer* which keys an iteration had produced.

Results are now divided into sections:

``inputs``
    What the caller passed to :meth:`Pipeline.run_pipeline`.
``outputs``
    What steps produced. Parallel branches nest one level deeper here, so two
    branches writing ``draft`` become ``outputs/news/draft`` and
    ``outputs/stats/draft`` rather than colliding.
``vars``
    Flow-control scratch: a loop's ``iteration``, a map's ``item`` and
    ``index``. Scoped to the body that owns it and removed afterwards.

Further sections can be added at any time — ``results.section("agent",
create=True)`` — without touching this module.

Addressing
----------

Prompt templates use a path::

    "Revise {outputs/draft} for {inputs/audience}"

Conditions use ordinary subscripts, which the sandbox already permits::

    outputs['draft'] != ''

A **bare** name still works in both, and resolves innermost-first —
``vars``, then ``outputs``, then ``inputs`` — the same rule a local variable
follows over a global. That is what keeps every pipeline written against the
old flat dictionary running unchanged.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Iterator, Mapping

__all__ = [
    "PipelineResults",
    "SECTION_LOOKUP_ORDER",
]

#: Section resolution order for a bare name: innermost scope wins.
SECTION_LOOKUP_ORDER = ("vars", "outputs", "inputs")

#: Sections every results object starts with.
DEFAULT_SECTIONS = ("inputs", "outputs", "vars")

#: Separator used in template paths and section addresses.
PATH_SEPARATOR = "/"


class PipelineResults(MutableMapping):
    """A sectioned results mapping that still behaves like the old flat dict.

    Reading and writing a bare key works exactly as before, so existing
    pipelines, prompt templates and calling code need no changes::

        results["draft"]            # resolves through vars -> outputs -> inputs
        results["draft"] = "..."    # writes to outputs

    The structure is available whenever it is wanted::

        results.outputs["draft"]
        results["outputs/draft"]
        results["outputs/news/draft"]
    """

    def __init__(
        self,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        vars: Mapping[str, Any] | None = None,
        sections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._sections: dict[str, dict[str, Any]] = {
            name: {} for name in DEFAULT_SECTIONS
        }
        if sections:
            for name, values in sections.items():
                self._sections.setdefault(name, {}).update(values)
        if inputs:
            self._sections["inputs"].update(inputs)
        if outputs:
            self._sections["outputs"].update(outputs)
        if vars:
            self._sections["vars"].update(vars)

    # -- construction ------------------------------------------------------

    @classmethod
    def coerce(cls, values: "PipelineResults | Mapping[str, Any] | None") -> "PipelineResults":
        """Return ``values`` as a ``PipelineResults``.

        A plain mapping is treated as inputs, so a caller that still hands over
        a flat dictionary keeps working.
        """

        if isinstance(values, cls):
            return values
        return cls(inputs=values or {})

    def copy(self) -> "PipelineResults":
        """Return a copy with independent section dictionaries.

        Section dictionaries are copied; the values inside them are not. A
        body that *mutates* a nested list or dict still affects the original,
        which is why bodies should rebind rather than mutate.
        """

        return PipelineResults(
            sections={name: dict(values) for name, values in self._sections.items()}
        )

    # -- sections ----------------------------------------------------------

    @property
    def inputs(self) -> dict[str, Any]:
        """Arguments passed to ``run_pipeline``."""

        return self._sections["inputs"]

    @property
    def outputs(self) -> dict[str, Any]:
        """Values produced by steps."""

        return self._sections["outputs"]

    @property
    def vars(self) -> dict[str, Any]:
        """Flow-control scratch state, scoped to the body that owns it."""

        return self._sections["vars"]

    @property
    def section_names(self) -> list[str]:
        """Return the names of all sections."""

        return list(self._sections)

    def section(self, name: str, create: bool = False) -> dict[str, Any]:
        """Return a section by name, optionally creating it.

        Adding a section is how a new kind of state joins the namespace — for
        example ``results.section("agent", create=True)`` for agent internals.
        """

        if name not in self._sections:
            if not create:
                known = ", ".join(self._sections)
                raise KeyError(
                    f"Unknown results section: '{name}'. Sections: {known}."
                )
            self._sections[name] = {}
        return self._sections[name]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return the sectioned structure as plain dictionaries."""

        return {name: dict(values) for name, values in self._sections.items()}

    # -- path addressing ---------------------------------------------------

    def resolve_path(self, path: str) -> tuple[bool, Any]:
        """Resolve ``path`` and return ``(found, value)``.

        A path beginning with a section name is read from that section;
        anything else is treated as a bare name and resolved by scope order.
        Returning a flag rather than raising lets templates and conditions
        each decide what an absent value means.
        """

        if PATH_SEPARATOR not in path:
            return self._resolve_bare(path)

        head, *rest = path.split(PATH_SEPARATOR)
        if head in self._sections:
            current: Any = self._sections[head]
        else:
            found, current = self._resolve_bare(head)
            if not found:
                return False, None

        for part in rest:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return False, None
        return True, current

    def _resolve_bare(self, name: str) -> tuple[bool, Any]:
        for section_name in SECTION_LOOKUP_ORDER:
            section = self._sections.get(section_name)
            if section is not None and name in section:
                return True, section[name]
        # Sections not in the lookup order are still addressable by name, and
        # a section itself is a legitimate value: `outputs` in a condition.
        if name in self._sections:
            return True, self._sections[name]
        return False, None

    def set_path(self, path: str, value: Any) -> None:
        """Write ``value`` at ``path``, creating intermediate mappings."""

        if PATH_SEPARATOR not in path:
            self.outputs[path] = value
            return

        head, *rest = path.split(PATH_SEPARATOR)
        current = self.section(head, create=True)
        for part in rest[:-1]:
            nested = current.get(part)
            if not isinstance(nested, dict):
                nested = {}
                current[part] = nested
            current = nested
        current[rest[-1]] = value

    # -- mapping protocol --------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        found, value = self.resolve_path(key)
        if not found:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Write a value. A bare key goes to ``outputs``."""

        self.set_path(key, value)

    def __delitem__(self, key: str) -> None:
        if PATH_SEPARATOR in key:
            head, *rest = key.split(PATH_SEPARATOR)
            current: Any = self._sections.get(head)
            for part in rest[:-1]:
                if not isinstance(current, Mapping):
                    raise KeyError(key)
                current = current.get(part)
            if not isinstance(current, dict) or rest[-1] not in current:
                raise KeyError(key)
            del current[rest[-1]]
            return

        for section_name in SECTION_LOOKUP_ORDER:
            section = self._sections.get(section_name)
            if section is not None and key in section:
                del section[key]
                return
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.flat())

    def __len__(self) -> int:
        return len(self.flat())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.resolve_path(key)[0]

    def flat(self) -> dict[str, Any]:
        """Return the flattened view a bare-name lookup would see.

        Outer scopes first, so inner ones overwrite: this is the dictionary
        the pipeline behaved like before sections existed.
        """

        flattened: dict[str, Any] = {}
        for section_name in reversed(SECTION_LOOKUP_ORDER):
            flattened.update(self._sections.get(section_name, {}))
        return flattened

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = ", ".join(
            f"{name}={len(values)}" for name, values in self._sections.items()
        )
        return f"PipelineResults({parts})"

"""A sink this project defines: append records to a JSON Lines file.

Dropping a module in ``adapter/output/`` is all it takes. Importing it
registers the type, and workflows can then use ``{"type": "jsonl", ...}``
exactly like a built-in sink.
"""

import json
from pathlib import Path

from ToolAgents.pipelines import Sink, register_sink_type
from ToolAgents.pipelines.data_io import render_path

NEWLINE = "\n"


@register_sink_type
class JsonLinesSink(Sink):
    """Append each item of a list to a ``.jsonl`` file, one object per line."""

    sink_type = "jsonl"

    #: This reaches the filesystem, so the allow_writes gate must cover it.
    writes = True

    def __init__(self, path: str, key: str | None = None) -> None:
        """Create the sink.

        Args:
            path: Destination file. May contain ``{section/key}`` placeholders.
            key: Optional field to keep from each record, instead of all of it.
        """
        self.path = path
        self.key = key

    def emit(self, value, results):
        """Write one line per item and return how many were written."""

        target = Path(render_path(self.path, results))
        target.parent.mkdir(parents=True, exist_ok=True)
        items = value if isinstance(value, list) else [value]

        with open(target, "a", encoding="utf-8") as handle:
            for item in items:
                record = item if isinstance(item, dict) else {"value": item}
                if self.key:
                    record = {self.key: record.get(self.key, record)}
                handle.write(json.dumps(record, default=str) + NEWLINE)
        return len(items)

    def to_dict(self):
        """Return the JSON form, so the workflow still round-trips."""

        data = {"type": self.sink_type, "path": self.path}
        if self.key:
            data["key"] = self.key
        return data

    @classmethod
    def from_dict(cls, data):
        """Rebuild the sink from a workflow document."""

        return cls(str(data["path"]), data.get("key"))

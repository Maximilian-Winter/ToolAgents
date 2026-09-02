---
title: Pipeline Sources and Sinks
---

# Pipeline Sources and Sinks

A [pipeline](pipelines.md) normally takes its input from `run_pipeline(**kwargs)`
and hands its output back as a return value. **Sources** and **sinks** let the
workflow file say where data comes from and where it goes: read a folder, chunk
it, and write one file per result, all without Python.

Both are ordinary processes, so they compose with everything else. Writing one
file per item is a sink inside a [map](pipeline-flow-control.md#map); writing
only when review passed is a sink inside a
[conditional](pipeline-flow-control.md#conditions). Neither needed new
machinery.

!!! warning "Writing is opt-in"

    Reading is enabled by default. **Writing files and making HTTP requests are
    not** — a sink is refused unless the pipeline was loaded with
    `allow_writes=True`:

    ```python
    Pipeline.load_from_json("workflow.json", allow_writes=True)
    ```

    A document that reads is a privacy question; one that writes destroys data
    and one that POSTs exfiltrates it. The gate is on *loading a document that
    writes*, not on writing Python that writes — a sink you construct in code is
    permitted, because writing the code is the intent.

## Sources

A source loads data and writes it into the **`inputs`** section, because it is
input: `outputs` stays for what the model produced.

```json
{
  "process_type": "source",
  "process_name": "load",
  "source": {"type": "folder", "path": "./notes", "glob": "*.md"},
  "splitter": {"type": "recursive_character", "chunk_size": 1000, "chunk_overlap": 100},
  "result_key": "chunks"
}
```

| `type` | Loads | Produces |
| --- | --- | --- |
| `text` | Inline text | a string |
| `file` | One file | a string |
| `files` | An explicit list of paths | a list of records |
| `folder` | Every file matching a glob, sorted by path | a list of records |

A **record** is `{"path": ..., "name": ..., "content": ...}`. Sources that
produce records pair naturally with [`map`](pipeline-flow-control.md#map).

Paths may contain placeholders, so a source can read somewhere the pipeline was
told about: `{"path": "{inputs/report_dir}/summary.md"}`. An unresolved
placeholder is an error rather than a silently mangled path.

`folder` and `files` accept `max_files` (default 1000), so a mistyped glob
fails instead of loading a filesystem.

### Chunking

`splitter` runs the loaded text through one of ToolAgents' text splitters:

| `type` | Options |
| --- | --- |
| `none` | — |
| `simple` | `chunk_size`, `overlap` |
| `recursive_character` | `chunk_size`, `chunk_overlap`, `separators`, `keep_separator` |

Splitting a source that yields **text** gives a list of chunk strings.
Splitting one that yields **records** gives more records, each keeping its
`path` and `name` and gaining a `chunk_index`. Either way the result is a list,
so the body downstream does not care which source produced it.

An unknown splitter option is reported rather than ignored — a silently
dropped `chunk_size` would give you the wrong chunks with no sign of it.

Register your own with `register_splitter_spec`.

## Sinks

A sink reads one path out of the results and emits it.

```json
{
  "process_type": "sink",
  "process_name": "save",
  "sink": {"type": "file", "path": "out/{inputs/name}.md"},
  "from": "outputs/draft",
  "record_as": "written_path"
}
```

`from` is a results path (`outputs/draft`, `inputs/chunks`); it defaults to the
whole `outputs` section. `record_as` stores whatever the sink returned — the
path written, or the HTTP status — so a later step can use it.

| `type` | Emits to | Gated |
| --- | --- | --- |
| `stream` | stdout or stderr | no — printing writes nothing |
| `file` | One file (`write` or `append`) | yes |
| `files` | One file per list item | yes |
| `http` | `POST`/`PUT`/`PATCH` to a URL | yes |

Non-string values are serialized as indented JSON.

### Writing one file per item

`files` writes a file per element of a list. The path template sees `{index}`
and, when the item is a mapping, its own keys — so records from a folder source
round-trip back to disk:

```json
{
  "process_type": "sink",
  "sink": {"type": "files", "path": "out/{name}", "content_key": "content"},
  "from": "inputs/docs"
}
```

### HTTP

```json
{
  "process_type": "sink",
  "sink": {
    "type": "http",
    "url": "https://example.internal/hook",
    "headers_from_env": {"Authorization": "MY_API_TOKEN"}
  },
  "from": "outputs/draft"
}
```

Secrets are never serialized: `headers_from_env` names the **environment
variable** holding a header's value, the same rule
[provider configs](pipeline-endpoints.md#keys-are-never-serialized) follow. A
literal `token`, `auth` or `api_key` in the config is rejected. A response of
400 or above raises.

## A whole workflow

Read a folder, chunk it, summarize each chunk, write the result, and print
where it went — with no input arguments and no output handling in Python:

```json
{
  "schema_version": 2,
  "processes": [
    {
      "process_type": "source",
      "process_name": "load",
      "source": {"type": "folder", "path": "./notes", "glob": "*.md"},
      "splitter": {"type": "recursive_character", "chunk_size": 800},
      "result_key": "chunks"
    },
    {
      "process_type": "map",
      "process_name": "summarize",
      "items": "chunks",
      "item_var": "chunk",
      "collect": "summary",
      "result_key": "summaries",
      "steps": [
        {
          "step_name": "summary",
          "system_message": "You summarize precisely.",
          "prompt_template": "Summarize:\n\n{vars/chunk}"
        }
      ]
    },
    {
      "process_type": "sink",
      "process_name": "save",
      "sink": {"type": "file", "path": "./out/summaries.md"},
      "from": "outputs/summaries",
      "record_as": "written"
    },
    {
      "process_type": "sink",
      "process_name": "report",
      "sink": {"type": "stream", "prefix": "wrote "},
      "from": "outputs/written"
    }
  ]
}
```

```python
pipeline = Pipeline.load_from_json("workflow.json", allow_writes=True)
pipeline.run_pipeline()
```

## Extending

`register_source_type` and `register_sink_type` add your own kinds, the same way
`register_process_type` adds processes. A source implements `load`, a sink
implements `emit`, and both implement `to_dict` / `from_dict` so they keep
round-tripping.

A sink that reaches outside the process should set `writes = True` so the
`allow_writes` gate covers it too.

---
title: The tool-agents CLI
---

# The tool-agents CLI

`tool-agents` runs [workflows](pipelines.md) from a project folder, so a
workflow can be committed alongside the code it serves and run by anyone on the
team without writing Python.

```bash
tool-agents init
tool-agents list
tool-agents show digest
tool-agents run digest --arg topic=otters --allow-writes
```

## The `.tool-agents` folder

```
.tool-agents/
  workflows/        *.json   pipeline documents, named by file stem
  tools/            *.py     modules whose tools become plugins
  prompts/          *.md     reusable prompt text
  providers/        *.json   shared agent and endpoint declarations
  adapter/
    input/          *.py     modules registering custom source types
    output/         *.py     modules registering custom sink types
```

The folder is found by walking up from the working directory, the way `git`
finds `.git`, so a workflow runs the same from anywhere inside the project.
Point somewhere else with `--workspace`.

Nothing here is a new concept for a workflow author. Each folder feeds an
existing mechanism:

| Folder | Reaches a workflow as |
| --- | --- |
| `workflows/` | the document itself |
| `tools/` | a tool plugin named for the file: `{"plugin": "math", "tool_name": "AddNumbers"}` |
| `prompts/` | a `prompts` [results section](pipelines.md#results-sections): `{prompts/reviewer}` |
| `providers/` | entries merged into the workflow's [`agents` block](pipeline-endpoints.md) |
| `adapter/` | registered [source and sink types](pipeline-io.md#extending) |

### workflows/

One JSON pipeline document per file. The name is the file stem, so
`workflows/digest.json` is `tool-agents run digest`.

### tools/

A module may define `TOOLS`, or a `create_tools()` function, or simply bind
`FunctionTool` objects at module level — a small file needs no boilerplate. The
plugin name is the file stem:

```python
# .tool-agents/tools/math.py
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class AddNumbers(BaseModel):
    """Add two numbers."""
    a: int = Field(..., description="First.")
    b: int = Field(..., description="Second.")
    def run(self) -> int:
        return self.a + self.b

TOOLS = [FunctionTool(AddNumbers)]
```

```json
{"step_name": "sum", "tools": [{"plugin": "math", "tool_name": "AddNumbers"}]}
```

A module exposing no tools is reported rather than silently contributing
nothing.

### prompts/

Each file becomes an entry in a `prompts` section, keyed by its stem. Because
it is an ordinary section, it is addressed like any other — in
`prompt_template` **and** in `system_message`, mixed freely with other
placeholders:

```markdown
<!-- .tool-agents/prompts/reviewer.md -->
You are a strict editor. Reply APPROVED, or name the single worst problem.
```

```json
{
  "step_name": "verdict",
  "system_message": "{prompts/reviewer}",
  "prompt_template": "Judge this draft:\n\n{outputs/draft}"
}
```

Two prompt files with the same stem are refused; the stem is the name.

### providers/

Shared endpoint declarations, so every workflow does not repeat them. A file
holds either a list of agents or an object with `agents` and an optional
`default_agent`:

```json
{
  "agents": [
    {
      "name": "writer",
      "provider": {
        "type": "openrouter",
        "model": "qwen/qwen3.5-9b",
        "api_key_env": "OPENROUTER_API_KEY"
      }
    }
  ],
  "default_agent": "writer"
}
```

These merge into every workflow's own `agents` block. **A workflow's own entry
wins**, so redeclaring a name overrides the shared one. An agent declared twice
across `providers/` is an error — the ambiguity has no good answer.

API keys are still never stored: a provider config names the
[environment variable](pipeline-endpoints.md#keys-are-never-serialized) holding
one.

### adapter/

Modules imported for their side effects, so a workflow can use a source or sink
this project defines:

```python
# .tool-agents/adapter/input/shout.py
from ToolAgents.pipelines import Source, register_source_type

@register_source_type
class ShoutSource(Source):
    """Load text, loudly."""
    source_type = "shout"
    yields_text = True
    def __init__(self, text): self.text = text
    def load(self, results): return self.text.upper()
    def to_dict(self): return {"type": "shout", "text": self.text}
    @classmethod
    def from_dict(cls, data): return cls(str(data["text"]))
```

```json
{"process_type": "source", "source": {"type": "shout", "text": "hello"}}
```

`adapter/input/` and `adapter/output/` are a convention rather than a
constraint — both are imported the same way, and a module may register either
kind.

## Commands

### `init`

Creates the folder structure. Takes an optional path; defaults to the working
directory.

### `list`

Shows everything the workspace holds, or one kind:

```bash
tool-agents list
tool-agents list workflows
```

### `show`

Describes one workflow without running it — its declared agents, its processes
in order, and every `{inputs/...}` placeholder it references, which is the
quickest way to see what arguments it expects.

### `run`

```bash
tool-agents run digest --arg topic=otters --arg depth=3
tool-agents run digest --json '{"topic": "otters"}'
tool-agents run digest --json-file args.json
```

`--arg` values are decoded as JSON when they can be, so `depth=3` is an
integer, `flag=true` a boolean and `tags=["a","b"]` a list, while
`topic=otters` stays the string it looks like. All three sources merge, with
`--arg` winning.

Output control:

| Flag | Prints |
| --- | --- |
| *(none)* | each output key, truncated |
| `--output outputs/draft` | just that value |
| `--json-output` | the whole sectioned results object |

!!! warning "`--allow-writes`"

    A workflow whose sinks write files or make HTTP requests needs
    `--allow-writes`. Without it the run stops at the first such sink and says
    so. See [sources and sinks](pipeline-io.md) for why reading and writing are
    gated differently.

### `tools`

Tool inspection, either from an installed module or from the workspace:

```bash
tool-agents tools list --plugin math
tool-agents tools list --module my_package.tools:create_tools
tool-agents tools schema AddNumbers --plugin math
tool-agents tools call AddNumbers --plugin math --json '{"a": 2, "b": 3}'
```

!!! note "`toolagents-tools` moved here"

    The old `toolagents-tools` command still works and prints a notice pointing
    at `tool-agents tools`. It will be removed in a future release; the notice
    goes to stderr, so piping stdout is unaffected in the meantime.

## Worked examples

Three runnable workspaces live in `examples/cli/`, building on each other:

| Example | Shows |
| --- | --- |
| `01-hello` | one workflow, one provider, printed to stdout |
| `02-review` | prompt files, two models, a refine loop, a branch on the outcome |
| `03-digest` | every folder at once: tools, a custom sink adapter, a folder source with chunking, map, parallel, file and stream sinks |

`cd` into one and run — the workspace is found by walking up, so there are no
paths to pass.

## From Python

The workspace is usable without the CLI:

```python
from ToolAgents.workspace import Workspace

workspace = Workspace.discover()
results = workspace.run_workflow(
    "digest", {"topic": "otters"}, allow_writes=True
)
print(results["outputs/draft"])
```

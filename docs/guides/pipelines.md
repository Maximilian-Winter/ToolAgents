---
title: Pipelines
---

# Pipelines

Pipelines let you describe a multi-step agent workflow and save the workflow
shape to JSON. Pipeline JSON stores process names, step names, system messages,
prompt templates, references to tools, flow control, and — optionally — the
endpoints the workflow runs against. Results are carried in named sections
(`inputs`, `outputs`, `vars`), addressed as `{outputs/draft}`.

This page covers building, serializing and loading a pipeline, and how its
results are organized. Two companion pages go further:

- **[Flow control](pipeline-flow-control.md)** — branching, loops, mapping over
  a list, and running branches in parallel.
- **[Endpoints](pipeline-endpoints.md)** — declaring the providers, models and
  API keys a workflow runs against, inside the workflow file itself.
- **[Sources and sinks](pipeline-io.md)** — reading files and folders in,
  chunking them, and writing results out to files, streams or HTTP.

Live agent objects are never serialized. You can either pass agents back in when
loading, or declare them in the JSON as provider configuration, in which case
they are constructed at load time from API keys held in environment variables.
Secrets are never written to the file.

## Serializing a Pipeline

Register tools in a `PipelineToolRegistry` before serializing steps that use
tools:

```python
from ToolAgents import FunctionTool
from ToolAgents.pipelines import (
    Pipeline,
    PipelineToolRegistry,
    ProcessStep,
    SequentialProcess,
)

def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

math_tool = FunctionTool(add_numbers)
tool_registry = PipelineToolRegistry().register_plugin(
    "math",
    [math_tool],
)

step = ProcessStep(
    step_name="sum",
    system_message="You are a careful arithmetic assistant.",
    prompt_template="Add {a} and {b}.",
    tools=[math_tool],
)

process = SequentialProcess(process_name="calculator")
process.add_step(step)

pipeline = Pipeline()
pipeline.add_process(process)

pipeline.save_to_json("calculator.pipeline.json", tool_registry=tool_registry)
```

## Custom Python Tool Plugins

For a self-contained JSON file, put tools in an importable Python module and
register the module source as `module:attribute`. The attribute can be a
zero-argument factory, a `ToolRegistry`, an iterable of tools, a single
`FunctionTool`, or a Pydantic model class.

```python
# my_pipeline_tools.py
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool

class AddNumbers(BaseModel):
    """Add two numbers."""

    a: int = Field(..., description="First number.")
    b: int = Field(..., description="Second number.")

    def run(self) -> int:
        return self.a + self.b

def create_tools():
    return [FunctionTool(AddNumbers)]
```

```python
from ToolAgents.pipelines import PipelineToolRegistry

tool_registry = PipelineToolRegistry().load_plugin(
    "math",
    "my_pipeline_tools:create_tools",
)

pipeline.save_to_json("calculator.pipeline.json", tool_registry=tool_registry)
```

The JSON will include plugin metadata:

```json
{
  "schema_version": 2,
  "tool_plugins": [
    {
      "name": "math",
      "source": "my_pipeline_tools:create_tools"
    }
  ],
  "processes": [
    {
      "process_type": "sequential",
      "process_name": "calculator",
      "steps": [
        {
          "step_name": "sum",
          "system_message": "You are a careful arithmetic assistant.",
          "prompt_template": "Add {a} and {b}.",
          "tools": [
            {
              "plugin": "math",
              "tool_name": "AddNumbers"
            }
          ]
        }
      ]
    }
  ]
}
```

## Loading a Pipeline

When loading, provide the agent that should run the restored process:

```python
from ToolAgents.pipelines import Pipeline

loaded = Pipeline.load_from_json(
    "calculator.pipeline.json",
    default_agent=agent,
)

result = loaded.run_pipeline(a=2, b=3)
```

Loading plugin declarations imports Python modules. Only load pipeline JSON from
trusted sources, or pass a prebuilt `PipelineToolRegistry` and set
`load_tool_plugins=False`.

## Results Sections

A pipeline carries one results object, divided into sections:

| Section | Holds | Written by |
| --- | --- | --- |
| `inputs` | What you passed to `run_pipeline` | the caller |
| `outputs` | What steps produced | steps, and flow control's bookkeeping |
| `vars` | Flow-control scratch: `iteration`, `item`, `index` | loops and maps, scoped to their body |

Prompt templates address a section with a path, and conditions with an ordinary
subscript:

```json
"prompt_template": "Revise {outputs/draft} for {inputs/audience}"
```

```python
ConditionalProcess(condition="outputs['verdict'] != ''")
```

A **bare** name still works in both and resolves innermost-first — `vars`, then
`outputs`, then `inputs` — the same rule a local variable follows over a global.
So `{draft}` and `outputs['draft']` usually mean the same thing, and every
pipeline written against the old flat dictionary keeps running unchanged.

The returned object reads like that flat dictionary too, while exposing the
structure when you want it:

```python
results = pipeline.run_pipeline(topic="otters")

results["greeting"]            # bare lookup, as before
results["outputs/greeting"]    # explicit
results.outputs["greeting"]    # same thing
results.to_dict()              # {"inputs": {...}, "outputs": {...}, "vars": {}}
```

### Why the separation matters

Sections are not decoration; they remove whole classes of collision that a flat
namespace could not express:

- A step named `topic` no longer destroys the caller's `topic` argument — one is
  in `outputs`, the other in `inputs`.
- A step named `iteration` no longer corrupts a loop's counter, which lives in
  `vars`.
- A map knows *exactly* what an iteration produced (the new or rebound keys in
  `outputs`) instead of inferring it by comparing against everything.
- Parallel branches that write the same key get a section each rather than a
  mangled name.

### Adding a section

Sections are open-ended. Anything that needs its own namespace can claim one:

```python
results.section("agent", create=True)["model"] = "claude-sonnet-4"
# then {agent/model} in a template, agent['model'] in a condition
```

Only `inputs`, `outputs` and `vars` take part in bare-name resolution; a custom
section is reached by its path, which is what keeps it from colliding with
anything.

## Composing a value without a model

Joining results together is string work, not reasoning. `TemplateProcess`
renders a template against the results and stores it, with no request:

```json
{
  "process_type": "template",
  "process_name": "assemble",
  "template": "# {outputs/title}

{outputs/body}",
  "result_key": "digest"
}
```

The alternative is asking a model to "return this unchanged", which costs a
request, adds latency, and is not guaranteed to comply.

## Next

- [Flow control](pipeline-flow-control.md) — conditional, loop, map and parallel
  processes, and the sandboxed conditions that drive them.
- [Endpoints](pipeline-endpoints.md) — put the providers and models in the
  workflow file so it runs without an agent argument.
- [Sources and sinks](pipeline-io.md) — read folders and files in, chunk them,
  and write results out.

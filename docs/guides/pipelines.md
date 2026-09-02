---
title: Pipelines
---

# Pipelines

Pipelines let you describe a multi-step agent workflow and save the workflow
shape to JSON. Pipeline JSON stores process names, step names, system messages,
prompt templates, references to tools, flow control, and — optionally — the
endpoints the workflow runs against. Results are carried in named sections
(`inputs`, `outputs`, `vars`), addressed as `{outputs/draft}`.

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

## Flow Control

Beyond `SequentialProcess`, four processes control the shape of a run. Each one
holds *other processes*, so they nest freely, and each serializes to JSON.

| Process | `process_type` | Purpose |
| --- | --- | --- |
| `ConditionalProcess` | `conditional` | Take one branch or the other |
| `LoopProcess` | `loop` | Repeat until a condition holds, under a cap |
| `MapProcess` | `map` | Run a body once per item of a list |
| `ParallelProcess` | `parallel` | Run branches concurrently and merge results |

All four accept either `processes=[...]` (full nesting) or the convenience
`steps=[...]`, which wraps the steps in a `SequentialProcess` for you. A child
process with no agent of its own inherits its parent's, so in the common case
you set the agent once at the top.

### Conditions

Conditions are **sandboxed expressions**, not Python. They are parsed to an AST
and validated against a whitelist, so pipeline JSON from disk or a database
cannot execute arbitrary code:

```python
ConditionalProcess(
    condition="score > 0.8 and not is_empty(draft)",
    then_steps=[publish_step],
    else_steps=[revise_step],
)
```

Names resolve to keys in the results mapping. Available helpers are `len`,
`abs`, `min`, `max`, `sum`, `round`, `int`, `float`, `str`, `bool`, `lower`,
`upper`, `strip`, `contains`, `startswith`, `endswith`, `is_empty`, and
`default`.

Rejected outright: attribute access, imports, lambdas, comprehensions, `**`,
f-strings, walrus assignment, chained multiplication, and any function not in
that list.

Names resolve lazily, so `and`, `or`, and conditional expressions short-circuit
properly. That matters once flow control is in play: a key written only inside a
branch that did not run genuinely does not exist. Use `defined('name')` to test
for it:

```python
"defined('score') and score > 0.8"     # safe when 'score' may not exist
"default(score, 0) > 0.8"              # for a key that exists but may be None
```

Reaching a name that is genuinely absent raises an error naming the results that
do exist, rather than silently evaluating false. A result whose name collides
with a helper (a step named `sum`) is reported rather than quietly shadowing it.

### Loops

`mode` decides **when** the condition is tested and **what it means**. The two
are inverses, as in most languages, so the same expression cannot simply be
moved from one mode to the other:

- **`"until"`** (default) — the condition is a *stop* condition. Run the body,
  then test; stop when it becomes **true**. The body always runs at least once,
  which is what a refine-until-good-enough loop needs, because the condition
  usually reads a value the body produces.
- **`"while"`** — the condition is a *continue* condition. Test before each
  iteration; stop when it becomes **false**. The body may run zero times, and
  the condition must only reference values that already exist.

`mode="until", condition="approved"` and `mode="while", condition="not approved"`
express the same loop.

```python
LoopProcess(
    condition="contains(lower(review), 'approved')",
    mode="until",
    max_iterations=4,
    steps=[draft_step, review_step],
)
```

`max_iterations` is always enforced, so a condition that never becomes true
cannot spin forever burning API credit. Set `on_max_iterations="error"` to make
hitting the cap a failure instead of a quiet exit — a loop that finishes the job
on its last permitted iteration still counts as success. The current index is
exposed as `{vars/iteration}` in prompt templates and is removed again when the
loop ends, so nested loops do not clobber one another; the total lands in
`outputs/<process_name>_iterations`.

### Map

Each iteration runs against its **own copy** of the results, so an iteration
rebinding a key cannot affect the next one, and only the collected list is
written back. The current item and index live in `vars`, addressed as
`{vars/item}` and `{vars/index}`. The copy is shallow — a body that *mutates* a
nested list or dict in place still affects the outer value — so rebind rather
than mutate:

```python
MapProcess(
    items="topics",          # a sandboxed expression; "topics[:3]" also works
    item_var="topic",
    collect="blurb",         # the results key to gather from each iteration
    result_key="blurbs",
    steps=[write_step],
)
```

With `collect` omitted, each entry is a dict of the outputs that iteration
produced — new or rebound keys in `outputs`, which is exact rather than
inferred.

### Parallel

Branches run in worker threads against their own copy of the results, and the
keys each branch added or changed are merged back afterwards:

```python
ParallelProcess(
    branches=[news_branch, stats_branch],
    max_workers=4,
    on_conflict="error",     # or "section", or "last_wins"
)
```

Branches writing the same output with *equal* values are agreeing, not
colliding, and do not trigger `on_conflict`. Under `"section"` a contested
output moves into a sub-section of `outputs` named for its branch:

```json
{"outputs": {"news": {"draft": "..."}, "stats": {"draft": "..."}}}
```

addressed as `{outputs/news/draft}` in a prompt and `outputs['news']['draft']`
in a condition. Any value the key already held at the top of `outputs` is left
untouched, and duplicate branch names are disambiguated. Merging happens in
branch order rather than completion order, so a run is reproducible.

As with `MapProcess`, each branch's copy of the results is shallow. A branch
that *mutates* a shared nested list or dict rather than rebinding a key is
writing to the same object as its siblings, concurrently — the merge cannot see
it and cannot order it. Rebind, do not mutate.

!!! warning "Agents are not thread-safe"

    `ChatToolAgent` keeps a `last_messages_buffer` on `self`, so two branches
    sharing one agent instance will interleave their transcripts. The
    `.response` text each step stores stays correct, but
    `ChatResponse.messages` does not. Give each branch its own agent when the
    transcript matters — `ParallelProcess` warns when branches would share one.

### Nested flow control in JSON

```json
{
  "process_type": "loop",
  "process_name": "refine",
  "mode": "until",
  "max_iterations": 3,
  "condition": "contains(lower(verdict), 'good')",
  "processes": [
    {
      "process_type": "conditional",
      "process_name": "gate",
      "condition": "len(draft) > 200",
      "then": [
        {
          "process_type": "sequential",
          "process_name": "shorten",
          "steps": [
            {
              "step_name": "draft",
              "system_message": "Edit tightly.",
              "prompt_template": "Shorten: {draft}"
            }
          ]
        }
      ]
    }
  ]
}
```

A bare string is accepted wherever a condition object is, so
`"condition": "score > 0.8"` and
`{"kind": "expression", "expression": "score > 0.8"}` mean the same thing.

## Declaring Agents and Endpoints

A pipeline can name the endpoints it runs against, so the JSON is
self-describing. Processes and steps then reference an agent by name:

```json
{
  "schema_version": 2,
  "agents": [
    {
      "name": "writer",
      "provider": {
        "type": "openai",
        "model": "qwen/qwen3.5-9b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "settings": {"temperature": 0.3, "top_p": 0.9}
      }
    },
    {
      "name": "judge",
      "provider": {
        "type": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "settings": {"max_tokens": 2048}
      }
    }
  ],
  "default_agent": "writer",
  "processes": [
    {
      "process_type": "sequential",
      "process_name": "work",
      "steps": [
        {
          "step_name": "draft",
          "system_message": "Write.",
          "prompt_template": "Draft {topic}"
        },
        {
          "step_name": "verdict",
          "system_message": "Judge.",
          "prompt_template": "Judge this: {draft}",
          "agent": "judge"
        }
      ]
    }
  ]
}
```

```python
pipeline = Pipeline.load_from_json("workflow.json")   # no agent argument needed
results = pipeline.run_pipeline(topic="otters")
```

### Provider types

`type` is one of `openai`, `anthropic`, `groq`, `mistral`, or an
OpenAI-compatible alias that supplies a default `base_url`: `openrouter`,
`together`, `deepseek`, `ollama`, `vllm`, and `openai_compatible` (which
requires an explicit `base_url`). `ollama` and `vllm` need no API key, since
local servers do not ask for one.

Every provider now accepts `base_url`, so an OpenAI- or Anthropic-shaped API can
be reached at any address — a gateway, a proxy, or a self-hosted server.

### Keys are never serialized

A provider config names the **environment variable** holding its key, never the
key itself. Each provider type has a conventional default (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and so on), overridable with `api_key_env`. A literal
`api_key` in the JSON is rejected, and a missing environment variable fails with
a message naming the variable to set. Agents are built on first reference, so a
document that merely *declares* an endpoint it never uses will not demand that
endpoint's key.

### Settings

`settings` values must already exist on the provider; a typo raises an error
listing the valid names rather than being silently ignored. To send a parameter
a provider does not declare — something a custom endpoint understands — use
`extra_settings`, which adds it as a request setting.

### Who wins when both are supplied

Agents injected from Python beat names declared in JSON *at the same level of
specificity*, and a more specific source beats a less specific one. In
descending priority:

1. `step_agents` entry for the step
2. the step's own JSON `agent` name
3. `process_agents` entry for the process
4. the process's own JSON `agent` name
5. `default_agent` passed from Python
6. the JSON `default_agent` name

Pass `build_agents=False` to ignore the `agents` block entirely and supply every
agent from Python.

!!! warning "Trust"

    Loading an `agents` block constructs API clients and reads environment
    variables; loading tool plugins imports Python modules. Only load pipeline
    JSON from trusted sources, or pass a prebuilt `PipelineToolRegistry` with
    `load_tool_plugins=False` and `build_agents=False`.

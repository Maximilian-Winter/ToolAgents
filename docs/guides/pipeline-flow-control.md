---
title: Pipeline Flow Control
---

# Pipeline Flow Control

A [pipeline](pipelines.md) runs its processes in order. These four processes
change that order: they branch, repeat, fan out over a list, or run work
concurrently — and because each one holds *other processes*, they nest freely.

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

## Conditions

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

## Loops

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

## Map

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

## Parallel

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

## Nested flow control in JSON

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


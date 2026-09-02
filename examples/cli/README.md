# tool-agents CLI examples

Three workspaces, each a complete `.tool-agents` folder you can run. They build
on each other: the first uses one folder, the last uses all five.

Every workflow reaches a real model, so each needs a key:

```bash
export OPENROUTER_API_KEY=...
```

or put it in a `.env` file inside the workspace, which is read automatically:

```
01-hello/.tool-agents/.env
```

```
OPENROUTER_API_KEY=sk-...
```

An exported variable wins over the file. Do not commit it.

Each example uses `qwen/qwen-2.5-7b-instruct` — a plain instruct model,
deliberately not a reasoning one. A reasoning model spends its token budget
thinking first, and if the budget runs out before it answers the step returns
an empty string. These examples are here to show the mechanism, not to debug a
token budget.

Each also sets `timeout: 120` and `max_tokens: 1200` on its providers. The
SDK default is 600 seconds with two retries — half an hour per step — so an
unbounded workflow can look like it has hung when one response stalls. Progress
is printed to stderr as each step runs; add `--quiet` to silence it.

The workspace is found by walking up from the working directory, so `cd` into an
example and run — no paths to pass.

---

## 01-hello — the smallest thing that runs

```
01-hello/.tool-agents/
  providers/models.json     one endpoint
  workflows/hello.json      one step, printed to stdout
```

```bash
cd 01-hello
tool-agents run hello --arg name=Max --arg topic=otters
```

The point: a workflow needs no Python. The provider file says which model to
call; the workflow says what to ask; `--arg` supplies the inputs, addressed in
the prompt as `{inputs/name}`.

The final step is a `stream` sink, which prints. Printing writes nothing outside
the process, so it needs no `--allow-writes`.

Try `tool-agents show hello` to see the inputs it expects without running it.

---

## 02-review — prompts, two models, a loop and a branch

```
02-review/.tool-agents/
  prompts/writer.md         reusable system prompts
  prompts/editor.md
  providers/models.json     two endpoints: writer (warm), critic (cold)
  workflows/review.json     draft -> review -> revise, until approved
```

```bash
cd 02-review
tool-agents run review --arg topic=otters --arg audience="curious adults"
```

What it adds:

**Prompts as files.** `prompts/editor.md` becomes `{prompts/editor}`, used as a
step's `system_message`. Editing the reviewer's instructions is editing a
Markdown file, not a JSON string.

**Two models, one workflow.** The critic runs at `temperature: 0.0` and the
writer at `0.7`. The critic step names its agent: `"agent": "critic"`.

**A loop with a real exit.** `mode: "until"` runs the body, *then* tests — the
condition reads `outputs['verdict']`, which does not exist until the body has
produced it. `max_iterations: 3` means an unhappy critic cannot spin forever.

**A guard inside the loop.** Because the whole body runs before the test, the
revise step sits inside a conditional on *not approved*. Without that guard the
loop rewrites the draft its critic just approved, and publishes a revision
nobody reviewed.

**A branch on the outcome.** If the loop exited approved, the draft prints to
stdout; if it ran out of revisions, the last complaint goes to stderr instead.

---

## 03-digest — every folder at once

```
03-digest/
  notes/                        input data, read at run time
  .tool-agents/
    tools/text_stats.py         tools the model may call
    prompts/*.md                three system prompts
    providers/models.json       two endpoints
    adapter/output/jsonl.py     a sink this project defines
    workflows/digest.json       folder -> chunks -> map -> parallel -> files
```

```bash
cd 03-digest
tool-agents run digest \
  --arg notes_dir=./notes \
  --arg out_dir=./out \
  --arg audience="curious adults" \
  --allow-writes
```

What it adds:

**A folder source with chunking.** `notes/*.md` is read and split by
`recursive_character` into 400-character chunks. Each chunk keeps its `path` and
`name` and gains a `chunk_index`.

**A map over the chunks.** Each is reduced to one sentence, collected into
`outputs/points`.

**A parallel fan-out.** Title and body are written concurrently by *different*
agents — the title by `stylist` at `temperature: 0.8`, the body by `worker` at
`0.2`. Each branch names its own agent deliberately: branches run in threads,
and an agent instance is not thread-safe.

**Tools from the workspace.** `tools/text_stats.py` becomes a plugin named for
the file, referenced as `{"plugin": "text_stats", "tool_name": "ReadingMinutes"}`.

**Assembly without a model.** The title and body are joined by a `template`
process, which renders `{outputs/title}` and `{outputs/body}` into one string.
Concatenating two results is string work; paying a model to "return this
unchanged" costs a request and may not comply.

**A sink this project invented.** `adapter/output/jsonl.py` registers a `jsonl`
type; the workflow then uses it exactly like a built-in. That file is worth
reading — it is the whole extension mechanism, about forty lines.

**`--allow-writes`.** This is the first example that writes files. Without the
flag the run stops at the first file sink and says so. Reading is always
permitted; writing and HTTP are not, because a workflow file that reads is a
privacy question while one that writes destroys data.

---

## Starting your own

```bash
mkdir my-project && cd my-project
tool-agents init
tool-agents list
```

`init` creates the folders empty. Copy a workflow out of `01-hello` to get
going, and see the [CLI guide](https://maximilian-winter.github.io/ToolAgents/guides/cli/)
for what each folder does.

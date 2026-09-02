---
title: Pipeline Endpoints
---

# Pipeline Endpoints

A [pipeline](pipelines.md) normally takes its agent from Python at load time.
It can instead declare the providers, models, endpoints and sampling settings
it runs against, so the workflow file is self-contained and
`Pipeline.load_from_json(path)` needs no agent argument.

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

## Provider types

`type` is one of `openai`, `anthropic`, `groq`, `mistral`, or an
OpenAI-compatible alias that supplies a default `base_url`: `openrouter`,
`together`, `deepseek`, `ollama`, `vllm`, and `openai_compatible` (which
requires an explicit `base_url`). `ollama` and `vllm` need no API key, since
local servers do not ask for one.

Every provider now accepts `base_url`, so an OpenAI- or Anthropic-shaped API can
be reached at any address — a gateway, a proxy, or a self-hosted server.

## Keys are never serialized

A provider config names the **environment variable** holding its key, never the
key itself. Each provider type has a conventional default (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and so on), overridable with `api_key_env`. A literal
`api_key` in the JSON is rejected, and a missing environment variable fails with
a message naming the variable to set. Agents are built on first reference, so a
document that merely *declares* an endpoint it never uses will not demand that
endpoint's key.

## Settings

`settings` values must already exist on the provider; a typo raises an error
listing the valid names rather than being silently ignored. To send a parameter
a provider does not declare — something a custom endpoint understands — use
`extra_settings`, which adds it as a request setting.

## Who wins when both are supplied

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

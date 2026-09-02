---
title: Pipelines API
---

# Pipelines API

A pipeline is a multi-step workflow that can be described in code or in JSON.
See the [pipelines guide](../guides/pipelines.md) for the narrative version;
this page is the reference.

```python
from ToolAgents.pipelines import Pipeline, SequentialProcess, ProcessStep
```

## Pipeline

::: ToolAgents.pipelines.pipeline.Pipeline

## Results

Results are carried in named sections — `inputs`, `outputs`, `vars` — addressed
by path in prompt templates (`{outputs/draft}`) and by subscript in conditions
(`outputs['draft']`). A bare name still resolves innermost-first.

::: ToolAgents.pipelines.results.PipelineResults

## Processes

Every process implements `run_process` and serializes to JSON. Flow-control
processes hold *other processes*, so they nest freely.

::: ToolAgents.pipelines.pipeline.Process

::: ToolAgents.pipelines.pipeline.ProcessStep

::: ToolAgents.pipelines.pipeline.SequentialProcess

::: ToolAgents.pipelines.pipeline.TemplateProcess

### Flow control

::: ToolAgents.pipelines.flow.FlowProcess

::: ToolAgents.pipelines.flow.ConditionalProcess

::: ToolAgents.pipelines.flow.LoopProcess

::: ToolAgents.pipelines.flow.MapProcess

::: ToolAgents.pipelines.flow.ParallelProcess

## Conditions

Conditions are compiled from a whitelisted subset of Python's grammar, never
`eval`'d, so pipeline JSON from an untrusted source cannot execute arbitrary
code.

::: ToolAgents.pipelines.conditions.SafeExpression

::: ToolAgents.pipelines.conditions.Condition

::: ToolAgents.pipelines.conditions.ExpressionCondition

::: ToolAgents.pipelines.conditions.condition_from_config

::: ToolAgents.pipelines.conditions.register_condition_kind

## Declaring agents and endpoints

A pipeline document can name the providers it runs against. API keys are never
serialized — a config names the environment variable holding one.

### ProviderConfig

::: ToolAgents.pipelines.agent_config.ProviderConfig

::: ToolAgents.pipelines.agent_config.AgentConfig

::: ToolAgents.pipelines.agent_config.ProviderSpec

::: ToolAgents.pipelines.agent_config.LazyAgentRegistry

::: ToolAgents.pipelines.agent_config.register_provider_spec

## Sources and sinks

A source loads data into the `inputs` section; a sink emits a value out of the
results. Both are processes, so they nest inside flow control. Writing is gated
behind `allow_writes` — see the
[sources and sinks guide](../guides/pipeline-io.md).

::: ToolAgents.pipelines.data_io.SourceProcess

::: ToolAgents.pipelines.data_io.SinkProcess

### Source types

::: ToolAgents.pipelines.data_io.Source

::: ToolAgents.pipelines.data_io.TextSource

::: ToolAgents.pipelines.data_io.FileSource

::: ToolAgents.pipelines.data_io.FilesSource

::: ToolAgents.pipelines.data_io.FolderSource

::: ToolAgents.pipelines.data_io.register_source_type

### Sink types

::: ToolAgents.pipelines.data_io.Sink

::: ToolAgents.pipelines.data_io.StreamSink

::: ToolAgents.pipelines.data_io.FileSink

::: ToolAgents.pipelines.data_io.FilesSink

::: ToolAgents.pipelines.data_io.HttpSink

::: ToolAgents.pipelines.data_io.register_sink_type

### Path placeholders

::: ToolAgents.pipelines.data_io.render_path

::: ToolAgents.pipelines.data_io.normalize_placeholders

## Chunking

::: ToolAgents.pipelines.chunking.build_splitter

::: ToolAgents.pipelines.chunking.SplitterSpec

::: ToolAgents.pipelines.chunking.register_splitter_spec

## Tools

::: ToolAgents.pipelines.pipeline.PipelineToolRegistry

::: ToolAgents.pipelines.pipeline.PipelineToolPlugin

::: ToolAgents.pipelines.pipeline.load_pipeline_tools_from_spec

## Loading and extension

::: ToolAgents.pipelines.pipeline.PipelineLoadContext

::: ToolAgents.pipelines.pipeline.register_process_type

::: ToolAgents.pipelines.pipeline.get_process_type

::: ToolAgents.pipelines.pipeline.process_from_dict

## Errors

::: ToolAgents.pipelines.pipeline.PipelineSerializationError

::: ToolAgents.pipelines.pipeline.PipelineExecutionError

::: ToolAgents.pipelines.conditions.PipelineConditionError

::: ToolAgents.pipelines.agent_config.AgentConfigurationError

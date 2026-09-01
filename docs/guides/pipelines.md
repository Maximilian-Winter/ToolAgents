---
title: Pipelines
---

# Pipelines

Pipelines let you describe a multi-step agent workflow and save the workflow
shape to JSON. Pipeline JSON stores process names, step names, system messages,
prompt templates, and references to tools. It does not store live agent objects;
pass agents back in when loading a runnable pipeline.

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
  "schema_version": 1,
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

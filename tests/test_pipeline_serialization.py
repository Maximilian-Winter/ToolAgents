from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool
from ToolAgents.pipelines import (
    Pipeline,
    PipelineSerializationError,
    PipelineToolRegistry,
    ProcessStep,
    SequentialProcess,
)
from ToolAgents.tool_adapters.execution import execute_tool_by_name


class AddNumbers(BaseModel):
    """Add two numbers."""

    a: int = Field(..., description="First number.")
    b: int = Field(..., description="Second number.")

    def run(self) -> int:
        return self.a + self.b


class EchoAgent:
    def get_response(self, messages, tool_registry=None):
        prompt = messages[-1].get_as_text()
        if tool_registry is None:
            return SimpleNamespace(response=prompt)

        result = execute_tool_by_name(
            tool_registry,
            "AddNumbers",
            {"a": 2, "b": 3},
        )
        return SimpleNamespace(response=f"{prompt} = {result}")


def test_pipeline_dict_roundtrip_with_registered_tools():
    add_tool = FunctionTool(AddNumbers)
    tool_registry = PipelineToolRegistry().register_plugin(
        "math",
        [add_tool],
    )

    step = ProcessStep(
        step_name="sum",
        system_message="You do arithmetic.",
        prompt_template="Add {a} and {b}",
        tools=[add_tool],
    )
    process = SequentialProcess(process_name="calculator", agent=EchoAgent())
    process.add_step(step)
    pipeline = Pipeline()
    pipeline.add_process(process)

    data = pipeline.to_dict(tool_registry=tool_registry)
    assert data == {
        "schema_version": 2,
        "processes": [
            {
                "process_type": "sequential",
                "process_name": "calculator",
                "steps": [
                    {
                        "step_name": "sum",
                        "system_message": "You do arithmetic.",
                        "prompt_template": "Add {a} and {b}",
                        "tools": [{"plugin": "math", "tool_name": "AddNumbers"}],
                    }
                ],
            }
        ],
        "tool_plugins": [],
    }

    loaded = Pipeline.from_dict(
        data,
        tool_registry=tool_registry,
        default_agent=EchoAgent(),
        load_tool_plugins=False,
    )

    assert loaded.run_pipeline(a=2, b=3)["sum"] == "Add 2 and 3 = 5"


def test_pipeline_json_roundtrip_loads_tool_plugin(tmp_path: Path, monkeypatch):
    module_path = tmp_path / "pipeline_tools.py"
    module_path.write_text(
        '''
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
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    source = "pipeline_tools:create_tools"
    add_tool = PipelineToolRegistry().load_plugin("math", source).get_tool(
        "math",
        "AddNumbers",
    )
    assert add_tool is not None

    process = SequentialProcess(process_name="calculator", agent=EchoAgent())
    process.add_step(
        ProcessStep(
            step_name="sum",
            system_message="You do arithmetic.",
            prompt_template="Add {a} and {b}",
            tools=[add_tool],
        )
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    json_path = tmp_path / "pipeline.json"
    pipeline.save_to_json(
        json_path,
        tool_registry=PipelineToolRegistry().load_plugin("math", source),
    )

    loaded = Pipeline.load_from_json(json_path, default_agent=EchoAgent())

    assert loaded.run_pipeline(a=2, b=3)["sum"] == "Add 2 and 3 = 5"


def test_serializing_unregistered_tools_requires_registry_entry():
    pipeline = Pipeline()
    process = SequentialProcess(process_name="calculator")
    process.add_step(
        ProcessStep(
            step_name="sum",
            system_message="You do arithmetic.",
            prompt_template="Add {a} and {b}",
            tools=[FunctionTool(AddNumbers)],
        )
    )
    pipeline.add_process(process)

    with pytest.raises(PipelineSerializationError, match="PipelineToolRegistry"):
        pipeline.to_dict()

    with pytest.raises(PipelineSerializationError, match="not registered"):
        pipeline.to_dict(tool_registry=PipelineToolRegistry())

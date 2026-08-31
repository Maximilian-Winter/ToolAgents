from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool, ToolRegistry
from ToolAgents.agent_memory.navigable_memory import InMemoryBackend, NavigableMemory
from ToolAgents.tool_adapters.execution import (
    async_execute_tool_by_name,
    execute_tool_by_name,
    find_tool,
    normalize_tools,
)
from ToolAgents.tool_adapters.mcp_server import create_mcp_tool_callable
from ToolAgents.tool_adapters.schemas import (
    function_tool_to_input_schema,
    function_tool_to_mcp_metadata,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class AddNumbers(BaseModel):
    """Add two numbers."""

    a: int = Field(..., description="First number.")
    b: int = Field(..., description="Second number.")

    def run(self) -> int:
        return self.a + self.b


class EchoText(BaseModel):
    """Echo text."""

    text: str = Field(..., description="Text to echo.")

    def run(self) -> str:
        return self.text


def test_execute_tool_by_name_accepts_registry_and_iterables():
    add_tool = FunctionTool(AddNumbers)
    registry = ToolRegistry()
    registry.add_tool(add_tool)

    assert execute_tool_by_name(registry, "AddNumbers", {"a": 2, "b": 3}) == 5
    assert execute_tool_by_name([add_tool], "AddNumbers", {"a": 4, "b": 5}) == 9
    assert find_tool([add_tool], "Missing") is None


@pytest.mark.anyio
async def test_async_execute_tool_by_name_handles_sync_tools():
    result = await async_execute_tool_by_name(
        [FunctionTool(EchoText)],
        "EchoText",
        {"text": "hello"},
    )

    assert result == "hello"


def test_schema_helpers_return_portable_metadata():
    tool = FunctionTool(AddNumbers)

    schema = function_tool_to_input_schema(tool)
    metadata = function_tool_to_mcp_metadata(tool)

    assert schema["type"] == "object"
    assert schema["properties"]["a"]["description"] == "First number."
    assert metadata.name == "AddNumbers"
    assert metadata.description == "Add two numbers."
    assert metadata.input_schema == schema


@pytest.mark.anyio
async def test_mcp_callable_wrapper_uses_tool_signature_and_execution():
    tool = FunctionTool(AddNumbers)
    wrapper = create_mcp_tool_callable(tool, [tool])

    assert wrapper.__name__ == "AddNumbers"
    assert list(wrapper.__signature__.parameters) == ["a", "b"]
    assert await wrapper(a=10, b=7) == "17"


def test_navigable_memory_tools_are_adapter_compatible():
    memory = NavigableMemory(InMemoryBackend())
    tools = normalize_tools(FunctionTool(tool) for tool in memory.create_tools())

    names = {tool.model.__name__ for tool in tools}

    assert "ReadDocument" in names
    assert "WriteDocument" in names
    assert "SearchKnowledge" in names


def test_legacy_mcp_session_imports_are_available():
    from ToolAgents.utilities.mcp_session import MCPServerTools, MCPTool, SessionManager

    assert MCPServerTools is not None
    assert MCPTool is not None
    assert SessionManager is not None


def test_cli_can_list_schema_and_call_tools(tmp_path: Path):
    module_path = tmp_path / "sample_tools.py"
    module_path.write_text(
        """
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool


class RepeatText(BaseModel):
    \"\"\"Repeat text.\"\"\"

    text: str = Field(..., description=\"Text to repeat.\")
    times: int = Field(1, description=\"Repeat count.\")

    def run(self) -> str:
        return self.text * self.times


def create_tools():
    return [FunctionTool(RepeatText)]
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), str(tmp_path), env.get("PYTHONPATH", "")]
    )

    list_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ToolAgents.tool_adapters.cli",
            "list",
            "--module",
            "sample_tools:create_tools",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert list_result.returncode == 0, list_result.stderr
    assert "RepeatText - Repeat text." in list_result.stdout

    schema_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ToolAgents.tool_adapters.cli",
            "schema",
            "RepeatText",
            "--module",
            "sample_tools:create_tools",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert schema_result.returncode == 0, schema_result.stderr
    assert json.loads(schema_result.stdout)["properties"]["text"]["description"] == (
        "Text to repeat."
    )

    call_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ToolAgents.tool_adapters.cli",
            "call",
            "RepeatText",
            "--module",
            "sample_tools:create_tools",
            "--json",
            '{"text": "ha", "times": 3}',
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert call_result.returncode == 0, call_result.stderr
    assert call_result.stdout.strip() == "hahaha"

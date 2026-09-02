"""Tests for the .tool-agents workspace and the tool-agents CLI."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ToolAgents.cli import collect_arguments, main, parse_argument
from ToolAgents.workspace import (
    DEFAULT_WORKSPACE_DIRNAME,
    Workspace,
    WorkspaceError,
    WorkspaceNotFoundError,
)

MATH_TOOL_MODULE = '''
from pydantic import BaseModel, Field
from ToolAgents import FunctionTool


class AddNumbers(BaseModel):
    """Add two numbers."""

    a: int = Field(..., description="First.")
    b: int = Field(..., description="Second.")

    def run(self) -> int:
        return self.a + self.b


TOOLS = [FunctionTool(AddNumbers)]
'''

SHOUT_ADAPTER = '''
from ToolAgents.pipelines import Source, register_source_type


@register_source_type
class ShoutSource(Source):
    """A source that shouts."""

    source_type = "shout"
    yields_text = True

    def __init__(self, text):
        self.text = text

    def load(self, results):
        return self.text.upper()

    def to_dict(self):
        return {"type": "shout", "text": self.text}

    @classmethod
    def from_dict(cls, data):
        return cls(str(data["text"]))
'''


class StubAgent:
    """Reports what it was given, so prompts and tools are inspectable."""

    def get_response(self, messages, tool_registry=None, **kwargs):
        tools = len(tool_registry.get_tools()) if tool_registry else 0
        return SimpleNamespace(
            response=f"sys={messages[0].get_as_text()}|"
            f"user={messages[-1].get_as_text()}|tools={tools}"
        )


@pytest.fixture
def workspace(tmp_path):
    """A fully populated workspace."""

    root = Workspace.create(tmp_path).root
    (root / "prompts" / "reviewer.md").write_text(
        "You are a strict editor.", encoding="utf-8"
    )
    (root / "tools" / "math.py").write_text(MATH_TOOL_MODULE, encoding="utf-8")
    (root / "adapter" / "input" / "shout.py").write_text(
        SHOUT_ADAPTER, encoding="utf-8"
    )
    (root / "providers" / "shared.json").write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "name": "writer",
                        "provider": {
                            "type": "openrouter",
                            "model": "qwen/qwen3.5-9b",
                            "api_key_env": "OPENROUTER_API_KEY",
                        },
                    }
                ],
                "default_agent": "writer",
            }
        ),
        encoding="utf-8",
    )
    (root / "workflows" / "digest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "processes": [
                    {
                        "process_type": "source",
                        "process_name": "load",
                        "source": {"type": "shout", "text": "hello"},
                        "result_key": "greeting",
                    },
                    {
                        "process_type": "sequential",
                        "process_name": "work",
                        "steps": [
                            {
                                "step_name": "verdict",
                                "system_message": "{prompts/reviewer}",
                                "prompt_template": "On {inputs/topic}: {inputs/greeting}",
                                "tools": [
                                    {"plugin": "math", "tool_name": "AddNumbers"}
                                ],
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return Workspace(root)


# ---------------------------------------------------------------------------
# Discovery and scaffolding
# ---------------------------------------------------------------------------


def test_create_scaffolds_every_folder(tmp_path):
    workspace = Workspace.create(tmp_path)

    assert workspace.root.name == DEFAULT_WORKSPACE_DIRNAME
    for name in ("workflows", "tools", "prompts", "providers"):
        assert (workspace.root / name).is_dir()
    assert (workspace.root / "adapter" / "input").is_dir()
    assert (workspace.root / "adapter" / "output").is_dir()


def test_discover_walks_up_from_a_nested_directory(tmp_path):
    Workspace.create(tmp_path)
    nested = tmp_path / "src" / "deep" / "here"
    nested.mkdir(parents=True)

    assert Workspace.discover(nested).root == tmp_path / DEFAULT_WORKSPACE_DIRNAME


def test_discover_says_where_it_looked(tmp_path):
    with pytest.raises(WorkspaceNotFoundError, match="tool-agents init"):
        Workspace.discover(tmp_path)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


def test_summary_lists_every_kind(workspace):
    summary = workspace.summary()

    assert summary["workflows"] == ["digest"]
    assert summary["tools"] == ["math"]
    assert summary["prompts"] == ["reviewer"]
    assert summary["providers"] == ["shared"]
    assert summary["adapters"] == ["input/shout"]


def test_unknown_workflow_lists_the_known_ones(workspace):
    with pytest.raises(WorkspaceError, match="Available: digest"):
        workspace.workflow_path("nope")


def test_prompts_load_by_file_stem(workspace):
    assert workspace.load_prompts() == {"reviewer": "You are a strict editor."}


def test_duplicate_prompt_stems_are_refused(workspace):
    (workspace.root / "prompts" / "reviewer.txt").write_text("other", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="both named 'reviewer'"):
        workspace.load_prompts()


def test_tool_modules_become_plugins_named_for_the_file(workspace):
    registry = workspace.build_tool_registry()

    assert registry.get_tool("math", "AddNumbers") is not None


def test_a_tool_module_may_use_create_tools(workspace):
    (workspace.root / "tools" / "extra.py").write_text(
        MATH_TOOL_MODULE.replace("TOOLS = [", "def create_tools():\n    return ["),
        encoding="utf-8",
    )
    registry = workspace.build_tool_registry()

    assert registry.get_tool("extra", "AddNumbers") is not None


def test_a_tool_module_exposing_nothing_says_so(workspace):
    (workspace.root / "tools" / "empty.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="exposes no tools"):
        workspace.build_tool_registry()


def test_a_broken_tool_module_names_the_file(workspace):
    (workspace.root / "tools" / "broken.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8"
    )

    with pytest.raises(WorkspaceError, match="broken.py"):
        workspace.build_tool_registry()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_shared_providers_are_merged_into_a_workflow(workspace):
    document = workspace.load_workflow_document("digest")

    assert [a["name"] for a in document["agents"]] == ["writer"]
    assert document["default_agent"] == "writer"


def test_a_workflow_can_override_a_shared_agent(workspace):
    path = workspace.workflow_path("digest")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["agents"] = [
        {"name": "writer", "provider": {"type": "anthropic", "model": "claude-x"}}
    ]
    path.write_text(json.dumps(document), encoding="utf-8")

    merged = workspace.load_workflow_document("digest")
    assert len(merged["agents"]) == 1
    assert merged["agents"][0]["provider"]["type"] == "anthropic"


def test_an_agent_declared_twice_across_providers_is_refused(workspace):
    (workspace.root / "providers" / "again.json").write_text(
        json.dumps(
            [
                {
                    "name": "writer",
                    "provider": {"type": "anthropic", "model": "claude-x"},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceError, match="declared twice"):
        workspace.load_agent_configs()


def test_a_malformed_provider_file_names_itself(workspace):
    (workspace.root / "providers" / "bad.json").write_text("{oops", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="bad.json"):
        workspace.load_agent_configs()


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_run_workflow_wires_prompts_tools_and_adapters(workspace):
    results = workspace.run_workflow(
        "digest",
        {"topic": "otters"},
        build_agents=False,
        default_agent=StubAgent(),
    )

    response = results["outputs/verdict"]
    assert "sys=You are a strict editor." in response  # prompts/ reached system_message
    assert "On otters: HELLO" in response  # adapter/input/ registered "shout"
    assert "tools=1" in response  # tools/ became a usable plugin


def test_prompts_land_in_their_own_section(workspace):
    results = workspace.run_workflow(
        "digest", {"topic": "x"}, build_agents=False, default_agent=StubAgent()
    )

    assert results.section("prompts") == {"reviewer": "You are a strict editor."}
    assert "reviewer" not in results.inputs


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pair, expected",
    [
        ("name=Ada", ("name", "Ada")),
        ("n=3", ("n", 3)),
        ("ratio=0.5", ("ratio", 0.5)),
        ("flag=true", ("flag", True)),
        ('tags=["a","b"]', ("tags", ["a", "b"])),
        ("path=C:/tmp/x.md", ("path", "C:/tmp/x.md")),
        ("empty=", ("empty", "")),
    ],
)
def test_arguments_decode_as_json_when_they_can(pair, expected):
    assert parse_argument(pair) == expected


def test_an_argument_without_a_name_is_refused():
    with pytest.raises(ValueError, match="key=value"):
        parse_argument("novalue")


def test_argument_sources_merge_with_arg_winning(tmp_path):
    json_file = tmp_path / "args.json"
    json_file.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")

    merged = collect_arguments(["b=99"], json.dumps({"c": 3}), str(json_file))
    assert merged == {"a": 1, "b": 99, "c": 3}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_init_creates_a_workspace(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    assert (tmp_path / DEFAULT_WORKSPACE_DIRNAME / "workflows").is_dir()
    assert "Created" in capsys.readouterr().out


def test_cli_list_shows_every_member(workspace, capsys):
    assert main(["--workspace", str(workspace.root), "list"]) == 0

    out = capsys.readouterr().out
    for expected in ("digest", "math", "reviewer", "shared", "input/shout"):
        assert expected in out


def test_cli_list_can_filter_to_one_kind(workspace, capsys):
    assert main(["--workspace", str(workspace.root), "list", "workflows"]) == 0

    out = capsys.readouterr().out
    assert "digest" in out
    assert "input/shout" not in out


def test_cli_show_reports_agents_processes_and_inputs(workspace, capsys):
    assert main(["--workspace", str(workspace.root), "show", "digest"]) == 0

    out = capsys.readouterr().out
    assert "writer: openrouter qwen/qwen3.5-9b" in out
    assert "source: load" in out
    assert "topic" in out


def test_cli_reports_an_unknown_workflow_without_a_traceback(workspace, capsys):
    assert main(["--workspace", str(workspace.root), "show", "ghost"]) == 2
    assert "Available: digest" in capsys.readouterr().err


def test_cli_without_a_workspace_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # A parent of tmp_path could hold a workspace; point at a definitely-empty one.
    assert main(["--workspace", str(tmp_path / "nothing"), "list"]) == 2
    assert "Not a workspace directory" in capsys.readouterr().err


def test_cli_tools_list_reads_a_workspace_module(workspace, capsys):
    exit_code = main(
        ["--workspace", str(workspace.root), "tools", "list", "--plugin", "math"]
    )

    assert exit_code == 0
    assert "AddNumbers" in capsys.readouterr().out


def test_cli_tools_call_runs_a_workspace_tool(workspace, capsys):
    exit_code = main(
        [
            "--workspace",
            str(workspace.root),
            "tools",
            "call",
            "AddNumbers",
            "--plugin",
            "math",
            "--json",
            '{"a": 2, "b": 3}',
        ]
    )

    assert exit_code == 0
    assert "5" in capsys.readouterr().out


def test_cli_tools_requires_a_source(workspace, capsys):
    assert main(["--workspace", str(workspace.root), "tools", "list"]) == 1
    assert "--module or --plugin" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The deprecated entry point
# ---------------------------------------------------------------------------


def test_the_old_cli_still_works_and_says_it_moved(tmp_path, capsys):
    module = tmp_path / "legacy_tools.py"
    module.write_text(MATH_TOOL_MODULE, encoding="utf-8")

    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        from ToolAgents.tool_adapters.cli import main as legacy_main

        assert legacy_main(["list", "--module", "legacy_tools:TOOLS"]) == 0
    finally:
        sys.path.remove(str(tmp_path))

    captured = capsys.readouterr()
    assert "AddNumbers" in captured.out
    assert "deprecated" in captured.err
    assert "tool-agents tools" in captured.err


# ---------------------------------------------------------------------------
# The shipped CLI examples
# ---------------------------------------------------------------------------

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples" / "cli"


class ScriptedAgent:
    """Echoes prompts, and approves on the second review."""

    def __init__(self) -> None:
        self.reviews = 0

    def get_response(self, messages, tool_registry=None, **kwargs):
        user = messages[-1].get_as_text()
        if user.startswith("Revision"):
            self.reviews += 1
            return SimpleNamespace(
                response="APPROVED" if self.reviews >= 2 else "Too vague."
            )
        return SimpleNamespace(response="<" + user.splitlines()[0][:40] + ">")


requires_examples = pytest.mark.skipif(
    not EXAMPLES.is_dir(), reason="CLI examples not present"
)


@requires_examples
def test_every_cli_example_workspace_lists_cleanly():
    for example in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
        workspace = Workspace(example / ".tool-agents")
        summary = workspace.summary()
        assert summary["workflows"], f"{example.name} declares no workflow"
        assert summary["providers"], f"{example.name} declares no provider"


@requires_examples
def test_hello_example_runs():
    workspace = Workspace(EXAMPLES / "01-hello" / ".tool-agents")
    results = workspace.run_workflow(
        "hello",
        {"name": "Max", "topic": "otters"},
        build_agents=False,
        default_agent=ScriptedAgent(),
    )

    assert "Max" in results["outputs/greeting"]


@requires_examples
def test_review_example_loops_then_takes_the_approved_branch():
    workspace = Workspace(EXAMPLES / "02-review" / ".tool-agents")
    results = workspace.run_workflow(
        "review",
        {"topic": "otters", "audience": "curious adults"},
        build_agents=False,
        default_agent=ScriptedAgent(),
    )

    assert results["outputs/refine_iterations"] == 2
    assert results["outputs/approved"] is True


@requires_examples
def test_digest_example_uses_every_workspace_folder(tmp_path):
    example = EXAMPLES / "03-digest"
    workspace = Workspace(example / ".tool-agents")

    # tools/ and adapter/ are wired before the run.
    assert workspace.build_tool_registry().get_tool("text_stats", "ReadingMinutes")
    assert "output/jsonl" in workspace.load_adapters()

    results = workspace.run_workflow(
        "digest",
        {
            "notes_dir": str(example / "notes"),
            "out_dir": str(tmp_path),
            "audience": "curious adults",
        },
        build_agents=False,
        default_agent=ScriptedAgent(),
        allow_writes=True,
    )

    assert len(results["inputs/chunks"]) >= 2  # folder source + splitter
    assert len(results["outputs/points"]) == len(results["inputs/chunks"])  # map
    assert (tmp_path / "digest.md").is_file()  # file sink
    assert (tmp_path / "points.jsonl").is_file()  # the workspace's own sink
    assert "{prompts/" not in results["outputs/digest"]  # prompts resolved


@requires_examples
def test_digest_example_refuses_to_write_without_the_flag(tmp_path):
    example = EXAMPLES / "03-digest"
    workspace = Workspace(example / ".tool-agents")

    with pytest.raises(Exception, match="allow_writes=True"):
        workspace.run_workflow(
            "digest",
            {
                "notes_dir": str(example / "notes"),
                "out_dir": str(tmp_path),
                "audience": "x",
            },
            build_agents=False,
            default_agent=ScriptedAgent(),
        )
    assert not (tmp_path / "digest.md").exists()


@requires_examples
def test_parallel_branches_in_the_digest_example_get_distinct_agents(monkeypatch):
    """Branches run in threads, so sharing one agent would be a race."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-used")
    workspace = Workspace(EXAMPLES / "03-digest" / ".tool-agents")
    pipeline = workspace.load_pipeline("digest")

    parallel = next(p for p in pipeline.processes if p.process_type == "parallel")
    agents = [branch.agent for branch in parallel.branches]
    assert agents[0] is not agents[1]


# ---------------------------------------------------------------------------
# Environment files
# ---------------------------------------------------------------------------


def test_a_workspace_env_file_is_loaded_before_agents_are_built(
    workspace, monkeypatch
):
    """Keys live beside the workflows that need them, in one gitignorable path."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (workspace.root / ".env").write_text(
        "OPENROUTER_API_KEY=from-workspace-env\n", encoding="utf-8"
    )

    pipeline = workspace.load_pipeline("digest")
    agent = pipeline.processes[1].agent
    assert agent.chat_api.client.api_key == "from-workspace-env"


def test_a_missing_workspace_env_file_is_not_an_error(workspace, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "exported")
    assert not (workspace.root / ".env").exists()

    assert workspace.load_pipeline("digest") is not None


def test_an_explicit_env_file_overrides_the_conventional_one(
    workspace, tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    explicit = tmp_path / "other.env"
    explicit.write_text("OPENROUTER_API_KEY=from-explicit\n", encoding="utf-8")

    pipeline = workspace.load_pipeline("digest", env_file=str(explicit))
    assert pipeline.processes[1].agent.chat_api.client.api_key == "from-explicit"

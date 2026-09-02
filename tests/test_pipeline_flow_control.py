"""Tests for pipeline flow control, condition sandboxing, and agent configs."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from ToolAgents.pipelines import (
    AgentConfigurationError,
    ConditionalProcess,
    LoopProcess,
    MapProcess,
    ParallelProcess,
    Pipeline,
    PipelineConditionError,
    PipelineExecutionError,
    PipelineSerializationError,
    ProcessStep,
    SafeExpression,
    SequentialProcess,
)


class EchoAgent:
    """Returns the rendered prompt, so results are inspectable."""

    def __init__(self, tag: str = "echo") -> None:
        self.tag = tag

    def get_response(self, messages, tool_registry=None, **kwargs):
        return SimpleNamespace(
            response=f"{self.tag}:{messages[-1].get_as_text()}"
        )


class CountdownAgent:
    """Answers 'no' a fixed number of times, then 'yes'."""

    def __init__(self, approve_on: int = 3) -> None:
        self.approve_on = approve_on
        self.calls = 0

    def get_response(self, messages, tool_registry=None, **kwargs):
        self.calls += 1
        verdict = "yes" if self.calls >= self.approve_on else "no"
        return SimpleNamespace(response=verdict)


def step(name: str, template: str, agent=None, agent_name=None) -> ProcessStep:
    return ProcessStep(
        step_name=name,
        system_message="system",
        prompt_template=template,
        agent=agent,
        agent_name=agent_name,
    )


# ---------------------------------------------------------------------------
# Condition sandbox
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "().__class__.__bases__[0].__subclasses__()",
        "draft.__class__",
        "draft.upper()",
        "(lambda: 1)()",
        "[value for value in draft]",
        "open('secrets.txt')",
        "eval('1')",
        "globals()",
        "2 ** 999999999",
        "(walrus := 5)",
        "f'{draft}'",
        "len(draft * 100000000) > 0",
    ],
)
def test_sandbox_rejects_unsafe_expressions(expression):
    with pytest.raises(PipelineConditionError):
        SafeExpression(expression)


@pytest.mark.parametrize(
    "expression, results, expected",
    [
        ("score > 0.8", {"score": 0.9}, True),
        ("score > 0.8", {"score": 0.1}, False),
        ("len(draft) < 20", {"draft": "short"}, True),
        ("contains(lower(draft), 'error')", {"draft": "An ERROR"}, True),
        ("status in ['ok', 'done']", {"status": "done"}, True),
        ("items[0] == 3", {"items": [3, 4]}, True),
        ("not is_empty(draft)", {"draft": ""}, False),
        ("default(missing, 0) < 5", {"missing": None}, True),
        ("draft * 3 == 'aaa'", {"draft": "a"}, True),
    ],
)
def test_sandbox_evaluates_safe_expressions(expression, results, expected):
    assert SafeExpression(expression).evaluate(results) is expected


def test_sandbox_names_missing_results():
    with pytest.raises(PipelineConditionError, match="unknown result"):
        SafeExpression("score > 1").evaluate({"other": 1})


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------


def test_conditional_runs_matching_branch():
    process = ConditionalProcess(
        condition="score < 0.7",
        process_name="gate",
        agent=EchoAgent(),
        then_steps=[step("action", "revise {topic}")],
        else_steps=[step("action", "publish {topic}")],
        record_as="was_low",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    low = pipeline.run_pipeline(score=0.5, topic="birds")
    assert low["action"] == "echo:revise birds"
    assert low["was_low"] is True

    high = pipeline.run_pipeline(score=0.9, topic="birds")
    assert high["action"] == "echo:publish birds"
    assert high["was_low"] is False


def test_conditional_with_no_else_branch_is_a_noop():
    process = ConditionalProcess(
        condition="flag",
        process_name="gate",
        agent=EchoAgent(),
        then_steps=[step("action", "ran")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert "action" not in pipeline.run_pipeline(flag=False)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def test_loop_until_runs_body_at_least_once_and_stops_when_satisfied():
    process = LoopProcess(
        condition="verdict == 'yes'",
        mode="until",
        max_iterations=6,
        process_name="refine",
        agent=CountdownAgent(approve_on=3),
        steps=[step("verdict", "check pass {iteration}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline()
    assert results["verdict"] == "yes"
    assert results["refine_iterations"] == 3


def test_loop_respects_max_iterations_cap():
    process = LoopProcess(
        condition="verdict == 'never'",
        mode="until",
        max_iterations=2,
        process_name="refine",
        agent=CountdownAgent(approve_on=99),
        steps=[step("verdict", "check")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert pipeline.run_pipeline()["refine_iterations"] == 2


def test_loop_can_error_when_cap_is_reached():
    process = LoopProcess(
        condition="verdict == 'never'",
        mode="until",
        max_iterations=2,
        process_name="refine",
        agent=CountdownAgent(approve_on=99),
        on_max_iterations="error",
        steps=[step("verdict", "check")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    with pytest.raises(PipelineExecutionError, match="without satisfying"):
        pipeline.run_pipeline()


def test_loop_without_condition_runs_a_fixed_number_of_times():
    process = LoopProcess(
        max_iterations=3,
        process_name="fixed",
        agent=EchoAgent(),
        steps=[step("out", "pass {iteration}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline()
    assert results["fixed_iterations"] == 3
    assert results["out"] == "echo:pass 2"


def test_while_loop_explains_the_pre_test_trap():
    process = LoopProcess(
        condition="verdict == 'yes'",
        mode="while",
        process_name="refine",
        agent=EchoAgent(),
        steps=[step("verdict", "check")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    with pytest.raises(PipelineConditionError, match="use\\s+mode 'until'"):
        pipeline.run_pipeline()


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


def test_map_collects_one_value_per_item():
    process = MapProcess(
        items="topics",
        process_name="per_topic",
        agent=EchoAgent(),
        item_var="topic",
        collect="blurb",
        result_key="blurbs",
        steps=[step("blurb", "write {topic}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert pipeline.run_pipeline(topics=["ants", "bees"])["blurbs"] == [
        "echo:write ants",
        "echo:write bees",
    ]


def test_map_iterations_do_not_leak_into_each_other():
    process = MapProcess(
        items="topics",
        process_name="per_topic",
        agent=EchoAgent(),
        item_var="topic",
        steps=[step("blurb", "write {topic}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(topics=["ants", "bees"])
    # The body's own key never escapes to the outer results.
    assert "blurb" not in results
    assert results["per_topic_results"] == [
        {"blurb": "echo:write ants"},
        {"blurb": "echo:write bees"},
    ]


def test_map_accepts_a_slicing_expression():
    process = MapProcess(
        items="topics[:2]",
        process_name="per_topic",
        agent=EchoAgent(),
        item_var="topic",
        collect="blurb",
        steps=[step("blurb", "write {topic}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(topics=["a", "b", "c"])
    assert len(results["per_topic_results"]) == 2


def test_map_reports_a_missing_collect_key():
    process = MapProcess(
        items="topics",
        process_name="per_topic",
        agent=EchoAgent(),
        collect="nope",
        steps=[step("blurb", "write {item}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    with pytest.raises(PipelineExecutionError, match="collects 'nope'"):
        pipeline.run_pipeline(topics=["a"])


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


def _branch(name: str, step_name: str, template: str) -> SequentialProcess:
    process = SequentialProcess(process_name=name, agent=EchoAgent(name))
    process.add_step(step(step_name, template))
    return process


def test_parallel_merges_branch_results():
    process = ParallelProcess(
        branches=[
            _branch("news", "news_out", "news on {q}"),
            _branch("stats", "stats_out", "stats on {q}"),
        ],
        process_name="research",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(q="otters")
    assert results["news_out"] == "news:news on otters"
    assert results["stats_out"] == "stats:stats on otters"
    assert results["q"] == "otters"


def test_parallel_conflicting_keys_error_by_default():
    process = ParallelProcess(
        branches=[
            _branch("news", "shared", "a {q}"),
            _branch("stats", "shared", "b {q}"),
        ],
        process_name="research",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    with pytest.raises(PipelineExecutionError, match="different values"):
        pipeline.run_pipeline(q="otters")


def test_parallel_can_section_conflicting_keys():
    process = ParallelProcess(
        branches=[
            _branch("news", "shared", "a {q}"),
            _branch("stats", "shared", "b {q}"),
        ],
        process_name="research",
        on_conflict="section",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(q="otters")
    assert results["outputs/news/shared"] == "news:a otters"
    assert results["outputs/stats/shared"] == "stats:b otters"
    assert results.outputs["news"]["shared"] == "news:a otters"
    assert "shared" not in results.outputs


def test_parallel_warns_when_branches_share_an_agent():
    shared = EchoAgent()
    branches = []
    for name, step_name in (("news", "news_out"), ("stats", "stats_out")):
        branch = SequentialProcess(process_name=name, agent=shared)
        branch.add_step(step(step_name, "on {q}"))
        branches.append(branch)

    pipeline = Pipeline()
    pipeline.add_process(ParallelProcess(branches=branches, process_name="research"))

    with pytest.warns(RuntimeWarning, match="same agent instance"):
        pipeline.run_pipeline(q="otters")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_nested_flow_control_round_trips_unchanged():
    inner = MapProcess(
        items="tags",
        process_name="tagmap",
        item_var="tag",
        collect="tagged",
        steps=[step("tagged", "tag {tag}")],
    )
    side = SequentialProcess(process_name="side")
    side.add_step(step("side_out", "side {seed}"))

    gate = ConditionalProcess(
        condition="depth < 2",
        process_name="gate",
        then_processes=[
            ParallelProcess(branches=[inner, side], process_name="fan")
        ],
        else_steps=[step("done", "done")],
    )
    outer = LoopProcess(
        condition="iteration >= 1",
        mode="until",
        max_iterations=2,
        process_name="outer",
        processes=[gate],
    )

    pipeline = Pipeline()
    pipeline.add_process(outer)

    data = pipeline.to_dict()
    restored = Pipeline.from_dict(data, default_agent=EchoAgent())

    assert restored.to_dict() == data

    results = restored.run_pipeline(depth=0, tags=["x", "y"], seed="s")
    assert results["tagmap_results"] == ["echo:tag x", "echo:tag y"]
    assert results["side_out"] == "echo:side s"
    assert results["outer_iterations"] == 2


def test_schema_version_1_documents_still_load():
    legacy = {
        "schema_version": 1,
        "processes": [
            {
                "process_type": "sequential",
                "process_name": "greet",
                "steps": [
                    {
                        "step_name": "hello",
                        "system_message": "Be nice.",
                        "prompt_template": "Greet {name}",
                    }
                ],
            }
        ],
    }
    pipeline = Pipeline.from_dict(legacy, default_agent=EchoAgent())
    assert pipeline.run_pipeline(name="Max")["hello"] == "echo:Greet Max"


def test_unknown_process_type_lists_the_known_ones():
    with pytest.raises(PipelineSerializationError, match="Known types"):
        Pipeline.from_dict(
            {"schema_version": 2, "processes": [{"process_type": "nope"}]}
        )


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(PipelineSerializationError, match="Supported versions"):
        Pipeline.from_dict({"schema_version": 99, "processes": []})


# ---------------------------------------------------------------------------
# Declarative agents and endpoints
# ---------------------------------------------------------------------------


AGENT_DOC = {
    "schema_version": 2,
    "agents": [
        {
            "name": "writer",
            "provider": {
                "type": "openrouter",
                "model": "qwen/qwen3.5-9b",
                "settings": {"temperature": 0.3},
            },
        },
        {
            "name": "judge",
            "provider": {
                "type": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "base_url": "https://anthropic-gateway.internal/v1",
                "settings": {"max_tokens": 2048},
            },
        },
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
                    "prompt_template": "Draft {topic}",
                },
                {
                    "step_name": "verdict",
                    "system_message": "Judge.",
                    "prompt_template": "Judge {draft}",
                    "agent": "judge",
                },
            ],
        }
    ],
}


@pytest.fixture
def api_key_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-router")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")


def test_agents_block_builds_providers_with_custom_endpoints(api_key_env):
    pipeline = Pipeline.from_json(json.dumps(AGENT_DOC))
    process = pipeline.processes[0]
    draft, verdict = process.steps

    assert process.agent.chat_api.base_url == "https://openrouter.ai/api/v1"
    assert process.agent.chat_api.model == "qwen/qwen3.5-9b"
    assert process.agent.chat_api.get_default_settings().get_value("temperature") == 0.3

    assert draft.agent is None  # inherits the process agent at run time
    assert verdict.agent.chat_api.get_provider_identifier() == "anthropic"
    assert verdict.agent.chat_api.base_url == "https://anthropic-gateway.internal/v1"
    assert (
        verdict.agent.chat_api.get_default_settings().get_value("max_tokens") == 2048
    )


def test_agents_block_round_trips(api_key_env):
    pipeline = Pipeline.from_json(json.dumps(AGENT_DOC))
    data = pipeline.to_dict()

    assert data["default_agent"] == "writer"
    assert data["agents"] == AGENT_DOC["agents"]
    assert data["processes"][0]["steps"][1]["agent"] == "judge"


def test_injected_agents_win_over_declared_ones(api_key_env):
    injected = EchoAgent("injected")
    pipeline = Pipeline.from_dict(AGENT_DOC, default_agent=injected)

    # default_agent replaces the JSON default, but the step's explicit
    # 'judge' reference is more specific and survives.
    assert pipeline.processes[0].agent is injected
    assert pipeline.processes[0].steps[1].agent is not injected


def test_build_agents_false_ignores_the_agents_block():
    injected = EchoAgent("injected")
    pipeline = Pipeline.from_dict(
        AGENT_DOC, default_agent=injected, build_agents=False
    )

    assert pipeline.processes[0].agent is injected
    # The 'judge' reference is ignored rather than raising, so the step has no
    # agent of its own and inherits the injected one at run time.
    assert pipeline.processes[0].steps[1].agent is None
    assert pipeline.run_pipeline(topic="otters")["verdict"].startswith("injected:")


def test_literal_api_key_is_refused():
    doc = json.loads(json.dumps(AGENT_DOC))
    doc["agents"][0]["provider"]["api_key"] = "sk-should-not-be-here"

    with pytest.raises(AgentConfigurationError, match="must not contain a literal"):
        Pipeline.from_dict(doc)


def test_missing_environment_variable_names_itself(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    doc = json.loads(json.dumps(AGENT_DOC))
    doc["agents"] = [doc["agents"][0]]
    doc["processes"][0]["steps"][1].pop("agent")

    with pytest.raises(AgentConfigurationError, match="OPENROUTER_API_KEY"):
        Pipeline.from_dict(doc)


def test_unreferenced_agents_are_not_built(monkeypatch):
    """A declared but unused endpoint must not demand a key it never needs."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-router")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    doc = json.loads(json.dumps(AGENT_DOC))
    doc["processes"][0]["steps"][1].pop("agent")  # nothing references 'judge'

    pipeline = Pipeline.from_dict(doc)
    assert pipeline.processes[0].agent is not None


def test_unknown_setting_name_is_reported_not_ignored(api_key_env):
    doc = json.loads(json.dumps(AGENT_DOC))
    doc["agents"][0]["provider"]["settings"] = {"temperatur": 0.3}

    with pytest.raises(AgentConfigurationError, match="has no setting 'temperatur'"):
        Pipeline.from_dict(doc)


def test_unknown_agent_reference_names_the_declared_agents(api_key_env):
    doc = json.loads(json.dumps(AGENT_DOC))
    doc["processes"][0]["steps"][1]["agent"] = "ghost"

    with pytest.raises(PipelineSerializationError, match="unknown agent 'ghost'"):
        Pipeline.from_dict(doc)


def test_extra_settings_reach_providers_that_do_not_declare_them(api_key_env):
    doc = json.loads(json.dumps(AGENT_DOC))
    doc["agents"][0]["provider"]["extra_settings"] = {"seed": 42}

    pipeline = Pipeline.from_dict(doc)
    settings = pipeline.processes[0].agent.chat_api.get_default_settings()
    assert settings.get_value("seed") == 42


# ---------------------------------------------------------------------------
# The shipped example
# ---------------------------------------------------------------------------


EXAMPLE_JSON = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    + "/examples/agents/pipeline/flow_control_pipeline.json"
)


class ReviewStub:
    """Approves on the second review, exercising both loop and branch."""

    def __init__(self) -> None:
        self.reviews = 0

    def get_response(self, messages, tool_registry=None, **kwargs):
        text = messages[-1].get_as_text()
        if text.startswith("Revision"):
            self.reviews += 1
            verdict = "APPROVED" if self.reviews >= 2 else "Too vague."
            return SimpleNamespace(response=verdict)
        return SimpleNamespace(response=f"<{text.splitlines()[0][:40]}>")


@pytest.mark.skipif(
    not os.path.exists(EXAMPLE_JSON), reason="example pipeline not present"
)
def test_shipped_example_pipeline_loads_and_runs():
    with open(EXAMPLE_JSON, encoding="utf-8") as handle:
        document = json.load(handle)

    pipeline = Pipeline.from_dict(
        document, build_agents=False, default_agent=ReviewStub()
    )
    assert [process.process_type for process in pipeline.processes] == [
        "map",
        "parallel",
        "sequential",
        "loop",
        "conditional",
    ]

    with pytest.warns(RuntimeWarning, match="same agent instance"):
        results = pipeline.run_pipeline(
            topics=["otter tool use", "otter social structure"],
            audience="curious non-specialists",
        )

    assert len(results["notes"]) == 2
    assert results["refine_iterations"] == 2
    assert results["passed_review"] is True
    assert results["summary"]


@pytest.mark.skipif(
    not os.path.exists(EXAMPLE_JSON), reason="example pipeline not present"
)
def test_shipped_example_serialization_is_idempotent():
    with open(EXAMPLE_JSON, encoding="utf-8") as handle:
        document = json.load(handle)

    once = Pipeline.from_dict(document, build_agents=False).to_dict()
    twice = Pipeline.from_dict(once, build_agents=False).to_dict()

    # The first pass normalizes bare-string conditions to their object form;
    # after that the document is a fixed point.
    assert once == twice
    assert once["agents"] == document["agents"]
    assert once["default_agent"] == document["default_agent"]

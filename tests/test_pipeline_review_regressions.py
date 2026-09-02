"""Regressions for bugs found reviewing the flow-control work.

Each test names the defect it pins down, so a future refactor that reopens one
fails loudly instead of quietly.
"""

from __future__ import annotations

import json
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
from ToolAgents.utilities.message_template import MessageTemplate


class WriterAgent:
    """Writes a fixed value, so branch outputs are controllable."""

    def __init__(self, value: str = "v") -> None:
        self.value = value

    def get_response(self, messages, tool_registry=None, **kwargs):
        return SimpleNamespace(response=self.value)


def step(name: str, template: str = "x") -> ProcessStep:
    return ProcessStep(
        step_name=name, system_message="s", prompt_template=template
    )


def branch(name: str, step_name: str, value: str) -> SequentialProcess:
    process = SequentialProcess(process_name=name, agent=WriterAgent(value))
    process.add_step(step(step_name))
    return process


# ---------------------------------------------------------------------------
# Condition sandbox
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "len('a' * 9999 * 9999)",
        "'a' * (10000 * 10000)",
        "[0] * 9999 * 9999",
    ],
)
def test_chained_multiplication_cannot_bypass_the_repeat_cap(expression):
    """A per-operand cap is chained around; only one multiply is allowed."""

    with pytest.raises(PipelineConditionError, match="multiplication"):
        SafeExpression(expression)


def test_single_multiplication_is_still_allowed():
    assert SafeExpression("score * 100 > 80").evaluate({"score": 0.9}) is True


@pytest.mark.parametrize(
    "expression, results, expected",
    [
        ("defined('score') and score > 3", {}, False),
        ("defined('score') and score > 3", {"score": 5}, True),
        ("not defined('score') or score > 3", {}, True),
        ("score if defined('score') else False", {}, False),
        ("has_score and score > 3", {"has_score": False}, False),
    ],
)
def test_conditions_can_tolerate_a_result_that_does_not_exist(
    expression, results, expected
):
    """Flow control creates optional keys; a condition must survive them.

    A key written only inside a branch that did not run genuinely does not
    exist. Eager name-checking made every such condition a hard failure with
    no expressible workaround.
    """

    assert SafeExpression(expression).evaluate(results) is expected


def test_a_genuinely_missing_name_still_explains_itself():
    with pytest.raises(PipelineConditionError, match="unknown result: score"):
        SafeExpression("score > 1").evaluate({"other": 1})


@pytest.mark.parametrize("shadow", ["len", "sum", "str", "default"])
def test_a_result_shadowing_a_helper_is_reported_not_silently_broken(shadow):
    """A step named `sum` used to poison every condition with a type error."""

    expression = f"{shadow}(items) > 1" if shadow != "sum" else "sum > 1"
    with pytest.raises(PipelineConditionError, match="collide"):
        SafeExpression(expression).evaluate({"items": [1, 2], shadow: "oops"})


# ---------------------------------------------------------------------------
# Parallel merge
# ---------------------------------------------------------------------------


def test_section_mode_keeps_a_pre_existing_outer_value():
    """The first branch's rename used to delete the incoming value too."""

    process = ParallelProcess(
        branches=[branch("a", "k", "A"), branch("b", "k", "B")],
        process_name="fan",
        on_conflict="section",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(k="ORIGINAL")
    assert results["outputs/a/k"] == "A"
    assert results["outputs/b/k"] == "B"
    assert results["inputs/k"] == "ORIGINAL"


def test_sectioned_keys_are_usable_by_templates_and_conditions():
    """A mangled or dotted key is unreadable downstream; a path is not."""

    process = ParallelProcess(
        branches=[branch("a", "k", "A"), branch("b", "k", "B")],
        process_name="fan",
        on_conflict="section",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)
    results = pipeline.run_pipeline()

    rendered = MessageTemplate.from_string(
        "<{outputs/a/k}>"
    ).generate_message_content(results)
    assert rendered == "<A>"
    assert SafeExpression("outputs['a']['k'] == 'A'").evaluate(results) is True


def test_duplicate_branch_names_do_not_destroy_a_branch_result():
    """Two branches sharing a name used to overwrite each other silently."""

    process = ParallelProcess(
        branches=[branch("w", "k", "A"), branch("w", "k", "B")],
        process_name="fan",
        on_conflict="section",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline()
    values = sorted(
        section["k"]
        for section in results.outputs.values()
        if isinstance(section, dict)
    )
    assert values == ["A", "B"]


def test_branches_agreeing_on_a_value_is_not_a_conflict():
    """Equal-but-not-identical values used to raise a hard conflict."""

    process = ParallelProcess(
        branches=[branch("a", "k", "same"), branch("b", "k", "same")],
        process_name="fan",
        on_conflict="error",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert pipeline.run_pipeline()["k"] == "same"


def test_merge_conflict_raises_an_execution_error_not_a_serialization_one():
    process = ParallelProcess(
        branches=[branch("a", "k", "A"), branch("b", "k", "B")],
        process_name="fan",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    with pytest.raises(PipelineExecutionError):
        pipeline.run_pipeline()


def test_parallel_merge_is_deterministic_across_runs():
    process = ParallelProcess(
        branches=[branch("a", "k", "A"), branch("b", "k", "B")],
        process_name="fan",
        on_conflict="last_wins",
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert {pipeline.run_pipeline()["k"] for _ in range(12)} == {"B"}


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class FlagAgent:
    """Sets the flag once the configured iteration is reached."""

    def __init__(self, done_on: int) -> None:
        self.done_on = done_on
        self.calls = 0

    def get_response(self, messages, tool_registry=None, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            response="yes" if self.calls >= self.done_on else "no"
        )


def test_while_loop_finishing_on_its_last_iteration_is_not_an_error():
    """The exit condition was never re-tested after the final body run."""

    process = LoopProcess(
        condition="done != 'yes'",
        mode="while",
        max_iterations=3,
        process_name="w",
        agent=FlagAgent(done_on=3),
        on_max_iterations="error",
        steps=[step("done")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    # The body sets done='yes' on the third and final permitted iteration.
    # Without a post-loop re-test this reported failure despite succeeding.
    results = pipeline.run_pipeline(done="no")
    assert results["w_iterations"] == 3
    assert results["done"] == "yes"


def test_loop_does_not_leave_its_iteration_counter_behind():
    """Scratch state leaking out let a nested loop clobber an outer one."""

    process = LoopProcess(
        max_iterations=2,
        process_name="w",
        agent=WriterAgent(),
        steps=[step("out", "pass {iteration}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline()
    assert "iteration" not in results
    assert results["w_iterations"] == 2


def test_nested_loops_do_not_clobber_each_others_iteration_counter():
    inner = LoopProcess(
        max_iterations=2, process_name="inner", steps=[step("i_out")]
    )
    outer = LoopProcess(
        max_iterations=2,
        process_name="outer",
        agent=WriterAgent(),
        processes=[inner],
    )
    pipeline = Pipeline()
    pipeline.add_process(outer)

    results = pipeline.run_pipeline()
    assert results["outer_iterations"] == 2
    assert results["inner_iterations"] == 2
    assert "iteration" not in results


def test_a_preexisting_iteration_value_is_restored():
    process = LoopProcess(
        max_iterations=2, process_name="w", agent=WriterAgent(), steps=[step("o")]
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert pipeline.run_pipeline(iteration="untouched")["iteration"] == "untouched"


def test_max_iterations_rejects_a_boolean():
    """`True` is an int in Python and would silently mean one iteration."""

    with pytest.raises(ValueError, match="positive integer"):
        LoopProcess(max_iterations=True, steps=[step("o")])


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


def test_map_collects_body_keys_that_shadow_an_outer_key():
    """`key not in results` treated an overwritten key as never produced."""

    process = MapProcess(
        items="topics",
        process_name="m",
        agent=WriterAgent("NEW"),
        steps=[step("draft")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(topics=["a"], draft="OLD")
    assert results["m_results"] == [{"draft": "NEW"}]
    assert results["draft"] == "OLD"


def test_map_rejects_a_mapping_as_items():
    process = MapProcess(
        items="topics", process_name="m", agent=WriterAgent(), steps=[step("o")]
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    with pytest.raises(PipelineExecutionError, match="to yield a list"):
        pipeline.run_pipeline(topics={"a": 1})


def test_map_can_run_twice_over_the_same_items():
    """A generator would be exhausted on a second pass; items are materialized."""

    process = MapProcess(
        items="topics",
        process_name="m",
        agent=WriterAgent("v"),
        collect="o",
        steps=[step("o")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    topics = ["a", "b"]
    assert pipeline.run_pipeline(topics=topics)["m_results"] == ["v", "v"]
    assert pipeline.run_pipeline(topics=topics)["m_results"] == ["v", "v"]


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------


NESTED_DOC = {
    "schema_version": 2,
    "processes": [
        {
            "process_type": "loop",
            "process_name": "refine",
            "max_iterations": 1,
            "processes": [
                {
                    "process_type": "sequential",
                    "process_name": "body",
                    "steps": [
                        {
                            "step_name": "out",
                            "system_message": "s",
                            "prompt_template": "x",
                        }
                    ],
                }
            ],
        }
    ],
}


def test_an_agent_injected_for_a_flow_process_reaches_its_children():
    """process_agents={"refine": X} used to apply only to the loop object."""

    outer = WriterAgent("outer")
    pipeline = Pipeline.from_dict(
        NESTED_DOC,
        default_agent=WriterAgent("default"),
        process_agents={"refine": outer},
    )

    loop = pipeline.processes[0]
    assert loop.agent is outer
    assert loop.processes[0].agent is outer
    assert pipeline.run_pipeline()["out"] == "outer"


def test_default_agent_still_reaches_a_nested_process():
    pipeline = Pipeline.from_dict(
        NESTED_DOC, default_agent=WriterAgent("default")
    )
    assert pipeline.run_pipeline()["out"] == "default"


# ---------------------------------------------------------------------------
# JSON shape
# ---------------------------------------------------------------------------


def test_a_loop_declared_with_steps_in_json_actually_runs_them():
    """`steps` on a flow process used to load fine and do nothing at all."""

    document = {
        "schema_version": 2,
        "processes": [
            {
                "process_type": "loop",
                "process_name": "w",
                "max_iterations": 2,
                "steps": [
                    {
                        "step_name": "out",
                        "system_message": "s",
                        "prompt_template": "x",
                    }
                ],
            }
        ],
    }
    pipeline = Pipeline.from_dict(document, default_agent=WriterAgent("ran"))
    results = pipeline.run_pipeline()

    assert results["out"] == "ran"
    assert results["w_iterations"] == 2


def test_steps_on_a_multi_branch_process_is_refused():
    document = {
        "schema_version": 2,
        "processes": [
            {
                "process_type": "parallel",
                "process_name": "fan",
                "steps": [
                    {
                        "step_name": "out",
                        "system_message": "s",
                        "prompt_template": "x",
                    }
                ],
            }
        ],
    }
    with pytest.raises(PipelineSerializationError, match="ambiguous"):
        Pipeline.from_dict(document, default_agent=WriterAgent())


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


def _local_doc(provider_type: str) -> dict:
    return {
        "schema_version": 2,
        "agents": [
            {
                "name": "local",
                "provider": {"type": provider_type, "model": "llama3"},
            }
        ],
        "default_agent": "local",
        "processes": [],
    }


@pytest.mark.parametrize("provider_type", ["ollama", "vllm"])
def test_local_endpoints_need_no_api_key(monkeypatch, provider_type):
    """The aliases whose whole purpose is local serving must work keyless."""

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    pipeline = Pipeline.from_dict(_local_doc(provider_type))
    built = pipeline.agent_configs[0].build()
    assert built.chat_api.base_url.startswith("http://localhost")


def test_extra_settings_cannot_silently_redefine_a_declared_setting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    document = {
        "schema_version": 2,
        "agents": [
            {
                "name": "a",
                "provider": {
                    "type": "openai",
                    "model": "gpt-4o",
                    "extra_settings": {"temperature": 0.2},
                },
            }
        ],
        "default_agent": "a",
        "processes": [
            {
                "process_type": "sequential",
                "process_name": "p",
                "steps": [
                    {
                        "step_name": "o",
                        "system_message": "s",
                        "prompt_template": "x",
                    }
                ],
            }
        ],
    }
    with pytest.raises(AgentConfigurationError, match="already a declared setting"):
        Pipeline.from_dict(document)


def test_an_empty_base_url_falls_back_to_the_provider_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from ToolAgents.pipelines import ProviderConfig

    config = ProviderConfig.from_dict(
        {"type": "anthropic", "model": "claude-sonnet-4-20250514", "base_url": ""}
    )
    assert config.base_url is None
    assert config.build().base_url is None

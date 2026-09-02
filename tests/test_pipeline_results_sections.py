"""Tests for sectioned pipeline results."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ToolAgents.pipelines import (
    LoopProcess,
    MapProcess,
    Pipeline,
    PipelineResults,
    ProcessStep,
    SafeExpression,
    SequentialProcess,
)
from ToolAgents.utilities.message_template import MessageTemplate


class EchoAgent:
    def get_response(self, messages, tool_registry=None, **kwargs):
        return SimpleNamespace(response=messages[-1].get_as_text())


def step(name: str, template: str) -> ProcessStep:
    return ProcessStep(step_name=name, system_message="s", prompt_template=template)


# -- the container ----------------------------------------------------------


def test_sections_are_separate_but_a_bare_name_still_resolves():
    results = PipelineResults(inputs={"topic": "otters"}, outputs={"draft": "D"})

    assert results["inputs/topic"] == "otters"
    assert results["outputs/draft"] == "D"
    assert results["topic"] == "otters"
    assert results["draft"] == "D"


def test_bare_lookup_is_innermost_first():
    results = PipelineResults(
        inputs={"x": "input"}, outputs={"x": "output"}, vars={"x": "var"}
    )
    assert results["x"] == "var"

    del results.vars["x"]
    assert results["x"] == "output"

    del results.outputs["x"]
    assert results["x"] == "input"


def test_a_bare_write_lands_in_outputs():
    results = PipelineResults(inputs={"topic": "otters"})
    results["draft"] = "D"

    assert results.outputs == {"draft": "D"}
    assert results.inputs == {"topic": "otters"}


def test_a_step_can_no_longer_overwrite_an_input():
    """The flat namespace let a step named `topic` destroy the caller's value."""

    process = SequentialProcess("p", agent=EchoAgent())
    process.add_step(step("topic", "derived from {topic}"))
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(topic="otters")
    assert results["inputs/topic"] == "otters"
    assert results["outputs/topic"] == "derived from otters"


def test_sections_can_be_added_for_other_kinds_of_state():
    results = PipelineResults()
    results.section("agent", create=True)["model"] = "sonnet"

    assert results["agent/model"] == "sonnet"
    assert "agent" in results.section_names

    with pytest.raises(KeyError, match="Unknown results section"):
        results.section("nope")


def test_flat_view_matches_the_old_dictionary_behaviour():
    results = PipelineResults(inputs={"a": 1, "b": 2}, outputs={"b": 3})
    assert results.flat() == {"a": 1, "b": 3}
    assert dict(results) == {"a": 1, "b": 3}


def test_copy_gives_independent_sections():
    results = PipelineResults(outputs={"draft": "D"})
    clone = results.copy()
    clone.outputs["draft"] = "E"

    assert results.outputs["draft"] == "D"


def test_nested_paths_round_trip():
    results = PipelineResults()
    results.set_path("outputs/news/draft", "A")

    assert results["outputs/news/draft"] == "A"
    assert results.outputs["news"] == {"draft": "A"}


# -- addressing from templates and conditions -------------------------------


def test_templates_can_address_sections():
    results = PipelineResults(
        inputs={"audience": "kids"}, outputs={"draft": "D", "news": {"draft": "A"}}
    )
    rendered = MessageTemplate.from_string(
        "{outputs/draft}|{inputs/audience}|{outputs/news/draft}|{draft}"
    ).generate_message_content(results)

    assert rendered == "D|kids|A|D"


def test_conditions_can_address_sections():
    results = PipelineResults(inputs={"topic": "otters"}, outputs={"draft": "D"})

    assert SafeExpression("outputs['draft'] == 'D'").evaluate(results) is True
    assert SafeExpression("inputs['topic'] == 'otters'").evaluate(results) is True
    assert SafeExpression("draft == 'D'").evaluate(results) is True


# -- flow-control scratch lives in vars -------------------------------------


def test_loop_counter_lives_in_vars_and_does_not_leak():
    process = LoopProcess(
        max_iterations=2,
        process_name="w",
        agent=EchoAgent(),
        steps=[step("out", "pass {iteration}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline()
    assert results["outputs/out"] == "pass 1"
    assert results.vars == {}
    assert results["outputs/w_iterations"] == 2


def test_a_step_named_iteration_cannot_break_the_loop_counter():
    """In the flat namespace this collision silently corrupted the counter."""

    process = LoopProcess(
        max_iterations=3,
        process_name="w",
        agent=EchoAgent(),
        steps=[step("iteration", "at {iteration}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline()
    assert results["outputs/iteration"] == "at 2"
    assert results["outputs/w_iterations"] == 3


def test_map_item_lives_in_vars_and_collection_is_exact():
    process = MapProcess(
        items="topics",
        process_name="m",
        agent=EchoAgent(),
        item_var="topic",
        steps=[step("draft", "on {topic}")],
    )
    pipeline = Pipeline()
    pipeline.add_process(process)

    results = pipeline.run_pipeline(topics=["a", "b"], draft="OLD")
    assert results["outputs/m_results"] == [
        {"draft": "on a"},
        {"draft": "on b"},
    ]
    assert results["inputs/draft"] == "OLD"
    assert results.vars == {}

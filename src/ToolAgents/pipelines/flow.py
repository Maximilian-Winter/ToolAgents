"""Flow-control processes: conditional, loop, map, and parallel.

Each of these is an ordinary :class:`~ToolAgents.pipelines.pipeline.Process`
whose body is *other processes*, so they nest freely: a loop containing a
branch containing a parallel fan-out is just three objects.

All four serialize to and from JSON alongside the rest of the pipeline, and
every condition they use is a sandboxed expression (see
:mod:`ToolAgents.pipelines.conditions`), never arbitrary Python.

Importing this module registers the four process types, which is why
:func:`~ToolAgents.pipelines.pipeline.get_process_type` imports it lazily.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping, Sequence

from ToolAgents.agents.base_llm_agent import BaseToolAgent
from ToolAgents.pipelines.conditions import (
    Condition,
    SafeExpression,
    PipelineConditionError,
    condition_from_config,
)
from ToolAgents.pipelines.pipeline import (
    PipelineExecutionError,
    PipelineLoadContext,
    PipelineSerializationError,
    PipelineToolRegistry,
    Process,
    ProcessStep,
    SequentialProcess,
    processes_from_config,
    processes_to_config,
    register_process_type,
)
from ToolAgents.pipelines.results import PipelineResults

__all__ = [
    "ConditionalProcess",
    "FlowProcess",
    "LoopProcess",
    "MapProcess",
    "ParallelProcess",
]

#: Hard ceiling on loop iterations, so a malformed condition cannot spin
#: forever burning API credit.
DEFAULT_MAX_ITERATIONS = 10


def _steps_from_config(
    step_configs: Any,
    context: PipelineLoadContext,
    process_name: str,
    agent: BaseToolAgent | None,
) -> SequentialProcess:
    """Wrap a JSON ``steps`` list into a sequential body process."""

    sequence = SequentialProcess(
        process_name=f"{process_name}_body", agent=agent
    )
    for step_data in step_configs:
        step_name = str(step_data.get("step_name", step_data.get("name")))
        step_agent_name = step_data.get("agent")
        sequence.add_step(
            ProcessStep.from_dict(
                step_data,
                tool_registry=context.tool_registry,
                agent=context.agent_for_step(
                    process_name,
                    step_name,
                    str(step_agent_name) if step_agent_name else None,
                ),
            )
        )
    return sequence


def _body_from_config(
    data: Mapping[str, Any],
    context: PipelineLoadContext,
    process_name: str,
    agent: BaseToolAgent | None,
) -> list[Process]:
    """Read a single-body flow process's children from JSON.

    ``steps`` is accepted alongside ``processes`` to mirror the Python
    constructors. Without it a document using ``steps`` would load cleanly and
    then do nothing at all.
    """

    body = processes_from_config(
        data.get("processes", data.get("body")),
        context,
        field_name="processes",
    )
    step_configs = data.get("steps")
    if step_configs:
        body.append(
            _steps_from_config(step_configs, context, process_name, agent)
        )
    return body


def _reject_steps(data: Mapping[str, Any], process_name: str, field_hint: str) -> None:
    """Fail loudly on a ``steps`` key a multi-branch process cannot place."""

    if data.get("steps"):
        raise PipelineSerializationError(
            f"Process '{process_name}' has more than one branch, so a bare "
            f"'steps' list is ambiguous. Put the steps in a sequential process "
            f"under {field_hint}."
        )


def _build_body(
    processes: Sequence[Process] | None,
    steps: Sequence[ProcessStep] | None,
    name: str,
    agent: BaseToolAgent | None,
) -> list[Process]:
    """Build a body list from processes and/or a convenience list of steps."""

    body = list(processes or [])
    if steps:
        sequence = SequentialProcess(process_name=name, agent=agent)
        sequence.add_steps(list(steps))
        body.append(sequence)
    return body


class FlowProcess(Process):
    """Base class for processes whose body is a list of other processes.

    ``add_step`` is kept working as a convenience: a step added to a flow
    process is appended to a trailing :class:`SequentialProcess` in its primary
    body, created on demand. Without this, ``add_step`` on a loop would append
    to the unused ``steps`` list inherited from ``Process`` and silently never
    run.
    """

    def primary_body(self) -> list[Process]:
        """Return the process list that ``add_step``/``add_process`` extend."""

        raise NotImplementedError

    def child_bodies(self) -> tuple[list[Process], ...]:
        """Return every child process list, for traversal."""

        return (self.primary_body(),)

    def add_process(self, process: Process) -> "FlowProcess":
        """Append a child process to this process's primary body."""

        self.primary_body().append(process)
        return self

    def add_processes(self, processes: Sequence[Process]) -> "FlowProcess":
        """Append several child processes to this process's primary body."""

        self.primary_body().extend(processes)
        return self

    def add_step(self, step: ProcessStep) -> "FlowProcess":
        """Append a step to the trailing sequence of this process's body."""

        body = self.primary_body()
        if body and isinstance(body[-1], SequentialProcess):
            body[-1].add_step(step)
            return self

        sequence = SequentialProcess(
            process_name=f"{self.process_name}_body",
            agent=self.agent,
        )
        sequence.add_step(step)
        body.append(sequence)
        return self

    def add_steps(self, steps: Sequence[ProcessStep]) -> "FlowProcess":
        """Append several steps to the trailing sequence of this body."""

        for step in steps:
            self.add_step(step)
        return self

    # -- shared serialization helpers -------------------------------------

    def _base_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "process_type": self.process_type,
            "process_name": self.process_name,
        }
        if self.agent_name:
            data["agent"] = self.agent_name
        return data

    @staticmethod
    def _read_identity(
        data: Mapping[str, Any],
        default_name: str,
    ) -> tuple[str, str | None]:
        """Return ``(process_name, agent_name)`` from a process config."""

        process_name = str(data.get("process_name", data.get("name", default_name)))
        agent_name = data.get("agent")
        return process_name, (str(agent_name) if agent_name else None)


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------


@register_process_type
class ConditionalProcess(FlowProcess):
    """Run one branch or the other depending on a condition.

    Example::

        ConditionalProcess(
            condition="score < 0.7",
            then_steps=[revise_step],
            else_steps=[publish_step],
        )

    JSON::

        {
          "process_type": "conditional",
          "process_name": "gate",
          "condition": "score < 0.7",
          "then": [ ...processes... ],
          "else": [ ...processes... ],
          "record_as": "needed_revision"
        }
    """

    process_type = "conditional"

    def __init__(
        self,
        condition: Condition | Mapping[str, Any] | str,
        process_name: str = "ConditionalProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
        then_processes: Sequence[Process] | None = None,
        else_processes: Sequence[Process] | None = None,
        then_steps: Sequence[ProcessStep] | None = None,
        else_steps: Sequence[ProcessStep] | None = None,
        record_as: str | None = None,
    ):
        """Initialize a conditional process.

        Args:
            condition: A condition object, config mapping, or bare expression
                string evaluated against the results mapping.
            process_name: Name identifier for the process.
            agent: Default agent lent to children that have none.
            agent_name: Name of a declared agent, for JSON round-tripping.
            then_processes: Processes run when the condition holds.
            else_processes: Processes run when it does not.
            then_steps: Convenience steps wrapped into a sequential process.
            else_steps: Convenience steps for the else branch.
            record_as: Optional results key recording which branch was taken.
        """
        super().__init__(process_name, agent, agent_name)
        self.condition = condition_from_config(condition)
        self.then_processes = _build_body(
            then_processes, then_steps, f"{process_name}_then", agent
        )
        self.else_processes = _build_body(
            else_processes, else_steps, f"{process_name}_else", agent
        )
        self.record_as = record_as

    def primary_body(self) -> list[Process]:
        return self.then_processes

    def child_bodies(self) -> tuple[list[Process], ...]:
        return (self.then_processes, self.else_processes)

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """Evaluate the condition and run the matching branch."""

        taken = self.condition.evaluate(results)
        if self.record_as:
            results.outputs[self.record_as] = taken
        branch = self.then_processes if taken else self.else_processes
        return self.run_child_processes(branch, results)

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this conditional process."""

        data = self._base_dict()
        data["condition"] = self.condition.to_dict()
        data["then"] = processes_to_config(self.then_processes, tool_registry)
        if self.else_processes:
            data["else"] = processes_to_config(self.else_processes, tool_registry)
        if self.record_as:
            data["record_as"] = self.record_as
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "ConditionalProcess":
        """Restore a conditional process from JSON."""

        process_name, agent_name = cls._read_identity(data, "ConditionalProcess")
        if "condition" not in data:
            raise PipelineSerializationError(
                f"Conditional process '{process_name}' is missing 'condition'."
            )
        agent = context.agent_for_process(process_name, agent_name)
        nested = context.nested(process_name, agent)
        _reject_steps(data, process_name, "'then' or 'else'")
        return cls(
            condition=condition_from_config(data["condition"]),
            process_name=process_name,
            agent=agent,
            agent_name=agent_name,
            then_processes=processes_from_config(
                data.get("then"), nested, field_name="then"
            ),
            else_processes=processes_from_config(
                data.get("else"), nested, field_name="else"
            ),
            record_as=_optional_str(data.get("record_as")),
        )


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


@register_process_type
class LoopProcess(FlowProcess):
    """Repeat a body until a condition is satisfied or a cap is reached.

    The two modes differ in **when** the condition is tested and in **what it
    means**. They are inverses, as in most languages, so the same expression
    cannot simply be moved from one to the other:

    ``"until"`` (the default)
        The condition is a *stop* condition. Run the body, then test; stop when
        it becomes **true**. The body always runs at least once, which is what
        a refine-until-good-enough loop needs, because the condition usually
        reads a value the body produces.

    ``"while"``
        The condition is a *continue* condition. Test before each iteration;
        stop when it becomes **false**. The body may run zero times, and the
        condition must only reference values that already exist.

    So ``mode="until", condition="approved"`` and
    ``mode="while", condition="not approved"`` express the same loop.

    ``max_iterations`` is always enforced, so a condition that never becomes
    true cannot spin forever burning API credit.

    JSON::

        {
          "process_type": "loop",
          "process_name": "refine",
          "mode": "until",
          "condition": "contains(lower(review), 'approved')",
          "max_iterations": 4,
          "on_max_iterations": "stop",
          "processes": [ ...body... ]
        }
    """

    process_type = "loop"

    MODES = ("until", "while")
    ON_MAX = ("stop", "error")

    def __init__(
        self,
        condition: Condition | Mapping[str, Any] | str | None = None,
        mode: str = "until",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        process_name: str = "LoopProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
        processes: Sequence[Process] | None = None,
        steps: Sequence[ProcessStep] | None = None,
        iteration_var: str = "iteration",
        iterations_key: str | None = None,
        on_max_iterations: str = "stop",
    ):
        """Initialize a loop process.

        Args:
            condition: Stop condition. When omitted the loop runs exactly
                ``max_iterations`` times.
            mode: ``"until"`` (post-test, body runs at least once) or
                ``"while"`` (pre-test, body may not run at all).
            max_iterations: Hard cap on iterations. Must be positive.
            process_name: Name identifier for the process.
            agent: Default agent lent to children that have none.
            agent_name: Name of a declared agent, for JSON round-tripping.
            processes: Body processes.
            steps: Convenience steps wrapped into a sequential body process.
            iteration_var: Results key holding the zero-based iteration index,
                so prompt templates can reference ``{iteration}``.
            iterations_key: Results key receiving the total iteration count.
                Defaults to ``"<process_name>_iterations"``.
            on_max_iterations: ``"stop"`` to exit quietly at the cap, or
                ``"error"`` to raise when the condition was never satisfied.
        """
        super().__init__(process_name, agent, agent_name)

        if mode not in self.MODES:
            raise ValueError(
                f"Loop mode must be one of {self.MODES}, got {mode!r}."
            )
        if on_max_iterations not in self.ON_MAX:
            raise ValueError(
                f"on_max_iterations must be one of {self.ON_MAX}, "
                f"got {on_max_iterations!r}."
            )
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError(
                f"max_iterations must be a positive integer, got {max_iterations!r}."
            )

        self.condition = (
            None if condition is None else condition_from_config(condition)
        )
        self.mode = mode
        self.max_iterations = max_iterations
        self.processes = _build_body(
            processes, steps, f"{process_name}_body", agent
        )
        self.iteration_var = iteration_var
        self.iterations_key = iterations_key or f"{process_name}_iterations"
        self.on_max_iterations = on_max_iterations

    def primary_body(self) -> list[Process]:
        return self.processes

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """Run the body repeatedly according to mode and condition."""

        # The counter lives in `vars`, the scratch section, so it can never
        # collide with a step named `iteration` or with a caller argument.
        had_iteration_var = self.iteration_var in results.vars
        previous_iteration = results.vars.get(self.iteration_var)

        iterations = 0
        satisfied = False

        for index in range(self.max_iterations):
            if self.mode == "while" and self.condition is not None:
                if not self._evaluate(results, first=index == 0):
                    satisfied = True
                    break

            results.vars[self.iteration_var] = index
            results = self.run_child_processes(self.processes, results)
            iterations = index + 1

            if self.mode == "until" and self.condition is not None:
                if self._evaluate(results, first=False):
                    satisfied = True
                    break
        else:
            # The cap was reached without breaking. In "while" mode the
            # condition is only tested at the top of an iteration, so the last
            # body run was never followed by a test: without this, a loop that
            # finished the job on its final permitted iteration would still be
            # reported as having failed.
            if self.mode == "while" and self.condition is not None:
                satisfied = not self._evaluate(results, first=False)

        # The iteration counter is scratch state for the body's prompts, not a
        # result. Leaving it behind would let a nested loop clobber the value
        # an enclosing loop is still using.
        if had_iteration_var:
            results.vars[self.iteration_var] = previous_iteration
        else:
            results.vars.pop(self.iteration_var, None)

        results.outputs[self.iterations_key] = iterations

        if (
            self.condition is not None
            and not satisfied
            and self.on_max_iterations == "error"
        ):
            goal = (
                "while its condition was still true"
                if self.mode == "while"
                else "without satisfying"
            )
            raise PipelineExecutionError(
                f"Loop '{self.process_name}' reached its cap of "
                f"{self.max_iterations} iterations {goal} "
                f"{self.condition.describe()!r}."
            )

        return results

    def _evaluate(self, results: Mapping[str, Any], first: bool) -> bool:
        """Evaluate the stop condition, explaining the classic 'while' trap."""

        try:
            return self.condition.evaluate(results)
        except PipelineConditionError as exc:
            if first and self.mode == "while":
                raise PipelineConditionError(
                    f"{exc} — loop '{self.process_name}' tests its condition "
                    "before the first iteration because mode is 'while'. If "
                    "the condition reads a value the body produces, use "
                    "mode 'until' so the body runs first."
                ) from exc
            raise

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this loop process."""

        data = self._base_dict()
        data["mode"] = self.mode
        data["max_iterations"] = self.max_iterations
        if self.condition is not None:
            data["condition"] = self.condition.to_dict()
        data["processes"] = processes_to_config(self.processes, tool_registry)
        if self.iteration_var != "iteration":
            data["iteration_var"] = self.iteration_var
        if self.iterations_key != f"{self.process_name}_iterations":
            data["iterations_key"] = self.iterations_key
        if self.on_max_iterations != "stop":
            data["on_max_iterations"] = self.on_max_iterations
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "LoopProcess":
        """Restore a loop process from JSON."""

        process_name, agent_name = cls._read_identity(data, "LoopProcess")
        agent = context.agent_for_process(process_name, agent_name)
        nested = context.nested(process_name, agent)
        condition = data.get("condition")

        return cls(
            condition=None if condition is None else condition_from_config(condition),
            mode=str(data.get("mode", "until")),
            max_iterations=int(data.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
            process_name=process_name,
            agent=agent,
            agent_name=agent_name,
            processes=_body_from_config(data, nested, process_name, agent),
            iteration_var=str(data.get("iteration_var", "iteration")),
            iterations_key=_optional_str(data.get("iterations_key")),
            on_max_iterations=str(data.get("on_max_iterations", "stop")),
        )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


@register_process_type
class MapProcess(FlowProcess):
    """Run a body once per item of a list, collecting the outputs.

    Each iteration runs against its **own copy** of the results mapping, so an
    iteration rebinding a key cannot affect the next one, and only the
    collected list is written back to the outer results. The copy is shallow:
    a body that *mutates* a nested list or dict in place still affects the
    outer value and later iterations. Rebind rather than mutate.

    JSON::

        {
          "process_type": "map",
          "process_name": "per_topic",
          "items": "topics",
          "item_var": "topic",
          "collect": "draft",
          "result_key": "drafts",
          "processes": [ ...body... ]
        }

    ``items`` is a sandboxed expression, so ``"topics"`` and
    ``"topics[:3]"`` are both valid. When ``collect`` names a results key, the
    output is the list of that key's value per iteration. When it is omitted,
    each entry is a dict of whatever keys that iteration added.
    """

    process_type = "map"

    def __init__(
        self,
        items: str | SafeExpression,
        process_name: str = "MapProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
        processes: Sequence[Process] | None = None,
        steps: Sequence[ProcessStep] | None = None,
        item_var: str = "item",
        index_var: str = "index",
        collect: str | None = None,
        result_key: str | None = None,
    ):
        """Initialize a map process.

        Args:
            items: Sandboxed expression selecting the list to iterate.
            process_name: Name identifier for the process.
            agent: Default agent lent to children that have none.
            agent_name: Name of a declared agent, for JSON round-tripping.
            processes: Body processes run per item.
            steps: Convenience steps wrapped into a sequential body process.
            item_var: Results key holding the current item.
            index_var: Results key holding the zero-based index.
            collect: Results key gathered from each iteration.
            result_key: Where the collected list lands. Defaults to
                ``"<process_name>_results"``.
        """
        super().__init__(process_name, agent, agent_name)
        self.items = items if isinstance(items, SafeExpression) else SafeExpression(items)
        self.processes = _build_body(
            processes, steps, f"{process_name}_body", agent
        )
        self.item_var = item_var
        self.index_var = index_var
        self.collect = collect
        self.result_key = result_key or f"{process_name}_results"

    def primary_body(self) -> list[Process]:
        return self.processes

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """Run the body once per item, collecting outputs into a list."""

        values = self.items.evaluate(results)
        if (
            isinstance(values, (str, bytes, Mapping))
            or not isinstance(values, Iterable)
        ):
            raise PipelineExecutionError(
                f"Map process '{self.process_name}' expected "
                f"{self.items.source!r} to yield a list, got "
                f"{type(values).__name__}."
            )
        # Materialize once: a generator would be empty on a second run, which
        # happens whenever a map sits inside a loop or the pipeline is reused.
        values = list(values)

        collected: list[Any] = []
        for index, item in enumerate(values):
            scope = results.copy()
            scope.vars[self.item_var] = item
            scope.vars[self.index_var] = index
            scope = self.run_child_processes(self.processes, scope)

            if self.collect is not None:
                if self.collect not in scope.outputs:
                    available = ", ".join(sorted(scope.outputs)) or "<none>"
                    raise PipelineExecutionError(
                        f"Map process '{self.process_name}' collects "
                        f"'{self.collect}', which iteration {index} did not "
                        f"produce. Available outputs: {available}."
                    )
                collected.append(scope.outputs[self.collect])
            else:
                # With outputs in their own section, "what did this iteration
                # produce" is exact rather than inferred: new or rebound keys
                # in `outputs`, and nothing from inputs or vars.
                collected.append(
                    {
                        key: value
                        for key, value in scope.outputs.items()
                        if key not in results.outputs
                        or value is not results.outputs[key]
                    }
                )

        results.outputs[self.result_key] = collected
        return results

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this map process."""

        data = self._base_dict()
        data["items"] = self.items.source
        data["processes"] = processes_to_config(self.processes, tool_registry)
        if self.item_var != "item":
            data["item_var"] = self.item_var
        if self.index_var != "index":
            data["index_var"] = self.index_var
        if self.collect is not None:
            data["collect"] = self.collect
        if self.result_key != f"{self.process_name}_results":
            data["result_key"] = self.result_key
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "MapProcess":
        """Restore a map process from JSON."""

        process_name, agent_name = cls._read_identity(data, "MapProcess")
        items = data.get("items")
        if items is None:
            raise PipelineSerializationError(
                f"Map process '{process_name}' is missing 'items'."
            )
        agent = context.agent_for_process(process_name, agent_name)
        nested = context.nested(process_name, agent)

        return cls(
            items=str(items),
            process_name=process_name,
            agent=agent,
            agent_name=agent_name,
            processes=_body_from_config(data, nested, process_name, agent),
            item_var=str(data.get("item_var", "item")),
            index_var=str(data.get("index_var", "index")),
            collect=_optional_str(data.get("collect")),
            result_key=_optional_str(data.get("result_key")),
        )


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


@register_process_type
class ParallelProcess(FlowProcess):
    """Run independent branches concurrently and merge their results.

    Each branch runs against its own copy of the results mapping in a worker
    thread; afterwards the keys each branch *added or changed* are merged back.

    .. warning::

        An agent instance is not thread-safe: ``ChatToolAgent`` keeps a
        ``last_messages_buffer`` on ``self``, so two branches sharing one agent
        will interleave their transcripts. The ``.response`` text each step
        stores stays correct, but ``ChatResponse.messages`` does not. Give each
        branch its own agent when the transcript matters; this process warns
        once if branches would share one.

    JSON::

        {
          "process_type": "parallel",
          "process_name": "research",
          "branches": [ ...one process per branch... ],
          "max_workers": 4,
          "on_conflict": "error"
        }

    Under ``on_conflict="section"`` a contested output moves into a
    sub-section named for its branch, so it is addressed by structure rather
    than by a mangled name::

        {outputs/news/draft}        in a prompt template
        outputs['news']['draft']    in a condition

    Any value the key already held at the top of ``outputs`` is untouched.
    """

    process_type = "parallel"

    ON_CONFLICT = ("error", "last_wins", "section")

    def __init__(
        self,
        branches: Sequence[Process] | None = None,
        process_name: str = "ParallelProcess",
        agent: BaseToolAgent = None,
        agent_name: str | None = None,
        max_workers: int | None = None,
        on_conflict: str = "error",
    ):
        """Initialize a parallel process.

        Args:
            branches: Processes run concurrently, one per branch.
            process_name: Name identifier for the process.
            agent: Default agent lent to branches that have none. See the
                thread-safety warning above.
            agent_name: Name of a declared agent, for JSON round-tripping.
            max_workers: Thread pool size. Defaults to the branch count.
            on_conflict: What to do when two branches write the same output
                with different values: ``"error"``, ``"last_wins"``, or
                ``"section"`` (give each branch its own sub-section of
                ``outputs``).
        """
        super().__init__(process_name, agent, agent_name)

        if on_conflict not in self.ON_CONFLICT:
            raise ValueError(
                f"on_conflict must be one of {self.ON_CONFLICT}, "
                f"got {on_conflict!r}."
            )
        self.branches = list(branches or [])
        self.max_workers = max_workers
        self.on_conflict = on_conflict

    def primary_body(self) -> list[Process]:
        return self.branches

    def run_process(self, results: PipelineResults) -> PipelineResults:
        """Run every branch concurrently and merge the results."""

        if not self.branches:
            return results

        for branch in self.branches:
            self.lend_agent(branch)
        self._warn_on_shared_agents()

        base = results

        def run_branch(
            indexed: tuple[int, Process]
        ) -> tuple[int, PipelineResults]:
            index, branch = indexed
            return index, branch.run_process(base.copy())

        workers = self.max_workers or len(self.branches)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(run_branch, enumerate(self.branches)))

        # Merge in branch order, not completion order, so a run is reproducible.
        outcomes.sort(key=lambda outcome: outcome[0])
        return self._merge(base, outcomes)

    def branch_labels(self) -> list[str]:
        """Return a unique, template-safe label for each branch.

        Labels are used to qualify keys under ``on_conflict="prefix"``. They
        must contain only word characters, because ``MessageTemplate``
        substitutes on ``\\w+`` and the condition sandbox rejects anything with
        a dot as attribute access — a qualified key that no prompt or condition
        could read would be worse than useless.
        """

        raw_names = [
            re.sub(r"\W+", "_", branch.process_name or "").strip("_")
            or f"branch{index}"
            for index, branch in enumerate(self.branches)
        ]
        counts = Counter(raw_names)
        return [
            name if counts[name] == 1 else f"{name}_{index}"
            for index, name in enumerate(raw_names)
        ]

    def _warn_on_shared_agents(self) -> None:
        """Warn when two branches would run on the same agent instance."""

        seen: set[int] = set()
        for branch in self.branches:
            if branch.agent is None:
                continue
            if id(branch.agent) in seen:
                warnings.warn(
                    f"Parallel process '{self.process_name}' runs multiple "
                    "branches on the same agent instance. Step responses stay "
                    "correct, but the agent's message buffer is shared across "
                    "threads, so ChatResponse.messages will interleave. Give "
                    "each branch its own agent when the transcript matters.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return
            seen.add(id(branch.agent))

    def _merge(
        self,
        base: PipelineResults,
        outcomes: Sequence[tuple[int, PipelineResults]],
    ) -> PipelineResults:
        """Merge each branch's added or changed outputs into one result.

        Claims are gathered first and resolved second, so the outcome does not
        depend on which branch happens to finish when. A key claimed by
        several branches with *equal* values is agreement, not a conflict.
        """

        labels = self.branch_labels()
        merged = base.copy()
        claims: dict[str, list[tuple[str, Any]]] = {}

        for index, scope in outcomes:
            for key, value in scope.outputs.items():
                if key in base.outputs and base.outputs[key] is value:
                    continue
                claims.setdefault(key, []).append((labels[index], value))

        for key, entries in claims.items():
            if len(entries) == 1 or _all_equal(value for _, value in entries):
                merged.outputs[key] = entries[-1][1]
                continue

            owners = ", ".join(f"'{label}'" for label, _ in entries)
            if self.on_conflict == "error":
                raise PipelineExecutionError(
                    f"Parallel process '{self.process_name}': branches "
                    f"{owners} wrote different values for output '{key}'. "
                    "Rename the step, or set on_conflict to 'section' or "
                    "'last_wins'."
                )
            if self.on_conflict == "last_wins":
                merged.outputs[key] = entries[-1][1]
            else:  # section
                # Each claimant gets its own sub-section of `outputs`, so the
                # value is addressed by structure rather than by a mangled
                # name: {outputs/news/draft} in a prompt, outputs['news']
                # in a condition. Any value the key already held is untouched.
                for label, value in entries:
                    section = merged.outputs.setdefault(label, {})
                    if not isinstance(section, dict):
                        raise PipelineExecutionError(
                            f"Parallel process '{self.process_name}' cannot "
                            f"create a section for branch '{label}': output "
                            f"'{label}' already holds a non-mapping value."
                        )
                    section[key] = value

        return merged

    def to_dict(
        self,
        tool_registry: PipelineToolRegistry | None = None,
    ) -> dict[str, Any]:
        """Serialize this parallel process."""

        data = self._base_dict()
        data["branches"] = processes_to_config(self.branches, tool_registry)
        if self.max_workers is not None:
            data["max_workers"] = self.max_workers
        if self.on_conflict != "error":
            data["on_conflict"] = self.on_conflict
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        context: PipelineLoadContext,
    ) -> "ParallelProcess":
        """Restore a parallel process from JSON."""

        process_name, agent_name = cls._read_identity(data, "ParallelProcess")
        agent = context.agent_for_process(process_name, agent_name)
        nested = context.nested(process_name, agent)
        _reject_steps(data, process_name, "'branches'")
        max_workers = data.get("max_workers")

        return cls(
            branches=processes_from_config(
                data.get("branches", data.get("processes")),
                nested,
                field_name="branches",
            ),
            process_name=process_name,
            agent=agent,
            agent_name=agent_name,
            max_workers=None if max_workers is None else int(max_workers),
            on_conflict=str(data.get("on_conflict", "error")),
        )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _all_equal(values: Iterable[Any]) -> bool:
    """Return whether every value compares equal to the first.

    Values from independent branches are distinct objects, so identity is not
    enough: two branches producing the same text are agreeing, not colliding.
    A type that refuses comparison counts as unequal.
    """

    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    for value in iterator:
        try:
            if not bool(first == value):
                return False
        except Exception:
            return False
    return True

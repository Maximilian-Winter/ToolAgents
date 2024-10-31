from enum import Enum
from typing import Callable, List, Dict, Set
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Optional

# Define our basic types
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
ConfigType = TypeVar('ConfigType')


@dataclass
class ExecutionContext:
    """The fundamental context for any execution"""
    trace_id: str
    timestamp: float
    metadata: dict
    parent_context: Optional['ExecutionContext'] = None


@dataclass
class ProcessResult(Generic[OutputType]):
    """The fundamental result of any process"""
    output: OutputType
    context: ExecutionContext
    metrics: dict
    errors: list


class BaseProcess(ABC, Generic[InputType, OutputType, ConfigType]):
    """The fundamental building block of all chains"""

    def __init__(self, config: ConfigType):
        self.config = config
        self._validate_config()

    @abstractmethod
    async def transform(self,
                        input_data: InputType,
                        context: ExecutionContext) -> ProcessResult[OutputType]:
        """Process input to output"""
        pass

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate process configuration"""
        pass

    async def pre_process(self, input_data: InputType, context: ExecutionContext) -> InputType:
        """Optional pre-processing hook"""
        return input_data

    async def post_transform(self, result: ProcessResult) -> ProcessResult:
        """Optional post-processing hook"""
        return result


def topological_sort(processes: List[BaseProcess],
                     dependencies: Dict[BaseProcess, Set[BaseProcess]]) -> List[BaseProcess]:
    """Perform topological sort on processes based on their dependencies"""
    # Create a copy of the dependency graph
    graph = {p: dependencies.get(p, set()).copy() for p in processes}

    # Find all nodes with no dependencies
    sorted_nodes = []
    no_deps = [p for p in processes if not graph[p]]

    while no_deps:
        # Take a node with no dependencies
        node = no_deps.pop(0)
        sorted_nodes.append(node)

        # Remove this node from others' dependencies
        for deps in graph.values():
            if node in deps:
                deps.remove(node)

        # Check for new nodes with no dependencies
        no_deps.extend(
            p for p in processes
            if p not in sorted_nodes
            and p not in no_deps
            and not graph[p]
        )

    # Check for cycles
    if any(graph.values()):
        raise ValueError("Dependency cycle detected")

    return sorted_nodes

def weighted_vote(results: List[ProcessResult],
                  weights: List[float]) -> ProcessResult:
    """Aggregate results using weighted voting"""
    if not results:
        return ProcessResult(None, None, {}, ["No results to vote on"])

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]

    # Combine metrics and errors
    combined_metrics = {}
    all_errors = []

    for result, weight in zip(results, normalized_weights):
        # Update metrics with weights
        for metric, value in result.metrics.items():
            if isinstance(value, (int, float)):
                combined_metrics[metric] = combined_metrics.get(metric, 0) + value * weight
        all_errors.extend(result.errors)

    # For the output, use the result with highest weight
    best_result = results[normalized_weights.index(max(normalized_weights))]

    return ProcessResult(
        output=best_result.output,
        context=best_result.context,
        metrics=combined_metrics,
        errors=all_errors
    )
class ChainableProcess(BaseProcess[InputType, OutputType, ConfigType]):
    """A process that can be chained with others"""

    def __init__(self, config: ConfigType, next_transformer: Optional[BaseProcess] = None):
        super().__init__(config)
        self.next = next_transformer

    async def transform_and_chain(self,
                                  input_data: InputType,
                                  context: ExecutionContext) -> ProcessResult:
        result = await self.transform(input_data, context)

        if self.next and not result.errors:
            next_result = await self.next.transform(result.output, context)
            result.output = next_result.output
            result.metrics.update(next_result.metrics)
            result.errors.extend(next_result.errors)

        return result

    async def transform(self, input_data: InputType, context: ExecutionContext) -> ProcessResult[OutputType]:
        pass

    def _validate_config(self) -> None:
        pass


class ParallelProcess(BaseProcess[InputType, OutputType, ConfigType]):
    """A process that processes inputs in parallel"""

    def __init__(self, config: ConfigType, transformers: list[BaseProcess]):
        super().__init__(config)
        self.transformers = transformers

    def _validate_config(self) -> None:
        pass

    async def transform(self,
                        input_data: InputType,
                        context: ExecutionContext) -> ProcessResult[OutputType]:
        tasks = [t.transform(input_data, context) for t in self.transformers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._aggregate_results(results, context)


class FeedbackProcess(ChainableProcess[InputType, OutputType, ConfigType]):
    """A transformer that can learn from its outputs"""

    def __init__(self, config: ConfigType, feedback_policy: callable):
        super().__init__(config)
        self.feedback_policy = feedback_policy
        self.learning_history = []

    async def transform(self,
                        input_data: InputType,
                        context: ExecutionContext) -> ProcessResult[OutputType]:
        result = await super().transform(input_data, context)
        feedback = self.feedback_policy(result)
        self.learning_history.append(feedback)
        return result


@dataclass
class StageConfig:
    """Configuration for an execution stage"""
    timeout: float = 30.0
    max_retries: int = 3
    error_policy: str = "fail_fast"
    retry_delay: float = 1.0
    stage_name: str = "unnamed_stage"


class ExecutionStage:
    """Represents a single stage in the execution plan"""

    def __init__(self,
                 processes: List[BaseProcess],
                 config: StageConfig,
                 aggregation_strategy: Optional[Callable] = None):
        self.processes = processes
        self.config = config
        self.error_policy = config.error_policy
        self.aggregation_strategy = aggregation_strategy or self._default_aggregation

    async def execute(self, input_data: Any, context: ExecutionContext) -> ProcessResult:
        """Execute all processes in this stage"""
        try:
            async with asyncio.timeout(self.config.timeout):
                results = await asyncio.gather(
                    *[p.transform(input_data, context) for p in self.processes],
                    return_exceptions=True
                )

                # Handle exceptions
                errors = [r for r in results if isinstance(r, Exception)]
                if errors and self.error_policy == "fail_fast":
                    raise errors[0]

                # Filter out exceptions if we're continuing
                valid_results = [r for r in results if not isinstance(r, Exception)]
                return self.aggregation_strategy(valid_results, context)

        except asyncio.TimeoutError:
            return ProcessResult(
                output=None,
                context=context,
                metrics={"status": "timeout"},
                errors=[f"Stage {self.config.stage_name} timed out"]
            )

    @staticmethod
    def _default_aggregation(results: List[ProcessResult],
                             context: ExecutionContext) -> ProcessResult:
        """Default strategy to aggregate results"""
        if not results:
            return ProcessResult(None, context, {}, ["No valid results"])

        # Combine metrics and errors
        combined_metrics = {}
        all_errors = []
        for r in results:
            combined_metrics.update(r.metrics)
            all_errors.extend(r.errors)

        # Use the last valid output as the final output
        return ProcessResult(
            output=results[-1].output,
            context=context,
            metrics=combined_metrics,
            errors=all_errors
        )


class ExecutionPlan:
    """Represents an optimized plan for executing processes"""

    def __init__(self):
        self.stages: List[ExecutionStage] = []
        self.error_handlers: Dict[str, Callable] = {}

    async def execute(self, input_data: Any, context: ExecutionContext) -> ProcessResult:
        results = []
        for stage in self.stages:
            try:
                stage_result = await stage.execute(input_data, context)
                results.append(stage_result)
            except Exception as e:
                if stage.error_policy == "retry":
                    results.append(await self._handle_retry(stage, input_data, context))
                elif stage.error_policy == "fallback":
                    results.append(await self._execute_fallback(stage, input_data, context))
                else:
                    raise e

        return self._aggregate_results(results)


class CompositionType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    FEEDBACK = "feedback"
    DAG = "directed_acyclic_graph"
    DYNAMIC = "dynamic"
    VOTING = "voting"


@dataclass
class CompositionRule:
    """Defines how transformers should be composed"""
    type: CompositionType
    condition: Optional[Callable] = None
    aggregator: Optional[Callable] = None
    max_retries: int = 3
    timeout: float = 30.0
    error_policy: str = "fail_fast"


class ParallelProcess:
    def _aggregate_results(self,
                           results: List[ProcessResult],
                           context: ExecutionContext) -> ProcessResult:
        """Aggregate results from parallel execution"""
        valid_results = [r for r in results if isinstance(r, ProcessResult)]
        exceptions = [r for r in results if isinstance(r, Exception)]

        combined_metrics = {}
        all_errors = []

        for result in valid_results:
            combined_metrics.update(result.metrics)
            all_errors.extend(result.errors)

        if exceptions:
            all_errors.extend([str(e) for e in exceptions])

        # Combine outputs - this is just one strategy, could be customized
        combined_output = [r.output for r in valid_results]

        return ProcessResult(
            output=combined_output,
            context=context,
            metrics=combined_metrics,
            errors=all_errors
        )


class AdvancedCompositionStrategy:
    def _analyze_dependencies(self, process: BaseProcess) -> Set[BaseProcess]:
        """Analyze process dependencies"""
        dependencies = set()

        # If it's a chainable process, add its next process
        if isinstance(process, ChainableProcess) and process.next:
            dependencies.add(process.next)

        # If it's a composite process, add all its transformers
        if isinstance(process, CompositeProcess):
            dependencies.update(process.transformers)

        return dependencies

    async def _execute_plan(self,
                            plan: ExecutionPlan,
                            input_data: Any,
                            context: ExecutionContext) -> ProcessResult:
        """Execute the plan"""
        start_time = asyncio.get_event_loop().time()

        try:
            result = await plan.execute(input_data, context)

            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self.metrics[context.trace_id] = {
                "execution_time": execution_time,
                "status": "success"
            }

            # Cache result if needed
            if self._should_cache(input_data, context):
                self.cache[self._cache_key(input_data)] = result

            return result

        except Exception as e:
            self.metrics[context.trace_id] = {
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "status": "error",
                "error": str(e)
            }
            raise


class CompositeProcess(BaseProcess[InputType, OutputType, ConfigType]):
    """A process that combines multiple processes in flexible ways"""

    def __init__(self, config: ConfigType, composition_strategy: callable):
        super().__init__(config)
        self.composition_strategy = composition_strategy
        self.transformers = []

    def add_transformer(self, transformer: BaseProcess):
        self.transformers.append(transformer)

    async def transform(self,
                        input_data: InputType,
                        context: ExecutionContext) -> ProcessResult[OutputType]:
        return await self.composition_strategy(self.transformers, input_data, context)


class DynamicCompositionBuilder:
    """Builds complex composition patterns dynamically"""

    def __init__(self):
        self.patterns = {}
        self.active_transformers = set()

    def add_pattern(self, name: str, pattern: Callable):
        """Add a new composition pattern"""
        self.patterns[name] = pattern

    def create_composition(self, pattern_name: str, transformers: List[BaseProcess]) -> CompositeProcess:
        """Create a new composition using a named pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        pattern = self.patterns[pattern_name]
        return pattern(transformers)


# Now let's define some powerful composition patterns!
class CompositionPatterns:
    """A collection of advanced composition patterns"""

    @staticmethod
    async def voting(transformers: List[BaseProcess],
                     input_data: Any,
                     context: ExecutionContext,
                     weight_strategy: Callable = None) -> ProcessResult:
        """Execute transformers and aggregate results by voting"""
        results = await asyncio.gather(*[t.transform(input_data, context) for t in transformers])
        weights = [weight_strategy(t) if weight_strategy else 1.0 for t in transformers]
        return weighted_vote(results, weights)

    @staticmethod
    async def dag_execution(transformers: List[BaseProcess],
                            dependencies: Dict,
                            input_data: Any,
                            context: ExecutionContext) -> ProcessResult:
        """Execute transformers as a directed acyclic graph"""
        execution_order = topological_sort(transformers, dependencies)
        results = {}

        for transformer in execution_order:
            deps_results = {dep: results[dep] for dep in dependencies[transformer]}
            results[transformer] = await transformer.transform(
                {**input_data, **deps_results}, context)

        return results[execution_order[-1]]

    @staticmethod
    async def adaptive(transformers: List[BaseProcess],
                       input_data: Any,
                       context: ExecutionContext,
                       performance_monitor: Any) -> ProcessResult:
        """Adaptively choose and execute transformers based on performance"""
        best_transformer = performance_monitor.select_best_transformer(transformers, input_data)
        return await best_transformer.transform(input_data, context)

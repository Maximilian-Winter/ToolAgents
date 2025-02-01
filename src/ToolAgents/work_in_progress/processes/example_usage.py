import asyncio
from dataclasses import dataclass
import time
import random

from ToolAgents.work_in_progress.processes.processes import BaseProcess, ExecutionContext, ProcessResult, ChainableProcess, \
    CompositionPatterns, ExecutionStage, StageConfig, ExecutionPlan, AdvancedCompositionStrategy


@dataclass
class SimpleConfig:
    name: str
    delay: float = 0.1
    error_rate: float = 0.0


class TextProcessResult:
    def __init__(self, text: str):
        self.text = text


class TextProcessor(BaseProcess[str, str, SimpleConfig]):
    """A simple text processing element"""

    async def transform(self, input_data: str, context: ExecutionContext) -> ProcessResult[str]:
        # Simulate processing time
        await asyncio.sleep(self.config.delay)

        # Simulate random errors
        if random.random() < self.config.error_rate:
            raise Exception(f"Random error in {self.config.name}")

        output = f"{self.config.name} processed: {input_data}"

        return ProcessResult(
            output=output,
            context=context,
            metrics={"processing_time": self.config.delay},
            errors=[]
        )

    def _validate_config(self) -> None:
        assert self.config.name, "Name must be provided"
        assert self.config.delay >= 0, "Delay must be non-negative"


# Now let's create some example usage patterns!

async def basic_chain_example():
    """Example of basic process chaining"""
    # Create processes
    p1 = TextProcessor(SimpleConfig("Uppercase", delay=0.1))
    p2 = TextProcessor(SimpleConfig("Reverse", delay=0.2))
    p3 = TextProcessor(SimpleConfig("AddPrefix", delay=0.1))

    # Chain them
    chain = ChainableProcess(SimpleConfig("chain"))
    chain.next = p1
    p1.next = p2
    p2.next = p3

    # Create context
    context = ExecutionContext(
        trace_id="example_1",
        timestamp=time.time(),
        metadata={}
    )

    # Execute
    result = await chain.transform_and_chain("hello world", context)
    print(f"Chain result: {result.output}")
    print(f"Metrics: {result.metrics}")


async def parallel_voting_example():
    """Example of parallel execution with voting"""
    # Create multiple processors with different behaviors
    processors = [
        TextProcessor(SimpleConfig("Fast", delay=0.1)),
        TextProcessor(SimpleConfig("Slow", delay=0.3)),
        TextProcessor(SimpleConfig("Medium", delay=0.2))
    ]

    # Create weights based on speed
    weights = [1.0, 0.3, 0.6]  # Favor faster processors

    # Create context
    context = ExecutionContext(
        trace_id="parallel_example",
        timestamp=time.time(),
        metadata={}
    )

    # Execute with voting
    result = await CompositionPatterns.voting(
        processors, "test input", context,
        weight_strategy=lambda t: weights[processors.index(t)]
    )

    print(f"Voting result: {result.output}")


async def dag_workflow_example():
    """Example of DAG-based workflow"""
    # Create processors
    p1 = TextProcessor(SimpleConfig("Extract", delay=0.1))
    p2 = TextProcessor(SimpleConfig("Transform", delay=0.2))
    p3 = TextProcessor(SimpleConfig("Load", delay=0.1))
    p4 = TextProcessor(SimpleConfig("Validate", delay=0.1))

    # Define dependencies
    dependencies = {
        p1: set(),  # No dependencies
        p2: {p1},  # Depends on p1
        p3: {p2},  # Depends on p2
        p4: {p1, p2}  # Depends on p1 and p2
    }

    context = ExecutionContext(
        trace_id="dag_example",
        timestamp=time.time(),
        metadata={}
    )

    # Execute DAG
    result = await CompositionPatterns.dag_execution(
        [p1, p2, p3, p4], dependencies, "input data", context
    )

    print(f"DAG result: {result.output}")


async def error_handling_example():
    """Example of error handling and retries"""
    # Create a processor that sometimes fails
    error_prone = TextProcessor(
        SimpleConfig("Unreliable", delay=0.1, error_rate=0.5)
    )

    # Create stage with retry configuration
    stage = ExecutionStage(
        processes=[error_prone],
        config=StageConfig(
            timeout=1.0,
            max_retries=3,
            error_policy="retry",
            retry_delay=0.1,
            stage_name="error_prone_stage"
        )
    )

    # Create execution plan
    plan = ExecutionPlan()
    plan.stages.append(stage)

    context = ExecutionContext(
        trace_id="error_example",
        timestamp=time.time(),
        metadata={}
    )

    try:
        result = await plan.execute("test input", context)
        print(f"Success after retries: {result.output}")
    except Exception as e:
        print(f"Failed after all retries: {str(e)}")


async def advanced_composition_example():
    """Example of advanced composition with caching"""
    strategy = AdvancedCompositionStrategy()

    # Create processes
    processors = [
        TextProcessor(SimpleConfig("Cache1", delay=0.1)),
        TextProcessor(SimpleConfig("Cache2", delay=0.2))
    ]

    # Create execution plan
    plan = ExecutionPlan()
    plan.stages.append(
        ExecutionStage(
            processes=processors,
            config=StageConfig(stage_name="cached_stage")
        )
    )

    # Execute with caching
    context = ExecutionContext(
        trace_id="cache_example",
        timestamp=time.time(),
        metadata={}
    )

    # First execution
    result1 = await strategy._execute_plan(plan, "test input", context)
    print(f"First execution: {result1.output}")

    # Second execution (should be cached)
    result2 = await strategy._execute_plan(plan, "test input", context)
    print(f"Second execution (cached): {result2.output}")

    print(f"Cache metrics: {strategy.metrics}")


# Run all examples
async def run_examples():
    print("=== Basic Chain Example ===")
    await basic_chain_example()

    print("\n=== Parallel Voting Example ===")
    await parallel_voting_example()

    print("\n=== DAG Workflow Example ===")
    await dag_workflow_example()

    print("\n=== Error Handling Example ===")
    await error_handling_example()

    print("\n=== Advanced Composition Example ===")
    await advanced_composition_example()


# Run everything
if __name__ == "__main__":
    asyncio.run(run_examples())

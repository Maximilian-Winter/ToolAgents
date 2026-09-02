"""A pipeline that describes its own endpoints and its own control flow.

Nothing about the workflow lives in this file: the JSON next to it declares the
providers, the models, the sampling settings, and the shape of the run —
map, parallel, loop, and conditional. This script only supplies the inputs.

Run with an OpenRouter key in the environment::

    OPENROUTER_API_KEY=... python example_flow_control_pipeline.py

To point the same workflow at a different endpoint, edit `base_url` and
`api_key_env` in the JSON. No Python change is needed.

Prompts in the JSON address results by section — `{outputs/draft}`,
`{inputs/audience}`, `{vars/topic}` — so a step can never collide with a
caller argument or with a loop's counter.
"""

from pathlib import Path

from dotenv import load_dotenv

from ToolAgents.pipelines import Pipeline

load_dotenv()

pipeline_path = Path(__file__).with_name("flow_control_pipeline.json")

# No `default_agent` argument: the JSON's own `agents` block builds the
# providers, reading each API key from the environment variable it names.
pipeline = Pipeline.load_from_json(pipeline_path)

results = pipeline.run_pipeline(
    topics=["otter tool use", "otter social structure"],
    audience="curious non-specialists",
)

# Results come back in sections. A bare name still resolves (innermost first:
# vars, then outputs, then inputs), but addressing the section is unambiguous.
print("=" * 70)
print("TITLE   :", results["outputs/title"])
print("REVIEWS :", results["outputs/refine_iterations"], "revision(s)")
print("APPROVED:", results["outputs/passed_review"])
print("AUDIENCE:", results["inputs/audience"])
print("=" * 70)
print(results["outputs/draft"])
print("-" * 70)
print("SUMMARY :", results["outputs/summary"])
print("-" * 70)
print("sections:", {name: sorted(values) for name, values in results.to_dict().items()})

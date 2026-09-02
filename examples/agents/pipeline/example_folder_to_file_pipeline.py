"""A workflow that reads a folder and writes a file, with no I/O in Python.

The JSON next to this file declares everything: which folder to read, how to
chunk it, which model summarizes each chunk, where the result is written, and
what gets printed. This script supplies three paths and nothing else.

Run with an OpenRouter key in the environment::

    OPENROUTER_API_KEY=... python example_folder_to_file_pipeline.py

Note `allow_writes=True`. Reading is permitted by default, but a pipeline
document may not write files or make HTTP requests unless the caller says so —
loading a file should not let it reach outside the process on its own.
"""

from pathlib import Path

from dotenv import load_dotenv

from ToolAgents.pipelines import Pipeline

load_dotenv()

here = Path(__file__).parent
pipeline = Pipeline.load_from_json(
    here / "folder_to_file_pipeline.json",
    allow_writes=True,
)

results = pipeline.run_pipeline(
    notes_dir=str(here / "notes"),
    out_dir=str(here / "out"),
    title="Otters, briefly",
)

print("=" * 70)
print("chunks read :", len(results["inputs/chunks"]))
print("points made :", len(results["outputs/points"]))
print("written to  :", results["outputs/digest_path"])
print("=" * 70)
print(results["outputs/digest"])
